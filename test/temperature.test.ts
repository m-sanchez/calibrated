import { test } from 'node:test';
import assert from 'node:assert/strict';
import { calibrationError } from '../src/metrics.ts';
import { fitTemperature, nll, softmax, toPredictions } from '../src/temperature.ts';
import type { LogitSample } from '../src/temperature.ts';

function seeded(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let z = s;
    z ^= z >>> 16;
    z = Math.imul(z, 0x21f0aaad);
    z ^= z >>> 15;
    z = Math.imul(z, 0x735a2d97);
    z ^= z >>> 15;
    return (z >>> 0) / 0x100000000;
  };
}

/** An overconfident 3-class model: the predicted class carries a large
 * margin (so confidence is near 1), but it is only right `accuracy` of the
 * time. That is exactly the miscalibration temperature scaling exists to
 * fix. */
function overconfident(n: number, accuracy: number, margin: number, seed = 3): LogitSample[] {
  const rand = seeded(seed);
  const samples: LogitSample[] = [];
  for (let i = 0; i < n; i++) {
    const label = Math.floor(rand() * 3);
    const predicted = rand() < accuracy ? label : (label + 1 + Math.floor(rand() * 2)) % 3;
    const logits = [rand(), rand(), rand()];
    logits[predicted] += margin; // a big, overconfident margin
    samples.push({ logits, label });
  }
  return samples;
}

test('softmax sums to one and is stable with huge logits', () => {
  const p = softmax([1000, 1001, 999]);
  assert.ok(Math.abs(p.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(p.every((x) => x >= 0 && x <= 1));
});

test('softmax rejects a non-positive temperature', () => {
  assert.throws(() => softmax([1, 2], 0), RangeError);
});

test('fitting temperature on an overconfident model returns T > 1 and lowers NLL', () => {
  const samples = overconfident(2000, 0.7, 8);
  const fit = fitTemperature(samples);
  assert.ok(fit.temperature > 1, `expected softening, got T=${fit.temperature}`);
  assert.ok(fit.improved);
  assert.ok(fit.nllAfter < fit.nllBefore, `nll ${fit.nllBefore} -> ${fit.nllAfter}`);
});

test('temperature scaling reduces calibration error without moving accuracy', () => {
  const samples = overconfident(2000, 0.7, 8);
  const fit = fitTemperature(samples);

  const before = toPredictions(samples, 1);
  const after = toPredictions(samples, fit.temperature);
  const eceBefore = calibrationError(before, 15).ece;
  const eceAfter = calibrationError(after, 15).ece;

  assert.ok(eceAfter < eceBefore * 0.6, `ece ${eceBefore.toFixed(3)} -> ${eceAfter.toFixed(3)}`);
  const accBefore = before.filter((p) => p.correct).length;
  const accAfter = after.filter((p) => p.correct).length;
  assert.equal(accBefore, accAfter, 'the argmax, and so accuracy, is unchanged');
});

test('an already-calibrated model gets a temperature near 1', () => {
  // Calibrated by construction: draw the model's own distribution, then
  // sample the label FROM it. Its stated probabilities are then honest, so
  // there is nothing for scaling to fix.
  const rand = seeded(9);
  const samples: LogitSample[] = [];
  for (let i = 0; i < 4000; i++) {
    const logits = [rand() * 4, rand() * 4, rand() * 4];
    const probs = softmax(logits);
    let u = rand();
    let label = 0;
    for (let c = 0; c < probs.length; c++) {
      u -= probs[c];
      if (u <= 0) {
        label = c;
        break;
      }
    }
    samples.push({ logits, label });
  }
  const fit = fitTemperature(samples);
  assert.ok(Math.abs(fit.temperature - 1) < 0.35, `T=${fit.temperature}`);
});

test('nll is finite even when a label has vanishing probability', () => {
  const samples: LogitSample[] = [{ logits: [50, -50], label: 1 }];
  assert.ok(Number.isFinite(nll(samples, 1)));
});
