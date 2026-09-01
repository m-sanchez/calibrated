import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bin } from '../src/binning.ts';
import { brier, calibrationError } from '../src/metrics.ts';
import type { Prediction } from '../src/binning.ts';

function seededUnit(seed: number): () => number {
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

/** A band of `count` predictions all at `confidence`, of which exactly
 * `accuracy * count` are correct. Lets us build sets with a known ECE. */
function band(confidence: number, accuracy: number, count: number): Prediction[] {
  const correct = Math.round(accuracy * count);
  return Array.from({ length: count }, (_, i) => ({ confidence, correct: i < correct }));
}

test('a perfectly calibrated set has ~zero calibration error', () => {
  const preds = [
    ...band(0.1, 0.1, 100),
    ...band(0.5, 0.5, 100),
    ...band(0.9, 0.9, 100)
  ];
  const { ece, mce } = calibrationError(preds, 10);
  assert.ok(ece < 1e-9, `ece should be ~0, got ${ece}`);
  assert.ok(mce < 1e-9);
});

test('an overconfident set surfaces its exact gap', () => {
  // 200 predictions at confidence 0.9, only 60% correct: gap 0.30
  const preds = band(0.9, 0.6, 200);
  const { ece, mce } = calibrationError(preds, 10);
  assert.ok(Math.abs(ece - 0.3) < 1e-9, `ece ${ece}`);
  assert.ok(Math.abs(mce - 0.3) < 1e-9);
});

test('MCE reports the single worst bin, not the average', () => {
  const preds = [...band(0.9, 0.85, 900), ...band(0.6, 0.1, 100)]; // one terrible bin
  const { ece, mce } = calibrationError(preds, 10);
  assert.ok(Math.abs(mce - 0.5) < 1e-9, `mce ${mce}`);
  assert.ok(ece < mce, 'ece is the weighted average, below the worst bin');
});

test('equal-mass binning keeps every bin populated when confidence is top-heavy', () => {
  const preds = [
    ...band(0.98, 0.9, 950),
    ...band(0.2, 0.2, 50) // a thin low-confidence tail
  ];
  const width = bin(preds, 10, 'equal-width').filter((b) => b.count > 0).length;
  const mass = bin(preds, 10, 'equal-mass');
  assert.ok(mass.every((b) => b.count > 0), 'no empty equal-mass bins');
  assert.ok(mass.length >= width, 'equal-mass resolves the crowded top better');
});

test('Brier score is the exact mean squared error of the probabilities', () => {
  const preds: Prediction[] = [
    { confidence: 1, correct: true }, // (1-1)^2 = 0
    { confidence: 0, correct: false }, // (0-0)^2 = 0
    { confidence: 0.5, correct: true } // (0.5-1)^2 = 0.25
  ];
  assert.ok(Math.abs(brier(preds).score - 0.25 / 3) < 1e-12);
});

test('Brier decomposition: a random-guess predictor has ~zero resolution', () => {
  // confidence and correctness drawn from independent streams: the
  // confidence carries no information about the outcome, so nothing is
  // resolved
  const rc = seededUnit(1);
  const rk = seededUnit(2);
  const preds = Array.from({ length: 4000 }, () => ({
    confidence: rc(),
    correct: rk() < 0.5
  }));
  const d = brier(preds, 10);
  assert.ok(d.resolution < 0.02, `resolution ${d.resolution}`);
  // score reconstructs from the three terms up to the binning approximation
  assert.ok(Math.abs(d.score - (d.reliability - d.resolution + d.uncertainty)) < 0.05);
});

test('binning rejects invalid inputs at the boundary', () => {
  assert.throws(() => bin([{ confidence: 1.5, correct: true }]), RangeError);
  assert.throws(() => bin([], 0), RangeError);
});

test('the report says how many bins actually carried data', () => {
  // 11 distinct confidences into 10 equal-mass bins: nine bins of one and
  // one of two. The requested count is echoed; the effective count is real.
  const preds: Prediction[] = Array.from({ length: 11 }, (_, i) => ({
    confidence: (i + 0.5) / 11,
    correct: i % 2 === 0
  }));
  const r = calibrationError(preds, 10, 'equal-mass');
  assert.equal(r.bins, 10, 'the requested bin count is echoed back');
  assert.equal(r.effectiveBins, bin(preds, 10, 'equal-mass').length);
  assert.equal(r.effectiveBins, 10);
  // equal-width leaves gaps: only the populated bins count
  const w = calibrationError(band(0.9, 0.6, 200), 10, 'equal-width');
  assert.equal(w.bins, 10);
  assert.equal(w.effectiveBins, 1);
});
