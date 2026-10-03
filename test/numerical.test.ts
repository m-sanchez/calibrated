import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fitTemperature, nll, softmax, toPredictions } from '../src/temperature.ts';

const sample = [{ logits: [0, 1], label: 1 }];

test('NLL matches independent closed-form answers without clipping', () => {
  assert.equal(nll([{ logits: [0, -1000], label: 1 }], 1), 1000);
  assert.ok(Math.abs(nll([{ logits: [0, 0], label: 1 }], 1) - Math.log(2)) < 1e-15);
  assert.ok(Math.abs(nll([{ logits: [Math.log(3), 0], label: 1 }], 1) - Math.log(4)) < 1e-15);
  assert.ok(nll([{ logits: [0, -40], label: 0 }], 1) > 0);
  const probabilities = softmax([Math.log(3), 0]);
  assert.ok(Math.abs(probabilities[0] - 0.75) < 1e-15);
  assert.ok(Math.abs(probabilities[1] - 0.25) < 1e-15);
});

test('centering handles huge equal logits and representable scaled differences', () => {
  assert.deepEqual(softmax([1e308, 1e308], 0.1), [0.5, 0.5]);
  assert.ok(Math.abs(nll([{ logits: [1e308, 1e308], label: 1 }], 0.1) - Math.log(2)) < 1e-15);
  assert.deepEqual(softmax([1e308, -1e308], 2), [1, 0]);
  assert.equal(nll([{ logits: [1e308, -1e308], label: 1 }], 2), 1e308);
  assert.throws(() => nll([{ logits: [1e308, -1e308], label: 1 }], 0.1), /numeric range/);
});

test('mean NLL stays finite when the unnormalized sum would overflow', () => {
  const rows = Array.from({ length: 8 }, () => ({ logits: [0, -1e308], label: 1 }));
  assert.equal(nll(rows, 1), 1e308);
});

test('raw-logit decisions survive probability rounding and exact ties', () => {
  const rows = [
    { logits: [0, 1e-16], label: 1 },
    { logits: [1, 1], label: 0 },
    { logits: [1, 1], label: 1 },
    { logits: [-1000, 1000], label: 1 }
  ];
  for (const temperature of [Number.MIN_VALUE, 0.1, 1, 2, 1e20, Number.MAX_VALUE]) {
    assert.deepEqual(toPredictions(rows, temperature).map((row) => row.correct), [true, true, false, true]);
  }
});

test('finite logit shifts preserve NLL and probabilities', () => {
  const logits = [-3, 1, 4];
  for (const offset of [-1e6, -100, 0, 100, 1e6]) {
    for (const temperature of [0.05, 0.5, 1, 10]) {
      const shifted = logits.map((value) => value + offset);
      assert.deepEqual(softmax(shifted, temperature), softmax(logits, temperature));
      assert.equal(nll([{ logits: shifted, label: 1 }], temperature), nll([{ logits, label: 1 }], temperature));
    }
  }
});

test('every temperature entry point validates temperature even without data', () => {
  for (const temperature of [0, -1, Infinity, -Infinity, NaN]) {
    assert.throws(() => softmax([0, 1], temperature), RangeError);
    assert.throws(() => nll([], temperature), RangeError);
    assert.throws(() => toPredictions([], temperature), RangeError);
  }
  assert.ok(Number.isNaN(nll([], 1)));
  assert.deepEqual(toPredictions([], 1), []);
  assert.throws(() => fitTemperature([]), /without samples/);
  assert.throws(() => softmax([]), RangeError);
});

test('all sample entry points reject inconsistent class dimensions', () => {
  const rows = [{ logits: [0, 1], label: 1 }, { logits: [0, 1, 2], label: 1 }];
  assert.throws(() => nll(rows, 1), /expected 2 logits/);
  assert.throws(() => fitTemperature(rows), /expected 2 logits/);
  assert.throws(() => toPredictions(rows), /expected 2 logits/);
});

test('temperature search rejects invalid bounds tolerance and iteration limits', () => {
  for (const options of [
    { lo: 0 }, { lo: -1 }, { lo: NaN }, { hi: Infinity }, { hi: NaN },
    { lo: 2, hi: 1 }, { lo: 1, hi: 1 }, { tolerance: 0 }, { tolerance: -1 },
    { tolerance: NaN }, { tolerance: Infinity }, { tolerance: 20 },
    { maxIterations: 0 }, { maxIterations: 1.5 }, { maxIterations: Infinity }, { maxIterations: 10_001 }
  ]) assert.throws(() => fitTemperature(sample, options), RangeError);
});

test('constant objectives and search termination have explicit statuses', () => {
  const constant = fitTemperature([{ logits: [4, 4, 4], label: 2 }]);
  assert.equal(constant.status, 'constant');
  assert.equal(constant.temperature, 1);
  assert.equal(constant.atBound, null);
  assert.equal(constant.improved, false);
  assert.equal(constant.iterations, 0);
  const capped = fitTemperature(sample, { maxIterations: 1 });
  assert.equal(capped.status, 'max-iterations');
  assert.equal(capped.iterations, 1);
  assert.equal(capped.atBound, null);
  const boundary = fitTemperature(sample);
  assert.equal(boundary.status, 'boundary');
  assert.equal(boundary.atBound, 'lo');
  const stalled = fitTemperature(sample, { tolerance: Number.MIN_VALUE, maxIterations: 1000 });
  assert.equal(stalled.status, 'stalled');
  assert.ok(stalled.iterations < 1000);
});

test('temperature search recovers an independent analytic optimum', () => {
  const rows = Array.from({ length: 400 }, (_, i) => ({ logits: [0, 2], label: i < 300 ? 1 : 0 }));
  const fitted = fitTemperature(rows, { tolerance: 1e-7 });
  assert.equal(fitted.status, 'converged');
  assert.ok(Math.abs(fitted.temperature - 2 / Math.log(3)) < 1e-6);
  assert.ok(Math.abs(fitted.nllAfter - (-0.75 * Math.log(0.75) - 0.25 * Math.log(0.25))) < 1e-12);
});
