import { test } from 'node:test';
import assert from 'node:assert/strict';
import { eceInterval, nullEce } from '../src/interval.ts';
import { calibrationError } from '../src/metrics.ts';
import type { Prediction } from '../src/binning.ts';

/** A band of `count` predictions all at `confidence`, of which exactly
 * `accuracy * count` are correct: a set with a known calibration gap. */
function band(confidence: number, accuracy: number, count: number): Prediction[] {
  const correct = Math.round(accuracy * count);
  return Array.from({ length: count }, (_, i) => ({ confidence, correct: i < correct }));
}

test('the bootstrap interval brackets the point estimate and covers the true gap', () => {
  const preds = band(0.9, 0.6, 1000); // true gap 0.30
  const r = eceInterval(preds, { bins: 15 });
  assert.ok(Math.abs(r.ece - 0.3) < 1e-12, `point estimate ${r.ece}`);
  assert.ok(r.low < r.ece && r.ece < r.high, `[${r.low}, ${r.high}] around ${r.ece}`);
  assert.ok(r.low <= 0.3 && 0.3 <= r.high, `[${r.low}, ${r.high}] must cover 0.30`);
  assert.equal(r.level, 0.95);
  assert.equal(r.iterations, 2000);
});

test('the interval reproduces exactly under a fixed seed', () => {
  const preds = band(0.9, 0.6, 400);
  const a = eceInterval(preds, { bins: 10, seed: 7 });
  const b = eceInterval(preds, { bins: 10, seed: 7 });
  assert.deepEqual(a, b);
  const other = eceInterval(preds, { bins: 10, seed: 8 });
  assert.notEqual(a.low, other.low, 'a different seed draws a different resample');
});

test('a higher confidence level gives a wider interval', () => {
  const preds = band(0.9, 0.6, 400);
  const narrow = eceInterval(preds, { bins: 10, level: 0.9 });
  const wide = eceInterval(preds, { bins: 10, level: 0.99 });
  assert.ok(wide.high - wide.low > narrow.high - narrow.low, `${wide.high - wide.low} vs ${narrow.high - narrow.low}`);
});

test('an interval over no data is no evidence, not a confident zero', () => {
  const r = eceInterval([], { bins: 10 });
  assert.ok(Number.isNaN(r.ece) && Number.isNaN(r.low) && Number.isNaN(r.high));
});

test('the null ECE floor falls with sample size and rises with bin count', () => {
  const opts = { strategy: 'equal-width' as const, iterations: 300, seed: 42 };
  const small = nullEce({ confidences: 100, bins: 15, ...opts });
  const large = nullEce({ confidences: 1000, bins: 15, ...opts });
  assert.ok(large.median < small.median, `n=1000 floor ${large.median} vs n=100 ${small.median}`);
  const few = nullEce({ confidences: 400, bins: 10, ...opts });
  const many = nullEce({ confidences: 400, bins: 30, ...opts });
  assert.ok(many.median > few.median, `30 bins ${many.median} vs 10 bins ${few.median}`);
  // the floor is not negligible: a perfectly calibrated 100-item eval set
  // still reports an ECE that would fail a naive 0.1 bar much of the time
  assert.ok(small.p95 > 0.1, `p95 at n=100 is ${small.p95}`);
});

test('the null floor is computed against the confidences you actually have', () => {
  // a set whose confidences are all 0.9 has no binning noise from the
  // confidence side at all: the floor is pure Bernoulli noise on accuracy
  const flat = nullEce({ confidences: new Array(1000).fill(0.9), bins: 15, iterations: 300, seed: 42 });
  assert.ok(flat.median < 0.03, `flat-confidence floor ${flat.median}`);
  assert.equal(flat.n, 1000);
  assert.equal(flat.iterations, 300);
  const again = nullEce({ confidences: new Array(1000).fill(0.9), bins: 15, iterations: 300, seed: 42 });
  assert.deepEqual(flat, again, 'the floor reproduces under a fixed seed');
});

test('the published noise-floor table is what nullEce reports', () => {
  // Every number in the README's noise-floor table, pinned. `npm run floor`
  // prints this table; if the simulation changes, the README is wrong and
  // this test says so.
  const table: Record<number, Record<number, [string, string]>> = {
    100: { 10: ['0.0678', '0.1155'], 15: ['0.0874', '0.1325'], 30: ['0.1184', '0.1611'] },
    200: { 10: ['0.0479', '0.0812'], 15: ['0.0620', '0.0928'], 30: ['0.0846', '0.1138'] },
    400: { 10: ['0.0345', '0.0572'], 15: ['0.0436', '0.0655'], 30: ['0.0605', '0.0818'] },
    1000: { 10: ['0.0217', '0.0361'], 15: ['0.0277', '0.0415'], 30: ['0.0379', '0.0522'] }
  };
  for (const [n, row] of Object.entries(table)) {
    for (const [bins, [median, p95]] of Object.entries(row)) {
      const r = nullEce({ confidences: Number(n), bins: Number(bins) });
      assert.equal(r.median.toFixed(4), median, `n=${n}, ${bins} bins: median`);
      assert.equal(r.p95.toFixed(4), p95, `n=${n}, ${bins} bins: p95`);
    }
  }
});

test('a 0.1 ship bar rejects a perfectly calibrated 100-item eval set a third of the time', () => {
  // The README quotes these rejection rates. Same LCG as src/interval.ts,
  // so the simulation is the one a reader can reproduce.
  const lcg = (seed: number) => {
    let s = seed >>> 0;
    return () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  };
  const rejectionRate = (n: number, bins: number) => {
    const rand = lcg(42);
    const iterations = 2000;
    let over = 0;
    for (let it = 0; it < iterations; it++) {
      const sample: Prediction[] = Array.from({ length: n }, () => {
        const confidence = 0.5 + rand() * 0.5;
        return { confidence, correct: rand() < confidence };
      });
      if (calibrationError(sample, bins, 'equal-width').ece > 0.1) over++;
    }
    return over / iterations;
  };
  assert.equal(rejectionRate(100, 15).toFixed(2), '0.31');
  assert.equal(rejectionRate(200, 15).toFixed(2), '0.03');
  assert.equal(rejectionRate(400, 15), 0);
});
