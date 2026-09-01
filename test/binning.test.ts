import { test } from 'node:test';
import assert from 'node:assert/strict';
import { bin } from '../src/binning.ts';
import { calibrationError } from '../src/metrics.ts';
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

function shuffle(items: Prediction[], rand: () => number): Prediction[] {
  const out = [...items];
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/** The realistic tie-heavy case: a top-heavy classifier whose confidences
 * are rounded to 2dp, so only 11 distinct values exist across 2000 rows. */
function topHeavy(seed: number): Prediction[] {
  const rand = seededUnit(seed);
  return Array.from({ length: 2000 }, () => {
    const confidence = Math.round((0.9 + rand() * 0.1) * 100) / 100;
    return { confidence, correct: rand() < confidence - 0.08 };
  });
}

test('equal-mass never splits a run of tied confidences', () => {
  // 1000 predictions all at confidence 0.9, 60% correct. There is exactly
  // one distinct confidence, so there is exactly one bin and the gap is 0.30.
  const preds: Prediction[] = Array.from({ length: 1000 }, (_, i) => ({
    confidence: 0.9,
    correct: i < 600
  }));
  assert.equal(bin(preds, 15, 'equal-mass').length, 1, 'one distinct confidence is one bin');
  const { ece, mce } = calibrationError(preds, 15, 'equal-mass');
  assert.ok(Math.abs(ece - 0.3) < 1e-12, `ece should be exactly 0.30, got ${ece}`);
  assert.ok(Math.abs(mce - 0.3) < 1e-12, `mce should be exactly 0.30, got ${mce}`);
});

test('equal-mass calibration error is a function of the data, not of row order', () => {
  const base = topHeavy(11);
  const reference = calibrationError(base, 15, 'equal-mass');
  for (let k = 0; k < 50; k++) {
    const permuted = calibrationError(shuffle(base, seededUnit(100 + k)), 15, 'equal-mass');
    assert.ok(
      Math.abs(permuted.ece - reference.ece) < 1e-12,
      `shuffle ${k}: ece ${permuted.ece} vs ${reference.ece}`
    );
    assert.ok(Math.abs(permuted.mce - reference.mce) < 1e-12, `shuffle ${k}: mce`);
    assert.equal(permuted.effectiveBins, reference.effectiveBins, `shuffle ${k}: bin count`);
  }
});

test('a confidence value never lands in two different equal-mass bins', () => {
  const bins = bin(topHeavy(11), 15, 'equal-mass');
  for (let i = 1; i < bins.length; i++) {
    assert.ok(
      bins[i].range[0] > bins[i - 1].range[1],
      `bin ${i} starts at ${bins[i].range[0]} but bin ${i - 1} already ran to ${bins[i - 1].range[1]}`
    );
  }
});

test('equal-mass keeps every bin on comparable footing when confidence is top-heavy', () => {
  // 95% of the mass above 0.9, continuous: the case the README recommends
  // equal-mass for. Equal-width puts almost everything in one bin, so nine
  // of its ten estimates rest on a handful of points each.
  const rand = seededUnit(4);
  const preds: Prediction[] = Array.from({ length: 2000 }, () => {
    const confidence = rand() < 0.95 ? 0.9 + rand() * 0.1 : rand() * 0.9;
    return { confidence, correct: rand() < confidence - 0.05 };
  });
  const mass = bin(preds, 10, 'equal-mass');
  assert.equal(mass.length, 10);
  for (const b of mass) assert.equal(b.count, 200, 'every equal-mass bin holds the same count');
  const width = bin(preds, 10, 'equal-width');
  const biggest = Math.max(...width.map((b) => b.count));
  assert.ok(biggest > 0.9 * preds.length, `equal-width piles ${biggest}/2000 into one bin`);
});
