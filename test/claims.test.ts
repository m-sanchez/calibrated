import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { brier, calibrationError } from '../src/metrics.ts';
import type { Prediction } from '../src/binning.ts';

test('the package has no runtime dependencies', () => {
  const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
  for (const field of ['dependencies', 'peerDependencies', 'optionalDependencies']) {
    assert.deepEqual(pkg[field] ?? {}, {}, `${field} must stay empty`);
  }
});

test('the demo prints the numbers the README quotes', () => {
  const demo = fileURLToPath(new URL('../demo/demo.ts', import.meta.url));
  const out = execFileSync(process.execPath, [demo], { encoding: 'utf8' });
  assert.match(out, /calibration n=3000 \(seed 7\), test n=3000 \(seed 19\)/);
  assert.match(out, /before scaling\s+ECE 0\.279\s+Brier 0\.279\s+accuracy 72\.0%/);
  assert.match(out, /after \(T=4\.45\)\s+ECE 0\.010\s+Brier 0\.202\s+accuracy 72\.0%/);
  assert.match(out, /held-out NLL 2\.520 -> 0\.901; fit status: converged/);
});

test('ECE depends on the bin count; the raw Brier score does not', () => {
  const preds: Prediction[] = Array.from({ length: 500 }, (_, i) => ({
    confidence: 0.5 + (i % 50) / 100,
    correct: i % 3 !== 0
  }));
  assert.equal(brier(preds, 10).score, brier(preds, 30).score);
  assert.notEqual(calibrationError(preds, 10).ece, calibrationError(preds, 30).ece);
  // the decomposition does move with the bin count
  assert.notEqual(brier(preds, 10).reliability, brier(preds, 30).reliability);
});
