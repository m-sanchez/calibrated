/** The claims the README makes out loud, enforced. Anything quoted on the
 * front page is either measured here or is not said. */

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
  assert.match(out, /before scaling\s+ECE 0\.284\s+Brier 0\.284\s+accuracy 71\.6%/);
  assert.match(out, /after \(T=4\.45\)\s+ECE 0\.006\s+Brier 0\.204\s+accuracy 71\.6%/);
  const before = Number(/before scaling\s+ECE (\d+\.\d+)/.exec(out)![1]);
  const after = Number(/after \(T=4\.45\)\s+ECE (\d+\.\d+)/.exec(out)![1]);
  assert.ok(after * 40 <= before, `README claims a 40x fall: ${before} -> ${after}`);
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
