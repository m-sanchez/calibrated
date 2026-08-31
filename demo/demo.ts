/** npm run demo: an overconfident classifier, before and after temperature
 * scaling. Synthetic, seeded, reproducible. */

import { calibrationError, brier } from '../src/metrics.ts';
import { fitTemperature, toPredictions } from '../src/temperature.ts';
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

const rand = seeded(7);
const samples: LogitSample[] = [];
for (let i = 0; i < 3000; i++) {
  const label = Math.floor(rand() * 4);
  const predicted = rand() < 0.72 ? label : (label + 1 + Math.floor(rand() * 3)) % 4;
  const logits = [rand(), rand(), rand(), rand()];
  logits[predicted] += 9; // a big, overconfident margin
  samples.push({ logits, label });
}

const fit = fitTemperature(samples);
const before = toPredictions(samples, 1);
const after = toPredictions(samples, fit.temperature);

const row = (label: string, preds: ReturnType<typeof toPredictions>) => {
  const ce = calibrationError(preds, 15);
  const b = brier(preds, 15);
  const acc = preds.filter((p) => p.correct).length / preds.length;
  return `${label.padEnd(22)} ECE ${ce.ece.toFixed(3)}   Brier ${b.score.toFixed(3)}   accuracy ${(acc * 100).toFixed(1)}%`;
};

console.log('a 4-class classifier, 72% accurate, wildly overconfident\n');
console.log(row('before scaling', before));
console.log(row(`after (T=${fit.temperature.toFixed(2)})`, after));
console.log(
  `\ntemperature scaling cut ECE without moving a single prediction; ` +
    `NLL ${fit.nllBefore.toFixed(3)} -> ${fit.nllAfter.toFixed(3)}.`
);
