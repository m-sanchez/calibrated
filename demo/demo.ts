import { calibrationError, brier } from '../src/metrics.ts';
import { fitTemperature, nll, toPredictions } from '../src/temperature.ts';
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

function generate(seed: number, count: number): LogitSample[] {
  const rand = seeded(seed);
  return Array.from({ length: count }, () => {
    const label = Math.floor(rand() * 4);
    const predicted = rand() < 0.72 ? label : (label + 1 + Math.floor(rand() * 3)) % 4;
    const logits = [rand(), rand(), rand(), rand()];
    logits[predicted] += 9;
    return { logits, label };
  });
}

const calibration = generate(7, 3000);
const test = generate(19, 3000);
const fit = fitTemperature(calibration);
const before = toPredictions(test, 1);
const after = toPredictions(test, fit.temperature);

const row = (label: string, preds: ReturnType<typeof toPredictions>) => {
  const ce = calibrationError(preds, 15);
  const b = brier(preds, 15);
  const acc = preds.filter((p) => p.correct).length / preds.length;
  return `${label.padEnd(22)} ECE ${ce.ece.toFixed(3)}   Brier ${b.score.toFixed(3)}   accuracy ${(acc * 100).toFixed(1)}%`;
};

console.log('synthetic 4-class classifier: calibration n=3000 (seed 7), test n=3000 (seed 19)\n');
console.log(row('before scaling', before));
console.log(row(`after (T=${fit.temperature.toFixed(2)})`, after));
console.log(
  `\nheld-out NLL ${nll(test, 1).toFixed(3)} -> ${nll(test, fit.temperature).toFixed(3)}; fit status: ${fit.status}.`
);
