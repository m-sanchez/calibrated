/** ECE resampling and simulation summaries do not establish calibration or correct estimator bias. */

import { calibrationError } from './metrics.ts';
import type { BinStrategy, Prediction } from './binning.ts';

// Duplicating the generator preserves the shared seeded stream without a runtime dependency.
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function quantile(sorted: Float64Array, q: number): number {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)));
  return sorted[i];
}

export interface EceInterval {
  ece: number;
  /** Percentile endpoints describe the bootstrap distribution without correcting ECE bias. */
  low: number;
  high: number;
  level: number;
  iterations: number;
  bins: number;
  strategy: BinStrategy;
  n: number;
}

export interface EceIntervalOptions {
  bins?: number;
  strategy?: BinStrategy;
  /** Defaults to 2000 draws; more draws reduce Monte Carlo variation, not sampling uncertainty. */
  iterations?: number;
  /** Central fraction of bootstrap draws to report, default 0.95. */
  level?: number;
  /** Fixed data, settings and seed reproduce the resampling summary. */
  seed?: number;
}

/** Resamples whole rows with replacement and reports central ECE percentiles without guaranteed population coverage. */
export function eceInterval(
  predictions: Prediction[],
  opts: EceIntervalOptions = {}
): EceInterval {
  const bins = opts.bins ?? 10;
  const strategy = opts.strategy ?? 'equal-width';
  const iterations = opts.iterations ?? 2000;
  const level = opts.level ?? 0.95;
  const seed = opts.seed ?? 42;
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new RangeError(`iterations must be a positive integer, got ${iterations}`);
  }
  if (!(level > 0 && level < 1)) {
    throw new RangeError(`level must be in (0, 1), got ${level}`);
  }
  const n = predictions.length;
  const ece = calibrationError(predictions, bins, strategy).ece;
  if (n === 0) {
    return { ece, low: NaN, high: NaN, level, iterations, bins, strategy, n };
  }

  const rand = lcg(seed);
  const draws = new Float64Array(iterations);
  const resample = new Array<Prediction>(n);
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < n; i++) resample[i] = predictions[Math.floor(rand() * n)];
    draws[it] = calibrationError(resample, bins, strategy).ece;
  }
  draws.sort();
  const loIdx = Math.floor(((1 - level) / 2) * iterations);
  const hiIdx = Math.ceil((1 - (1 - level) / 2) * iterations) - 1;
  return { ece, low: draws[loIdx], high: draws[hiIdx], level, iterations, bins, strategy, n };
}

export interface NullEce {
  /** Summaries of simulated ECE under independent Bernoulli(confidence) correctness. */
  median: number;
  mean: number;
  /** The simulated 95th percentile is a reference quantile, not a calibration certificate or deployment threshold. */
  p95: number;
  n: number;
  bins: number;
  strategy: BinStrategy;
  iterations: number;
}

export interface NullEceOptions {
  /** An array fixes confidences; a sample count redraws them uniformly on [0.5, 1] in every simulation. */
  confidences: number[] | number;
  bins?: number;
  strategy?: BinStrategy;
  /** Defaults to 2000 simulated datasets. */
  iterations?: number;
  seed?: number;
}

/** Simulates independent correctness under a calibration null; subtracting its summaries does not produce an unbiased ECE. */
export function nullEce(opts: NullEceOptions): NullEce {
  const bins = opts.bins ?? 10;
  const strategy = opts.strategy ?? 'equal-width';
  const iterations = opts.iterations ?? 2000;
  if (!Number.isInteger(iterations) || iterations < 1) {
    throw new RangeError(`iterations must be a positive integer, got ${iterations}`);
  }
  const rand = lcg(opts.seed ?? 42);
  if (typeof opts.confidences === 'number' && !(Number.isInteger(opts.confidences) && opts.confidences >= 0)) {
    throw new RangeError(`confidences must be an array or a non-negative integer n, got ${opts.confidences}`);
  }
  const fixed = typeof opts.confidences === 'number' ? null : opts.confidences;
  const n = fixed === null ? (opts.confidences as number) : fixed.length;
  if (n === 0) {
    return { median: NaN, mean: NaN, p95: NaN, n, bins, strategy, iterations };
  }

  const draws = new Float64Array(iterations);
  const sample = new Array<Prediction>(n);
  let total = 0;
  for (let it = 0; it < iterations; it++) {
    for (let i = 0; i < n; i++) {
      const confidence = fixed === null ? 0.5 + rand() * 0.5 : fixed[i];
      sample[i] = { confidence, correct: rand() < confidence };
    }
    const ece = calibrationError(sample, bins, strategy).ece;
    draws[it] = ece;
    total += ece;
  }
  draws.sort();
  return {
    median: quantile(draws, 0.5),
    mean: total / iterations,
    p95: quantile(draws, 0.95),
    n,
    bins,
    strategy,
    iterations
  };
}
