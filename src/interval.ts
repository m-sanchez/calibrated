/** Uncertainty for ECE.
 *
 * ECE is a point estimate over binned counts, and it is positively biased:
 * a perfectly calibrated model still reports a non-zero ECE, because each
 * bin's accuracy is measured on a finite sample and the absolute value
 * turns that sampling noise into error. The bias grows with the bin count
 * and shrinks with the sample size. So a bare "ECE 0.08" cannot be read as
 * miscalibration, and a ship gate of the form `ece <= 0.1` on a few hundred
 * examples can fail a model that is perfectly honest.
 *
 * Two functions, answering the two questions that turns a reading into a
 * decision. `eceInterval` resamples your predictions to say how much of the
 * reading is sampling noise. `nullEce` simulates a perfectly calibrated
 * model over the same confidences and binning to say what ECE looks like
 * when there is nothing wrong at all - the floor to compare against. */

import { calibrationError } from './metrics.ts';
import type { BinStrategy, Prediction } from './binning.ts';

/** The same linear congruential generator as @m-sanchez/ab-significance's
 * paired bootstrap, duplicated here on purpose: this package ships zero
 * runtime dependencies, and intervals reported across the family should be
 * drawn from the same stream so they are directly comparable. */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

/** The order statistic at `q` of an ascending array. */
function quantile(sorted: Float64Array, q: number): number {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * sorted.length)));
  return sorted[i];
}

export interface EceInterval {
  /** the point estimate on the data as it stands */
  ece: number;
  /** the `level` percentile interval of the bootstrap distribution */
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
  /** bootstrap resamples; more is tighter but slower. Default 2000. */
  iterations?: number;
  /** interval coverage, default 0.95 */
  level?: number;
  /** seed for the resampling, so a reported interval reproduces exactly */
  seed?: number;
}

/** A seeded non-parametric bootstrap interval for ECE: resample the
 * predictions with replacement `iterations` times, recompute ECE on each
 * resample, and report the middle `level` of that distribution.
 *
 * The interval covers sampling noise, not the bias: ECE is biased upward,
 * so the interval sits around the biased point estimate rather than around
 * the true calibration gap. Compare its low end with `nullEce` before
 * calling a reading real. */
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
  /** the ECE a perfectly calibrated model of this size and binning reports */
  median: number;
  mean: number;
  /** an observed ECE below this is not distinguishable from binning noise */
  p95: number;
  n: number;
  bins: number;
  strategy: BinStrategy;
  iterations: number;
}

export interface NullEceOptions {
  /** the confidences you actually have - pass `predictions.map(p =>
   * p.confidence)` - or just a sample size, in which case confidences are
   * drawn uniform on [0.5, 1] as a rough stand-in for a top-heavy set. Your
   * own confidences give the floor that applies to your reading. */
  confidences: number[] | number;
  bins?: number;
  strategy?: BinStrategy;
  /** simulated datasets; default 2000 */
  iterations?: number;
  seed?: number;
}

/** The noise floor: the ECE distribution of a model that is perfectly
 * calibrated at these confidences, under this binning. Each iteration draws
 * `correct ~ Bernoulli(confidence)` - honest by construction - and measures
 * the ECE anyway. Whatever comes back is what the metric reports when there
 * is nothing to report. Subtract it, or set your bar above its p95. */
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
  // given actual confidences the floor is conditional on them; given only a
  // size, the confidence spread is redrawn each iteration too, so the answer
  // is not hostage to one lucky draw
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
