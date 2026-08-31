/** Calibration metrics over binned predictions.
 *
 * ECE (expected calibration error) is the headline number: the average
 * gap between confidence and accuracy, weighted by how many predictions
 * sit in each bin. MCE is the worst single bin. Brier score measures the
 * raw squared error of the probabilities, and its decomposition splits
 * that into reliability (calibration), resolution (how much the
 * predictions vary with outcome), and the irreducible uncertainty of the
 * task - so a bad Brier score can be read as "miscalibrated" versus
 * "uninformative". These are standard, published measures; the point of
 * this package is a careful, dependency-free, well-tested implementation. */

import { bin } from './binning.ts';
import type { BinStrategy, Prediction } from './binning.ts';

export interface CalibrationError {
  /** expected calibration error: sum_b (n_b / N) * |acc_b - conf_b| */
  ece: number;
  /** maximum calibration error: max_b |acc_b - conf_b| over non-empty bins */
  mce: number;
  bins: number;
  strategy: BinStrategy;
  n: number;
}

/** The reliability diagram: per-bin mean confidence vs observed accuracy,
 * the data you plot to see calibration at a glance. A perfectly calibrated
 * model sits on the diagonal (confidence == accuracy in every bin); bars
 * below it are overconfidence, above it underconfidence. This returns the
 * numbers; render them with whatever you like. */
export function reliabilityDiagram(
  predictions: Prediction[],
  bins = 10,
  strategy: BinStrategy = 'equal-width'
) {
  return bin(predictions, bins, strategy);
}

export function calibrationError(
  predictions: Prediction[],
  bins = 10,
  strategy: BinStrategy = 'equal-width'
): CalibrationError {
  const n = predictions.length;
  const groups = bin(predictions, bins, strategy).filter((b) => b.count > 0);
  const ece = groups.reduce((s, b) => s + (b.count / n) * b.gap, 0);
  const mce = groups.reduce((m, b) => Math.max(m, b.gap), 0);
  return { ece: n === 0 ? 0 : ece, mce, bins, strategy, n };
}

export interface BrierDecomposition {
  /** mean((confidence - outcome)^2); lower is better, in [0, 1] */
  score: number;
  /** reliability (calibration): 0 is perfectly calibrated. Lower is better */
  reliability: number;
  /** resolution: how much predictions separate outcomes. Higher is better */
  resolution: number;
  /** uncertainty: base-rate variance of the task; the floor Brier can reach */
  uncertainty: number;
}

/** Brier score and Murphy's three-term decomposition. The decomposition
 * needs bins (it is defined over grouped forecasts); the raw score does
 * not, so the raw score is exact while the decomposition depends on the
 * bin count, and `score ≈ reliability - resolution + uncertainty` up to
 * the binning approximation. */
export function brier(
  predictions: Prediction[],
  bins = 10,
  strategy: BinStrategy = 'equal-width'
): BrierDecomposition {
  const n = predictions.length;
  if (n === 0) return { score: 0, reliability: 0, resolution: 0, uncertainty: 0 };
  const outcome = (p: Prediction) => (p.correct ? 1 : 0);
  const score = predictions.reduce((s, p) => s + (p.confidence - outcome(p)) ** 2, 0) / n;

  const base = predictions.filter((p) => p.correct).length / n; // overall accuracy
  const uncertainty = base * (1 - base);
  const groups = bin(predictions, bins, strategy).filter((b) => b.count > 0);
  let reliability = 0;
  let resolution = 0;
  for (const b of groups) {
    reliability += (b.count / n) * (b.confidence - b.accuracy) ** 2;
    resolution += (b.count / n) * (b.accuracy - base) ** 2;
  }
  return { score, reliability, resolution, uncertainty };
}
