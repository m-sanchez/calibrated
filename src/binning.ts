/** Confidence binning for calibration. A model is calibrated when, among
 * the predictions it makes at confidence p, a fraction p are correct. To
 * measure that we group predictions into confidence bins and compare each
 * bin's mean confidence to its observed accuracy.
 *
 * Two strategies, because equal-width bins mislead when confidence piles
 * up at the top (the usual case for a modern classifier): equal-width
 * cuts [0,1] into M fixed slices; equal-mass (adaptive) cuts so each bin
 * holds the same number of predictions, which keeps every bin's accuracy
 * estimate on comparable footing. */

export interface Prediction {
  /** the model's confidence in its prediction, in [0, 1] */
  confidence: number;
  /** whether that prediction was correct */
  correct: boolean;
}

export interface Bin {
  /** [lo, hi) confidence range this bin covers (hi inclusive on the last) */
  range: [number, number];
  count: number;
  /** mean confidence of the predictions that fell here */
  confidence: number;
  /** fraction of them that were correct */
  accuracy: number;
  /** |confidence - accuracy|: the calibration gap for this bin */
  gap: number;
}

export type BinStrategy = 'equal-width' | 'equal-mass';

function assertProbability(value: number, label: string): void {
  if (!Number.isFinite(value) || value < 0 || value > 1) {
    throw new RangeError(`${label} must be a probability in [0, 1], got ${value}`);
  }
}

function summarize(range: [number, number], group: Prediction[]): Bin {
  const count = group.length;
  if (count === 0) return { range, count: 0, confidence: 0, accuracy: 0, gap: 0 };
  const confidence = group.reduce((s, p) => s + p.confidence, 0) / count;
  const accuracy = group.filter((p) => p.correct).length / count;
  return { range, count, confidence, accuracy, gap: Math.abs(confidence - accuracy) };
}

/** Partition predictions into calibration bins. Empty bins are kept (with
 * count 0) for equal-width so a reliability diagram shows the gaps; for
 * equal-mass every returned bin is non-empty by construction. */
export function bin(
  predictions: Prediction[],
  bins = 10,
  strategy: BinStrategy = 'equal-width'
): Bin[] {
  if (!Number.isInteger(bins) || bins < 1) {
    throw new RangeError(`bins must be a positive integer, got ${bins}`);
  }
  for (const p of predictions) assertProbability(p.confidence, 'confidence');

  if (strategy === 'equal-width') {
    const edges = Array.from({ length: bins + 1 }, (_, i) => i / bins);
    return edges.slice(0, -1).map((lo, i) => {
      const hi = edges[i + 1];
      const isLast = i === bins - 1;
      const group = predictions.filter(
        (p) => p.confidence >= lo && (isLast ? p.confidence <= hi : p.confidence < hi)
      );
      return summarize([lo, hi], group);
    });
  }

  // equal-mass: sort by confidence and cut into near-equal groups
  const sorted = [...predictions].sort((a, b) => a.confidence - b.confidence);
  const n = sorted.length;
  if (n === 0) return [];
  const perBin = Math.ceil(n / bins);
  const out: Bin[] = [];
  for (let start = 0; start < n; start += perBin) {
    const group = sorted.slice(start, start + perBin);
    const lo = group[0].confidence;
    const hi = group[group.length - 1].confidence;
    out.push(summarize([lo, hi], group));
  }
  return out;
}
