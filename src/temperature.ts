/** Temperature scaling: the single-parameter recalibration of Guo et al.
 * (2017), "On Calibration of Modern Neural Networks".
 *
 * A modern classifier is usually overconfident. Temperature scaling fixes
 * that without touching the model's decisions: divide every logit by one
 * learned scalar T before the softmax. T > 1 softens the distribution
 * (less confident), T < 1 sharpens it; the argmax never moves, so accuracy
 * is unchanged and only the confidences are recalibrated. T is fit by
 * minimising negative log-likelihood on a held-out set - here by a
 * dependency-free golden-section search, since it is a smooth 1-D problem. */

export interface LogitSample {
  /** raw pre-softmax scores, one per class */
  logits: number[];
  /** index of the true class */
  label: number;
}

/** Numerically stable softmax of logits divided by temperature. */
export function softmax(logits: number[], temperature = 1): number[] {
  if (!(temperature > 0)) throw new RangeError(`temperature must be > 0, got ${temperature}`);
  const scaled = logits.map((z) => z / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

/** Mean negative log-likelihood of the labels under temperature T. */
export function nll(samples: LogitSample[], temperature: number): number {
  if (samples.length === 0) return 0;
  let total = 0;
  for (const s of samples) {
    const p = softmax(s.logits, temperature)[s.label];
    total += -Math.log(Math.max(p, 1e-12));
  }
  return total / samples.length;
}

export interface TemperatureFit {
  temperature: number;
  nllBefore: number;
  nllAfter: number;
  /** true when scaling actually helped (nll did not increase) */
  improved: boolean;
}

/** Fit the calibrating temperature by golden-section search over
 * [lo, hi]. NLL as a function of T is smooth and unimodal for a fixed
 * validation set, so a bracketing search converges without gradients. */
export function fitTemperature(
  samples: LogitSample[],
  opts: { lo?: number; hi?: number; tolerance?: number } = {}
): TemperatureFit {
  const lo = opts.lo ?? 0.05;
  const hi = opts.hi ?? 20;
  const tol = opts.tolerance ?? 1e-4;
  const nllBefore = nll(samples, 1);
  if (samples.length === 0) return { temperature: 1, nllBefore, nllAfter: nllBefore, improved: false };

  const phi = (Math.sqrt(5) - 1) / 2; // 0.618...
  let a = lo;
  let b = hi;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  let fc = nll(samples, c);
  let fd = nll(samples, d);
  while (b - a > tol) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - phi * (b - a);
      fc = nll(samples, c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + phi * (b - a);
      fd = nll(samples, d);
    }
  }
  const temperature = (a + b) / 2;
  const nllAfter = nll(samples, temperature);
  return { temperature, nllBefore, nllAfter, improved: nllAfter <= nllBefore + 1e-9 };
}

/** Turn a fitted temperature into calibrated (confidence, correct) pairs,
 * ready for `calibrationError` / `brier`: the recalibrated confidence is
 * the top softmax probability, and correctness is argmax == label. */
export function toPredictions(samples: LogitSample[], temperature = 1) {
  return samples.map((s) => {
    const probs = softmax(s.logits, temperature);
    let argmax = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[argmax]) argmax = i;
    return { confidence: probs[argmax], correct: argmax === s.label };
  });
}
