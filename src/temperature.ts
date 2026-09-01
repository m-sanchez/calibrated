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

/** Numerically stable softmax of logits divided by temperature. The maximum
 * is taken in a loop rather than with `Math.max(...logits)`: the spread form
 * puts every logit on the call stack and throws above roughly 100k of them,
 * and a modern LLM vocabulary is 128k-256k classes. */
export function softmax(logits: number[], temperature = 1): number[] {
  if (!(temperature > 0)) throw new RangeError(`temperature must be > 0, got ${temperature}`);
  const k = logits.length;
  let max = -Infinity;
  for (let i = 0; i < k; i++) {
    const z = logits[i];
    if (!Number.isFinite(z)) throw new RangeError(`logit at index ${i} must be finite, got ${z}`);
    const scaled = z / temperature;
    if (scaled > max) max = scaled;
  }
  const out = new Array<number>(k);
  let sum = 0;
  for (let i = 0; i < k; i++) {
    const e = Math.exp(logits[i] / temperature - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < k; i++) out[i] /= sum;
  return out;
}

/** Reject the input that silently produces a wrong answer: a label outside
 * the class range makes `softmax(...)[label]` undefined, so the NLL is NaN,
 * so every comparison in the search is false and the search walks to its
 * upper bracket and reports that as a fit. The commonest cause by far is a
 * 1-indexed label column. */
function validate(samples: LogitSample[]): void {
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i];
    if (s.logits.length < 2) {
      throw new RangeError(`sample ${i}: need at least 2 logits (one per class), got ${s.logits.length}`);
    }
    for (let c = 0; c < s.logits.length; c++) {
      if (!Number.isFinite(s.logits[c])) {
        throw new RangeError(`sample ${i}: logit ${c} must be finite, got ${s.logits[c]}`);
      }
    }
    if (!Number.isInteger(s.label) || s.label < 0 || s.label >= s.logits.length) {
      const hint =
        s.label === s.logits.length ? ' - a label equal to the class count means 1-indexed labels' : '';
      throw new RangeError(
        `sample ${i}: label must be an integer class index in [0, ${s.logits.length}), got ${s.label}${hint}`
      );
    }
  }
}

/** Mean negative log-likelihood of the labels under temperature T. */
export function nll(samples: LogitSample[], temperature: number): number {
  validate(samples);
  return nllUnchecked(samples, temperature);
}

/** The search calls this tens of times; the samples are validated once, at
 * the public entry point, rather than on every evaluation. */
function nllUnchecked(samples: LogitSample[], temperature: number): number {
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
  /** `'lo'` or `'hi'` when the search converged onto its own bracket edge
   * instead of an interior minimum. The returned temperature is then a
   * boundary, not a fit: the real optimum lies outside [lo, hi] and the
   * bracket needs widening. `null` when the minimum is interior. */
  atBound: 'lo' | 'hi' | null;
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
  validate(samples);
  const nllBefore = nllUnchecked(samples, 1);
  if (samples.length === 0) {
    return { temperature: 1, nllBefore, nllAfter: nllBefore, improved: false, atBound: null };
  }

  const phi = (Math.sqrt(5) - 1) / 2; // 0.618...
  let a = lo;
  let b = hi;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  let fc = nllUnchecked(samples, c);
  let fd = nllUnchecked(samples, d);
  while (b - a > tol) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - phi * (b - a);
      fc = nllUnchecked(samples, c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + phi * (b - a);
      fd = nllUnchecked(samples, d);
    }
  }
  const temperature = (a + b) / 2;
  const nllAfter = nllUnchecked(samples, temperature);
  const edge = Math.max(tol, 1e-9);
  const atBound = temperature <= lo + edge ? 'lo' : temperature >= hi - edge ? 'hi' : null;
  return { temperature, nllBefore, nllAfter, improved: nllAfter <= nllBefore + 1e-9, atBound };
}

/** Turn a fitted temperature into calibrated (confidence, correct) pairs,
 * ready for `calibrationError` / `brier`: the recalibrated confidence is
 * the top softmax probability, and correctness is argmax == label. */
export function toPredictions(samples: LogitSample[], temperature = 1) {
  validate(samples);
  return samples.map((s) => {
    const probs = softmax(s.logits, temperature);
    let argmax = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[argmax]) argmax = i;
    return { confidence: probs[argmax], correct: argmax === s.label };
  });
}
