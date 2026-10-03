export interface LogitSample {
  logits: number[];
  label: number;
}

function validateTemperature(temperature: number): void {
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError(`temperature must be finite and > 0, got ${temperature}`);
  }
}

function validateLogits(logits: number[], label: string): void {
  if (!Array.isArray(logits) || logits.length === 0) {
    throw new RangeError(`${label} must be a non-empty array`);
  }
  for (let i = 0; i < logits.length; i++) {
    if (!Number.isFinite(logits[i])) {
      throw new RangeError(`${label} at index ${i} must be finite, got ${logits[i]}`);
    }
  }
}

function argmax(logits: number[]): number {
  let index = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[index]) index = i;
  return index;
}

function scaledGap(logit: number, maximum: number, temperature: number): number {
  const gap = logit - maximum;
  // Scaling first recovers representable differences when raw subtraction overflows.
  return Number.isFinite(gap) ? gap / temperature : logit / temperature - maximum / temperature;
}

function softmaxUnchecked(logits: number[], temperature: number): number[] {
  const maximum = logits[argmax(logits)];
  const out = new Array<number>(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const value = Math.exp(scaledGap(logits[i], maximum, temperature));
    out[i] = value;
    sum += value;
  }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

export function softmax(logits: number[], temperature = 1): number[] {
  validateTemperature(temperature);
  validateLogits(logits, 'logit');
  return softmaxUnchecked(logits, temperature);
}

function validate(samples: LogitSample[]): void {
  const classes = samples[0]?.logits?.length;
  for (let i = 0; i < samples.length; i++) {
    const sample = samples[i];
    validateLogits(sample.logits, `sample ${i}: logit`);
    if (sample.logits.length < 2) {
      throw new RangeError(`sample ${i}: need at least 2 logits, got ${sample.logits.length}`);
    }
    if (sample.logits.length !== classes) {
      throw new RangeError(`sample ${i}: expected ${classes} logits, got ${sample.logits.length}`);
    }
    if (!Number.isInteger(sample.label) || sample.label < 0 || sample.label >= sample.logits.length) {
      const hint = sample.label === sample.logits.length ? ' - check for 1-indexed labels' : '';
      throw new RangeError(`sample ${i}: label must be an integer class index in [0, ${sample.logits.length}), got ${sample.label}${hint}`);
    }
  }
}

export function nll(samples: LogitSample[], temperature: number): number {
  validateTemperature(temperature);
  validate(samples);
  return nllUnchecked(samples, temperature);
}

function nllUnchecked(samples: LogitSample[], temperature: number): number {
  if (samples.length === 0) return NaN;
  let mean = 0;
  let correction = 0;
  for (let i = 0; i < samples.length; i++) {
    const { logits, label } = samples[i];
    const maximumIndex = argmax(logits);
    const maximum = logits[maximumIndex];
    let tail = 0;
    for (let c = 0; c < logits.length; c++) {
      if (c !== maximumIndex) tail += Math.exp(scaledGap(logits[c], maximum, temperature));
    }
    const loss = Math.log1p(tail) - scaledGap(logits[label], maximum, temperature);
    if (!Number.isFinite(loss)) {
      throw new RangeError(`sample ${i}: NLL exceeds finite numeric range at temperature ${temperature}`);
    }
    const contribution = loss / samples.length - correction;
    const next = mean + contribution;
    correction = next - mean - contribution;
    mean = next;
  }
  if (!Number.isFinite(mean)) throw new RangeError('mean NLL exceeds finite numeric range');
  return mean;
}

export interface TemperatureFit {
  temperature: number;
  nllBefore: number;
  nllAfter: number;
  improved: boolean;
  atBound: 'lo' | 'hi' | null;
  status: 'converged' | 'boundary' | 'constant' | 'max-iterations' | 'stalled';
  iterations: number;
}

export function fitTemperature(
  samples: LogitSample[],
  opts: { lo?: number; hi?: number; tolerance?: number; maxIterations?: number } = {}
): TemperatureFit {
  const lo = opts.lo ?? 0.05;
  const hi = opts.hi ?? 20;
  const tolerance = opts.tolerance ?? 1e-4;
  const maxIterations = opts.maxIterations ?? 256;
  validateTemperature(lo);
  validateTemperature(hi);
  if (lo >= hi) throw new RangeError('temperature bounds must satisfy lo < hi');
  if (!Number.isFinite(tolerance) || tolerance <= 0 || tolerance >= hi - lo) {
    throw new RangeError('tolerance must be finite, > 0 and smaller than hi - lo');
  }
  if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 10_000) {
    throw new RangeError('maxIterations must be an integer in [1, 10000]');
  }
  validate(samples);
  if (samples.length === 0) throw new RangeError('cannot fit temperature without samples');
  const nllBefore = nllUnchecked(samples, 1);
  if (samples.every(({ logits }) => logits.every((value) => value === logits[0]))) {
    return {
      temperature: Math.min(hi, Math.max(lo, 1)),
      nllBefore,
      nllAfter: nllBefore,
      improved: false,
      atBound: null,
      status: 'constant',
      iterations: 0
    };
  }

  const phi = (Math.sqrt(5) - 1) / 2;
  let a = lo;
  let b = hi;
  let c = b - phi * (b - a);
  let d = a + phi * (b - a);
  let fc = nllUnchecked(samples, c);
  let fd = nllUnchecked(samples, d);
  let iterations = 0;
  let stalled = false;
  while (b - a > tolerance && iterations < maxIterations) {
    const previousA = a;
    const previousB = b;
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
    iterations++;
    if (a === previousA && b === previousB) {
      stalled = true;
      break;
    }
  }
  const temperature = a + (b - a) / 2;
  const nllAfter = nllUnchecked(samples, temperature);
  const converged = b - a <= tolerance;
  const atBound = converged
    ? temperature - lo <= tolerance ? 'lo' : hi - temperature <= tolerance ? 'hi' : null
    : null;
  const status = converged ? atBound === null ? 'converged' : 'boundary' : stalled ? 'stalled' : 'max-iterations';
  return { temperature, nllBefore, nllAfter, improved: nllAfter <= nllBefore + 1e-9, atBound, status, iterations };
}

export function toPredictions(samples: LogitSample[], temperature = 1) {
  validateTemperature(temperature);
  validate(samples);
  return samples.map(({ logits, label }) => {
    // Rounded probabilities can tie even when raw logits have a unique maximum.
    const predicted = argmax(logits);
    const probabilities = softmaxUnchecked(logits, temperature);
    return { confidence: probabilities[predicted], correct: predicted === label };
  });
}
