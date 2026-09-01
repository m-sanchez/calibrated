# calibrated

![TypeScript](https://img.shields.io/badge/TypeScript-erasable_syntax-3178C6?logo=typescript&logoColor=white)
![Node](https://img.shields.io/badge/node-%3E%3D22.18-5FA04E?logo=nodedotjs&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-B45309)
[![CI](https://github.com/m-sanchez/calibrated/actions/workflows/test.yml/badge.svg)](https://github.com/m-sanchez/calibrated/actions/workflows/test.yml)
![License](https://img.shields.io/badge/license-MIT-6E6E6E)
[![npm](https://img.shields.io/npm/v/@m-sanchez/calibrated?color=CB3837&logo=npm&logoColor=white)](https://www.npmjs.com/package/@m-sanchez/calibrated)

> **In plain English:** when a model says it is "90% sure", is it actually right 90% of the time? `calibrated` measures whether those confidence numbers can be trusted, and corrects them when they cannot.

Is your model's confidence honest? ECE, Brier decomposition, reliability
diagrams, and temperature scaling - zero dependencies.

[More tools](https://github.com/m-sanchez) · [Working rules](https://miguelsanchez.co.uk/ethics) ·
[Worked example: routing-study](https://github.com/m-sanchez/routing-study)

*Provenance: a fresh, dependency-free implementation of standard methods,
written to test the systems the other tools came from. First published
2026-08-31.*

A model is *calibrated* when the confidence it reports matches how often it
is right: among the predictions it makes at 90% confidence, 90% should be
correct. Modern classifiers usually are not - they are overconfident - and
an overconfident model that also refuses or escalates on low confidence is
making those decisions on numbers that do not mean what they say. This
measures the gap and closes it, from labelled `(confidence, correct)`
pairs, with a careful dependency-free implementation of standard measures.

```ts
import { calibrationError, brier, reliabilityDiagram } from '@m-sanchez/calibrated';

const { ece, mce } = calibrationError(predictions, 15);   // expected & worst-bin gap
const { score, reliability, resolution, uncertainty } = brier(predictions);
const bins = reliabilityDiagram(predictions, 15);         // confidence vs accuracy, to plot
```

- **ECE** (expected calibration error): the average confidence-vs-accuracy
  gap, weighted by bin population. The headline number.
- **MCE**: the single worst bin - the confidence level you can trust least.
- **Brier score** with Murphy's decomposition into **reliability**
  (calibration), **resolution** (how much predictions separate outcomes),
  and **uncertainty** (the task's irreducible floor), so a bad score reads
  as *miscalibrated* versus *uninformative*.
- **Reliability diagram** data: per-bin mean confidence vs observed
  accuracy, with equal-width or equal-mass (adaptive) binning - the latter
  keeps every bin's estimate on comparable footing when confidence piles
  up near 1.0, which it always does. Equal-mass cuts on values rather than
  positions, so a repeated confidence is never split across a boundary and
  the answer does not depend on the order the rows arrived in; when the
  data has fewer distinct confidences than bins you get fewer bins, and
  `effectiveBins` says how many.

## Is the number real?

ECE is a point estimate over binned counts, and it is biased upward: a
perfectly calibrated model still reports a non-zero ECE, because every
bin's accuracy is measured on a finite sample and the absolute value
turns that noise into error. On a few hundred examples, "ECE 0.08" is not
evidence of anything by itself.

```ts
import { eceInterval, nullEce } from '@m-sanchez/calibrated';

const { ece, low, high } = eceInterval(predictions, { bins: 15 });
const floor = nullEce({ confidences: predictions.map((p) => p.confidence), bins: 15 });
```

`eceInterval` resamples your predictions with replacement and reports the
middle 95% of the ECEs that come back - how much of the reading is
sampling noise. `nullEce` goes the other way: it simulates a model that is
calibrated by construction, over your confidences and your binning, and
reports the ECE the metric produces anyway. Both are seeded, so a reported
interval reproduces exactly.

The floor is not small. `npm run floor`, on confidences uniform on
[0.5, 1], 2000 draws, median / p95:

| n | 10 bins | 15 bins | 30 bins |
| --: | :-- | :-- | :-- |
| 100 | 0.0678 / 0.1155 | 0.0874 / 0.1325 | 0.1184 / 0.1611 |
| 200 | 0.0479 / 0.0812 | 0.0620 / 0.0928 | 0.0846 / 0.1138 |
| 400 | 0.0345 / 0.0572 | 0.0436 / 0.0655 | 0.0605 / 0.0818 |
| 1000 | 0.0217 / 0.0361 | 0.0277 / 0.0415 | 0.0379 / 0.0522 |

A perfectly calibrated model on a 100-item eval set with 15 bins reports a
median ECE of 0.087, and fails a bare `ece <= 0.1` ship bar in 31% of
draws - with nothing wrong with it. At n=200 that falls to 3%, and at
n=400 it did not happen once in 2000 draws. Compare a reading against the
floor before calling it miscalibration, and prefer more data to more bins.

## Fix it: temperature scaling

Temperature scaling (Guo et al., 2017) recalibrates without touching a
single decision: divide every logit by one learned scalar `T` before the
softmax. `T > 1` softens overconfidence; the argmax never moves, so
accuracy is unchanged. `T` is fit by minimising NLL on a held-out set,
here by a dependency-free golden-section search.

```ts
import { fitTemperature, toPredictions } from '@m-sanchez/calibrated';

const fit = fitTemperature(logitSamples);   // { temperature, nllBefore, nllAfter, improved, atBound }
const recalibrated = toPredictions(logitSamples, fit.temperature);
```

`atBound` is `'lo'` or `'hi'` when the search converged onto the edge of
its own bracket: what came back is a boundary, not a fit, and the real
optimum lies outside `[lo, hi]` - widen it with
`fitTemperature(samples, { hi: 200 })` and refit. Labels and logits are
validated, so a 1-indexed label column throws rather than quietly
returning that bracket edge with `nllBefore: NaN`.

`npm run demo` on a seeded, wildly overconfident 4-class model:

```
before scaling         ECE 0.284   Brier 0.284   accuracy 71.6%
after (T=4.45)         ECE 0.006   Brier 0.204   accuracy 71.6%
```

The calibration error fell by 40x and not one prediction changed.

## Honest limits

- These are confidence-calibration measures over the model's top
  prediction, not full multi-class calibration (classwise-ECE) or
  regression calibration.
- ECE and the Brier decomposition depend on the bin count; the raw Brier
  score does not. Report the binning you used, and prefer equal-mass when
  confidence is top-heavy.
- ECE is positively biased: the bias grows with the bin count and shrinks
  with the sample size, and it does not vanish for a perfect model. Read
  it next to `eceInterval` and `nullEce`, not alone.
- No predictions is not perfect calibration. `calibrationError([])` and
  `brier([])` return NaN, not 0, so an empty slice of a dashboard cannot
  pass an `ece <= x` bar.
- Temperature scaling needs logits (per-class scores), not just the final
  confidence; the metrics need only `(confidence, correct)`.

## Run

```bash
npm install       # dev-only: typescript
npm test
npm run demo
npm run floor     # the null-ECE table above, remeasured
npm run typecheck
```

Install: `npm install @m-sanchez/calibrated` (or a pinned git tag,
`github:m-sanchez/calibrated#v2.0.0`; CI proves the packed tarball imports).
Node 22.18+, zero runtime dependencies.

## The tests are the point

| Test | Claim |
| :-- | :-- |
| a perfectly calibrated set has ~zero ECE | the metric bottoms out where it should |
| an overconfident set surfaces its exact gap | ECE reads the miscalibration, not noise |
| MCE reports the worst bin, not the average | the least-trustworthy confidence is named |
| temperature scaling cuts ECE without moving accuracy | recalibration is free of the decision |
| an already-calibrated model gets T near 1 | scaling does nothing when there is nothing to fix |
| a random-guess predictor has ~zero resolution | the Brier decomposition separates the two failure modes |
| softmax is stable with huge logits | no overflow between the model and the metric |
| softmax over 200k classes stays normalised | token-level calibration of an LLM actually runs |
| equal-mass ECE is identical across 50 shuffles of the same rows | the metric is a function of the data, not of row order |
| all-tied confidences read their exact gap, not an inflated one | a repeated confidence is never split across two bins |
| a bootstrap interval covers a known gap and reproduces from its seed | an ECE reading arrives with its uncertainty |
| every cell of the noise-floor table is what the simulation reports | the table above is measured, not asserted |
| a fit pinned to the bracket edge is reported as pinned | a boundary is never returned as a success |
| a 1-indexed label throws | the commonest data-prep mistake is refused, not absorbed |
| an empty set reports NaN | no data cannot pass a calibration bar |

Every claim on this page is mapped to the test that enforces it in
[CLAIMS.md](CLAIMS.md).
