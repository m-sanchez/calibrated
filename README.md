# calibrated

![TypeScript](https://img.shields.io/badge/TypeScript-erasable_syntax-3178C6?logo=typescript&logoColor=white)
![Node](https://img.shields.io/badge/node-%3E%3D22.18-5FA04E?logo=nodedotjs&logoColor=white)
![Dependencies](https://img.shields.io/badge/dependencies-0-B45309)
[![CI](https://github.com/m-sanchez/calibrated/actions/workflows/test.yml/badge.svg)](https://github.com/m-sanchez/calibrated/actions/workflows/test.yml)
![License](https://img.shields.io/badge/license-MIT-6E6E6E)

Is your model's confidence honest? ECE, Brier decomposition, reliability
diagrams, and temperature scaling - zero dependencies.

[More tools](https://github.com/m-sanchez) · [Working rules](https://miguelsanchez.co.uk/ethics)

A model is *calibrated* when the confidence it reports matches how often it
is right: among the predictions it makes at 90% confidence, 90% should be
correct. Modern classifiers usually are not - they are overconfident - and
an overconfident model that also refuses or escalates on low confidence is
making those decisions on numbers that do not mean what they say. This
measures the gap and closes it, from labelled `(confidence, correct)`
pairs, with a careful dependency-free implementation of standard measures.

```ts
import { calibrationError, brier, reliabilityDiagram } from 'calibrated';

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
  up near 1.0, which it always does.

## Fix it: temperature scaling

Temperature scaling (Guo et al., 2017) recalibrates without touching a
single decision: divide every logit by one learned scalar `T` before the
softmax. `T > 1` softens overconfidence; the argmax never moves, so
accuracy is unchanged. `T` is fit by minimising NLL on a held-out set,
here by a dependency-free golden-section search.

```ts
import { fitTemperature, toPredictions } from 'calibrated';

const fit = fitTemperature(logitSamples);   // { temperature, nllBefore, nllAfter }
const recalibrated = toPredictions(logitSamples, fit.temperature);
```

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
- Temperature scaling needs logits (per-class scores), not just the final
  confidence; the metrics need only `(confidence, correct)`.

## Run

```bash
npm install       # dev-only: typescript
npm test
npm run demo
npm run typecheck
```

Install: `npm install github:m-sanchez/calibrated#v1.0.0` (not yet on npm;
CI proves the packed tarball imports). Node 22.18+, zero runtime
dependencies.

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
