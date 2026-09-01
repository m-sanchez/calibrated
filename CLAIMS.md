# CLAIMS

Every externally falsifiable claim this package makes - on the README and
in the package description - and the executable check that enforces it. If
a claim is not in this table, it is not made. If a check here fails, the
front page is wrong and should be changed before the code ships.

Run them with `npm test`, `npm run typecheck` and `npm run build`; the
packaging claims are enforced by `.github/workflows/test.yml` on Node 22,
24 and 26.

## Product and packaging

| Claim | Where | Enforced by |
| :-- | :-- | :-- |
| Zero runtime dependencies | description; badge; README "Run" | `test/claims.test.ts::the package has no runtime dependencies` |
| Runs on Node 22.18+ | badge; README "Run" | `.github/workflows/test.yml` (matrix `node: [22, 24, 26]` runs the full suite) |
| Erasable TypeScript syntax only | badge | `npm run typecheck` with `erasableSyntaxOnly: true` in `tsconfig.json`, run in CI |
| CI proves the packed tarball imports | README "Run" | `.github/workflows/test.yml::install proof` - packs, installs into a scratch project, imports |
| Ships ECE, Brier decomposition, reliability diagrams and temperature scaling | description | the rows below, one per measure |

## The metrics

| Claim | Where | Enforced by |
| :-- | :-- | :-- |
| ECE is the confidence-vs-accuracy gap weighted by bin population, and bottoms out at zero for an honest model | README bullets | `test/metrics.test.ts::a perfectly calibrated set has ~zero calibration error` |
| ECE reads a known miscalibration exactly | README bullets; tests table | `test/metrics.test.ts::an overconfident set surfaces its exact gap` - 200 predictions at 0.9, 60% correct, asserts exactly 0.30 |
| MCE is the single worst bin, not the average | README bullets; tests table | `test/metrics.test.ts::MCE reports the single worst bin, not the average` |
| Brier score is the raw mean squared error of the probabilities | README bullets | `test/metrics.test.ts::Brier score is the exact mean squared error of the probabilities` |
| Murphy's decomposition separates *miscalibrated* from *uninformative* | README bullets; tests table | `test/metrics.test.ts::Brier decomposition: a random-guess predictor has ~zero resolution` |
| ECE and the decomposition depend on the bin count; the raw Brier score does not | README "Honest limits" | `test/claims.test.ts::ECE depends on the bin count; the raw Brier score does not` |
| Reliability-diagram data is per-bin mean confidence vs observed accuracy | README bullets | `test/metrics.test.ts::a perfectly calibrated set has ~zero calibration error` - the bins sit on the diagonal by construction |
| Confidences outside [0, 1] are refused | implied by the `(confidence, correct)` contract | `test/metrics.test.ts::binning rejects invalid inputs at the boundary` |
| No predictions is no evidence: `calibrationError([])`, `brier([])` and `eceInterval([])` return NaN, not 0 | README "Honest limits"; tests table | `test/metrics.test.ts::an empty set reports no evidence, not perfect calibration`; `test/interval.test.ts::an interval over no data is no evidence, not a confident zero` |

## Binning

| Claim | Where | Enforced by |
| :-- | :-- | :-- |
| Equal-mass keeps every bin's estimate on comparable footing when confidence piles up near 1.0 | README bullets; "Honest limits" | `test/binning.test.ts::equal-mass keeps every bin on comparable footing when confidence is top-heavy` - 10 bins of exactly 200 where equal-width puts 95% of the data in one bin; `test/metrics.test.ts::equal-mass binning keeps every bin populated when confidence is top-heavy` |
| A repeated confidence is never split across a bin boundary | README bullets; tests table | `test/binning.test.ts::equal-mass never splits a run of tied confidences`; `test/binning.test.ts::a confidence value never lands in two different equal-mass bins` |
| The answer does not depend on the order the rows arrived in | README bullets; tests table | `test/binning.test.ts::equal-mass calibration error is a function of the data, not of row order` - 50 seeded shuffles, identical to 1e-12 |
| Fewer distinct confidences than bins gives fewer bins, and `effectiveBins` says how many | README bullets | `test/metrics.test.ts::the report says how many bins actually carried data` |

## Uncertainty

| Claim | Where | Enforced by |
| :-- | :-- | :-- |
| ECE is biased upward; the bias grows with bins and shrinks with n | README "Is the number real?"; "Honest limits" | `test/interval.test.ts::the null ECE floor falls with sample size and rises with bin count` |
| `eceInterval` is a resample-with-replacement bootstrap reporting the middle `level` of the ECEs | README "Is the number real?" | `test/interval.test.ts::the bootstrap interval brackets the point estimate and covers the true gap`; `test/interval.test.ts::a higher confidence level gives a wider interval` |
| `nullEce` simulates an honest model over *your* confidences and *your* binning | README "Is the number real?" | `test/interval.test.ts::the null floor is computed against the confidences you actually have` |
| Both reproduce exactly from their seed | README "Is the number real?" | `test/interval.test.ts::the interval reproduces exactly under a fixed seed` |
| Every cell of the published noise-floor table | README table | `test/interval.test.ts::the published noise-floor table is what nullEce reports` - all 12 cells, median and p95, to 4dp |
| A bare `ece <= 0.1` bar rejects a perfectly calibrated 100-item set in 31% of draws; 3% at n=200; not once in 2000 at n=400 | README "Is the number real?" | `test/interval.test.ts::a 0.1 ship bar rejects a perfectly calibrated 100-item eval set a third of the time` |

## Temperature scaling

| Claim | Where | Enforced by |
| :-- | :-- | :-- |
| `T > 1` softens an overconfident model and lowers NLL | README "Fix it" | `test/temperature.test.ts::fitting temperature on an overconfident model returns T > 1 and lowers NLL` |
| The argmax never moves, so accuracy is unchanged | README "Fix it"; tests table | `test/temperature.test.ts::temperature scaling reduces calibration error without moving accuracy` |
| Scaling does nothing when there is nothing to fix | tests table | `test/temperature.test.ts::an already-calibrated model gets a temperature near 1` |
| `atBound` reports a fit that converged onto the bracket edge, and is null for an interior fit | README "Fix it"; tests table | `test/temperature.test.ts::fitTemperature says when the optimum is pinned to the bracket`; `test/temperature.test.ts::a fit that lands inside the bracket reports no bound` |
| A 1-indexed label column throws rather than returning the bracket edge | README "Fix it"; tests table | `test/temperature.test.ts::fitTemperature refuses a label outside the logit range`; `test/temperature.test.ts::fitTemperature refuses malformed logits` |
| softmax is stable with huge logits | tests table | `test/temperature.test.ts::softmax sums to one and is stable with huge logits`; `test/temperature.test.ts::nll is finite even when a label has vanishing probability` |
| softmax over a 200k-class vocabulary stays normalised | tests table | `test/temperature.test.ts::softmax survives a real vocabulary, not just huge magnitudes` |
| A non-finite logit is refused, not silently turned into NaN | README "Fix it" | `test/temperature.test.ts::softmax refuses a non-finite logit instead of returning NaN` |
| A non-positive temperature is refused | implied by `T > 0` | `test/temperature.test.ts::softmax rejects a non-positive temperature` |
| The demo's printed before/after numbers, and the 40x fall in ECE | README "Fix it" | `test/claims.test.ts::the demo prints the numbers the README quotes` - runs `demo/demo.ts` and matches the quoted lines |
| Temperature scaling needs logits; the metrics need only `(confidence, correct)` | README "Honest limits" | the exported signatures, checked by `npm run typecheck` in CI |

## Claims with no enforcing test

| Claim | Why not |
| :-- | :-- |
| "Worked example: routing-study" (header link) | Cross-repo, and routing-study pins `#v1.0.1` of this package. Turning that study's scored outputs into an in-repo fixture with pinned expected values is owned by a later pass; until then the link is a pointer, not a proved claim. |
| "a fresh, dependency-free implementation of standard methods"; "First published 2026-08-31" | Provenance, not behaviour. The measures are pinned against constructions with closed-form answers - a band at confidence 0.9 that is 60% correct has ECE exactly 0.30, Brier is the exact MSE - rather than against an external reference implementation. There is no parity fixture in this repo. |
| "Modern classifiers usually are not [calibrated] - they are overconfident" | A statement about the world, cited to Guo et al. (2017), not about this code. |
