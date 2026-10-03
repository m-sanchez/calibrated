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
| ECE weights empirical confidence-vs-accuracy gaps by bin population; a constructed set with zero occupied-bin gaps has approximately zero ECE | README bullets; tests table | `test/metrics.test.ts::a perfectly calibrated set has ~zero calibration error` |
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
| The stated calibrated simulation produces positive ECE, with different reference distributions as sample size and bins change | README "How much can ECE vary?"; "Honest limits" | `test/interval.test.ts::the null ECE floor falls with sample size and rises with bin count` |
| `eceInterval` resamples whole rows with replacement and reports the middle `level` of the resulting ECEs | README "How much can ECE vary?" | `test/interval.test.ts::the bootstrap interval brackets the point estimate and covers the true gap`; `test/interval.test.ts::a higher confidence level gives a wider interval` - these fixtures do not establish general population coverage |
| Array-input `nullEce` fixes confidences and draws independent Bernoulli correctness under the selected binning | README "How much can ECE vary?" | `test/interval.test.ts::the null floor is computed against the confidences you actually have` |
| Both summaries reproduce with the same data, settings and seed | README "How much can ECE vary?" | `test/interval.test.ts::the interval reproduces exactly under a fixed seed`; `test/interval.test.ts::the null floor is computed against the confidences you actually have` |
| Every cell of the published null-simulation table | README table | `test/interval.test.ts::the published noise-floor table is what nullEce reports` - all 12 cells, median and p95, to 4dp |
| Under the stated uniform-confidence generator and seed, ECE exceeds 0.1 in about 31% of draws at n=100, about 3% at n=200 and zero of 2000 draws at n=400 | README "How much can ECE vary?" | `test/interval.test.ts::a 0.1 ship bar rejects a perfectly calibrated 100-item eval set a third of the time` - simulation frequencies, not population guarantees |

## Temperature scaling

| Claim | Where | Enforced by |
| :-- | :-- | :-- |
| `T > 1` softens an overconfident model and lowers NLL | README "Fix it" | `test/temperature.test.ts::fitting temperature on an overconfident model returns T > 1 and lowers NLL` |
| Raw-logit argmax is preserved, including rounded probability ties | README "Fix it"; tests table | `test/temperature.test.ts::temperature scaling reduces calibration error without moving accuracy`; `test/numerical.test.ts::raw-logit decisions survive probability rounding and exact ties` |
| Scaling does nothing when there is nothing to fix | tests table | `test/temperature.test.ts::an already-calibrated model gets a temperature near 1` |
| `atBound` reports a fit that converged onto the bracket edge, and is null for an interior fit | README "Fix it"; tests table | `test/temperature.test.ts::fitTemperature says when the optimum is pinned to the bracket`; `test/temperature.test.ts::a fit that lands inside the bracket reports no bound` |
| A 1-indexed label column throws rather than returning the bracket edge | README "Fix it"; tests table | `test/temperature.test.ts::fitTemperature refuses a label outside the logit range`; `test/temperature.test.ts::fitTemperature refuses malformed logits` |
| softmax is stable with huge logits | tests table | `test/temperature.test.ts::softmax sums to one and is stable with huge logits`; `test/temperature.test.ts::nll is finite even when a label has vanishing probability` |
| softmax over a 200k-class vocabulary stays normalised | tests table | `test/temperature.test.ts::softmax survives a real vocabulary, not just huge magnitudes` |
| A non-finite logit is refused, not silently turned into NaN | README "Fix it" | `test/temperature.test.ts::softmax refuses a non-finite logit instead of returning NaN` |
| Non-finite and non-positive temperatures are refused at every entry point | README "Fix it" | `test/numerical.test.ts::every temperature entry point validates temperature even without data` |
| The demo's printed held-out before/after numbers | README "Fix it" | `test/claims.test.ts::the demo prints the numbers the README quotes` - checks the separate calibration/test seeds and reported test metrics |
| NLL is not clipped at a probability floor | README "Fix it" | `test/numerical.test.ts::NLL matches independent closed-form answers without clipping`; `validation/verify_scipy.py` |
| Search termination and constant objectives have explicit statuses | README "Fix it" | `test/numerical.test.ts::constant objectives and search termination have explicit statuses` |
| Empty NLL is NaN and empty fitting is refused | README "Honest limits" | `test/numerical.test.ts::every temperature entry point validates temperature even without data` |
| Temperature scaling needs logits; the metrics need only `(confidence, correct)` | README "Honest limits" | the exported signatures, checked by `npm run typecheck` in CI |

## Claims with no enforcing test

| Claim | Why not |
| :-- | :-- |
| "Worked example: routing-study" (header link) | Cross-repo, and routing-study pins `#v1.0.1` of this package. Turning that study's scored outputs into an in-repo fixture with pinned expected values is owned by a later pass; until then the link is a pointer, not a proved claim. |
| "a fresh, dependency-free implementation of standard methods"; "First published 2026-08-31" | Provenance, not behaviour. Closed-form metric fixtures are complemented by independent SciPy softmax, log-sum-exp and optimization checks in `validation/verify_scipy.py`. |
| "Modern classifiers usually are not [calibrated] - they are overconfident" | A statement about the world, cited to Guo et al. (2017), not about this code. |
