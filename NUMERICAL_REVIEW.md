# Numerical corrections in 2.0.1

The `2.0.1` corrections build on commit `55c8175b3609b6561e45aebfd563afdb85f3d9f7`. The changes preserve the exported function names while correcting numerical edge cases and making search termination explicit.

- NLL now uses centered log-sum-exp without the former probability floor. `[0, -1000]` with true class 1 returns 1000 instead of 27.631.
- Centering occurs before temperature scaling, with an overflow fallback for representable scaled differences. Probabilities remain defined for huge equal logits.
- Decisions use raw-logit argmax with the first index winning exact ties. Rounded softmax ties no longer change correctness.
- Finite positive temperature, consistent class dimensions, valid labels, ordered finite bounds, positive tolerance and bounded iteration limits are enforced.
- Empty NLL returns NaN; empty fitting and unrepresentable NLL throw RangeError. Existing exports and fit fields remain available.
- Fit results add `status` and `iterations`. Status is `converged`, `boundary`, `constant`, `max-iterations` or `stalled`. Constant objectives retain T=1 where the bounds allow it and do not claim improvement.
- The demo fits on 3,000 calibration rows from seed 7 and evaluates separate 3,000 test rows from seed 19. Documentation reports the held-out numbers without a general improvement claim.

## Validation

Run `npm ci`, `npm test`, `npm run typecheck` and `npm run build` with Node 22.18 or newer. The independent check is `python validation/verify_scipy.py`; it requires NumPy and SciPy and records versions and results in `validation/scipy-report.json`.

Closed-form fixtures cover loss 1000, log(2), log(4), tiny positive loss, probabilities 0.75/0.25, a known analytic optimum, empty data, invalid options, overflow, class ties, shift invariance and search termination. SciPy checks 403 probability/loss fixtures and three independently optimized calibration datasets.

## Compatibility notes

Empty `nll` changes from 0 to NaN, empty fitting throws instead of returning a no-op result, and `softmax([])` throws instead of returning an empty array. Inputs with infinite temperatures, inconsistent class counts or invalid search settings now throw. Consumers that depended on those values or permissive inputs must handle the explicit error or missing-data result.

`TemperatureFit` adds required `status` and `iterations` fields. Code that constructs this interface directly must supply them; existing code that reads returned fits retains the earlier fields.

## Integration limitations

`improved` retains the existing meaning of non-increased calibration NLL within numerical tolerance; it does not imply held-out improvement or an acceptable fit status. Consumers must inspect `status` before treating the temperature as a completed fit. A boundary result does not establish the location of an unconstrained optimum.

Numerical-range errors are explicit exceptions. Source input validation does not establish dataset independence, split provenance, class-order consistency between separate datasets or policy evaluation validity; those remain the workbench's responsibility.

The corrections have been checked on this machine's Node 24 runtime. Check the source revision's CI result for the Node 22/24/26 matrix; the local checks alone do not establish cross-version compatibility. SciPy is used for development verification only and is not a runtime dependency.
