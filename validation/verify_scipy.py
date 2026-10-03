import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


root = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(20261002)
fixtures = []
for classes in [2, 3, 8, 32]:
    for temperature in [0.05, 0.2, 1.0, 7.0, 20.0]:
        for _ in range(20):
            logits = rng.normal(0, 12, classes) + rng.choice([-10000, 0, 10000])
            fixtures.append({"logits": logits.tolist(), "label": int(rng.integers(classes)), "temperature": temperature})
fixtures.extend([
    {"logits": [0, -1000], "label": 1, "temperature": 1},
    {"logits": [0, 0], "label": 1, "temperature": 1},
    {"logits": [0, -40], "label": 0, "temperature": 1},
])
fits = []
for classes, scale in [(2, 0.6), (3, 3), (8, 5)]:
    base_logits = rng.normal(0, 1.2, (1000, classes))
    labels = [int(rng.choice(classes, p=softmax(row))) for row in base_logits]
    fits.append([{"logits": (row * scale).tolist(), "label": label} for row, label in zip(base_logits, labels)])

javascript = """
import { readFileSync } from 'node:fs';
import { nll, softmax, fitTemperature } from './dist/index.js';
const input = JSON.parse(readFileSync(0, 'utf8'));
console.log(JSON.stringify({
  fixtures: input.fixtures.map(({logits,label,temperature}) => ({
    probabilities: softmax(logits,temperature), nll: nll([{logits,label}],temperature)
  })),
  fits: input.fits.map(rows => fitTemperature(rows,{tolerance:1e-7}))
}));
"""
run = subprocess.run(
    ["node", "--input-type=module", "-e", javascript],
    cwd=root,
    input=json.dumps({"fixtures": fixtures, "fits": fits}),
    text=True,
    capture_output=True,
    check=True,
)
actual = json.loads(run.stdout)
probability_error = 0.0
nll_error = 0.0
for fixture, result in zip(fixtures, actual["fixtures"]):
    logits = np.asarray(fixture["logits"], dtype=np.float64)
    scaled = (logits - logits.max()) / fixture["temperature"]
    expected_probabilities = softmax(scaled)
    expected_nll = float(logsumexp(scaled) - scaled[fixture["label"]])
    probability_error = max(probability_error, float(np.max(np.abs(expected_probabilities - result["probabilities"]))))
    nll_error = max(nll_error, abs(expected_nll - result["nll"]))
    np.testing.assert_allclose(result["probabilities"], expected_probabilities, atol=2e-14, rtol=2e-13)
    np.testing.assert_allclose(result["nll"], expected_nll, atol=2e-12, rtol=2e-13)

fit_results = []
for rows, result in zip(fits, actual["fits"]):
    logits = np.asarray([row["logits"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    def objective(temperature):
        scaled = (logits - logits.max(axis=1, keepdims=True)) / temperature
        return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(rows)), labels]))
    expected = minimize_scalar(objective, bounds=(0.05, 20), method="bounded", options={"xatol": 1e-10})
    assert expected.success
    assert result["status"] == "converged"
    assert abs(result["temperature"] - expected.x) < 2e-5
    assert abs(result["nllAfter"] - expected.fun) < 2e-12
    fit_results.append({
        "classes": len(rows[0]["logits"]),
        "samples": len(rows),
        "temperature": result["temperature"],
        "scipy_temperature": float(expected.x),
        "objective_difference": abs(result["nllAfter"] - float(expected.fun)),
    })

report = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "seed": 20261002,
    "fixture_count": len(fixtures),
    "maximum_probability_absolute_error": probability_error,
    "maximum_nll_absolute_error": nll_error,
    "fits": fit_results,
    "passed": True,
}
destination = root / "validation" / "scipy-report.json"
destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
