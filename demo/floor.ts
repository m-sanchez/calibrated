/** npm run floor: the ECE a perfectly calibrated model still reports.
 *
 * Every row is a simulation, not a claim: `nullEce` draws a model whose
 * confidences are honest by construction, measures its ECE anyway, and
 * reports the middle and the 95th percentile of that distribution. Read a
 * column downwards to see the floor fall with sample size, read a row
 * across to see it rise with bin count. Pass your own numbers as
 * `npm run floor -- <n> <bins>`. */

import { nullEce } from '../src/interval.ts';

const [argN, argBins] = process.argv.slice(2).map(Number);

if (Number.isFinite(argN) && Number.isFinite(argBins)) {
  const r = nullEce({ confidences: argN, bins: argBins });
  console.log(`n=${r.n}, ${r.bins} bins: median ECE ${r.median.toFixed(4)}, p95 ${r.p95.toFixed(4)}`);
} else {
  const sizes = [100, 200, 400, 1000];
  const bins = [10, 15, 30];
  console.log('null ECE (perfectly calibrated, confidence ~ U(0.5, 1), 2000 draws)\n');
  console.log(['    n', ...bins.map((b) => `${b} bins`.padStart(16))].join(''));
  for (const n of sizes) {
    const cells = bins.map((b) => {
      const r = nullEce({ confidences: n, bins: b });
      return `${r.median.toFixed(4)} / ${r.p95.toFixed(4)}`.padStart(16);
    });
    console.log([`${n}`.padStart(5), ...cells].join(''));
  }
  console.log('\nmedian / p95. An observed ECE under the p95 is not evidence of miscalibration.');
}
