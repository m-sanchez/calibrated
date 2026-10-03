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
  console.log('\nmedian / p95 describe this simulation; neither establishes calibration or a deployment threshold.');
}
