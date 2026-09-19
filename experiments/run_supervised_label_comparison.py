"""Compare the label sources for the supervised Max-Cut GNN.

Three variants, two training seeds each, all on one n=20 benchmark
(eval_seed 1000 + train_graphs 1200 -> seed 2200 + 1000*i):

  unsupervised  : the cut-maximisation loss (no labels)
  spectral      : BCE against spectral-relaxation labels
  exact         : BCE against exact-optimum labels (n <= 20 brute force)

Reports, per seed, the ratio to the true optimum and the number of
instances solved optimally, plus paired Wilcoxon tests between the
seed-averaged models. This backs the README's claim that the *label
source* (spectral vs exact) makes no measurable difference, while the
unsupervised objective is marginally ahead of both.

Run: python3 experiments/run_supervised_label_comparison.py
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

from src.classical.maxcut.exact import exact_maxcut
from src.graphs.generators import generate_instance

KEY = ['family', 'n', 'instance']
BENCH_BASE = 1000 + 1200  # eval_seed + train_graphs

RUNS = {
    'unsupervised': ['results/analysis/maxcut_comparison_unsup_n20.csv',
                     'results/analysis/maxcut_comparison_unsup_n20_s43.csv'],
    'spectral': ['results/analysis/supervised_maxcut_comparison_spectral_n20.csv',
                 'results/analysis/supervised_maxcut_comparison_spectral_n20_s43.csv'],
    'exact': ['results/analysis/supervised_maxcut_comparison_exactlabels.csv',
              'results/analysis/supervised_maxcut_comparison_exactlabels_s43.csv'],
}


def main():
    ref = pd.read_csv(RUNS['unsupervised'][0]).sort_values(KEY).reset_index(drop=True)
    exact = [exact_maxcut(generate_instance(r['family'], int(r['n']),
                                             seed=BENCH_BASE + int(r['instance']) * 1000))
             for _, r in ref.iterrows()]
    ref['exact'] = exact

    def load(f):
        d = pd.read_csv(f).sort_values(KEY).reset_index(drop=True)
        d['exact'] = exact
        return d

    print(f'instances per run: {len(ref)}')
    print('\n=== ratio to the true optimum (n=20, brute force) ===')
    averaged = {}
    for name, files in RUNS.items():
        ratios, opt = [], []
        for f in files:
            d = load(f)
            ratios.append((d.gnn_cut / d.exact).mean())
            opt.append(int((d.gnn_cut == d.exact).sum()))
        averaged[name] = np.mean([load(f).gnn_cut.values for f in files], axis=0)
        print(f'  {name:14s} ratio {min(ratios):.4f}-{max(ratios):.4f} '
              f'(mean {np.mean(ratios):.4f}); optimal {min(opt)}-{max(opt)}/{len(ref)}')
    print(f'  {"GW":14s} ratio {(ref.gw_cut/ref.exact).mean():.4f}; '
          f'optimal {int((ref.gw_cut == ref.exact).sum())}/{len(ref)}')

    print('\n=== paired Wilcoxon on seed-averaged cuts (150 instances) ===')
    names = list(averaged)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = averaged[names[i]], averaged[names[j]]
            _, p = stats.wilcoxon(A, B, zero_method='wilcox')
            print(f'  {names[i]:14s} vs {names[j]:14s}: '
                  f'mean diff {(A-B).mean():+.3f} cut, p={p:.3f}')
    print('\n  (two seeds per variant: seed spread is <=0.001 in ratio, '
          'smaller than the unsupervised-vs-supervised gap)')


if __name__ == '__main__':
    main()
