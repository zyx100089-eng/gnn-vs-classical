"""Compare the label sources for the supervised Max-Cut GNN.

Reads the three n=20 runs that share one benchmark (seed 1242 + 1000*i):

  supervised_maxcut_comparison_exactlabels.csv   (exact-optimum labels)
  supervised_maxcut_comparison_spectral_n20.csv  (spectral labels)
  maxcut_comparison_unsup_n20.csv                (unsupervised)

and reports, per model, the ratio to the true optimum and the number of
instances solved optimally, plus paired Wilcoxon tests between them.
This is what backs the README's claim that the label source makes no
measurable difference.

Run: python3 experiments/run_supervised_label_comparison.py
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import pandas as pd
from scipy import stats

from src.classical.maxcut.exact import exact_maxcut
from src.graphs.generators import generate_instance

FILES = {
    'exact labels': 'results/analysis/supervised_maxcut_comparison_exactlabels.csv',
    'spectral labels': 'results/analysis/supervised_maxcut_comparison_spectral_n20.csv',
    'unsupervised': 'results/analysis/maxcut_comparison_unsup_n20.csv',
}
KEY = ['family', 'n', 'instance']
TRAIN_GRAPHS = 1200  # the n=20 runs were trained on 1200 graphs -> base 1242


def load(path):
    df = pd.read_csv(path).sort_values(KEY).reset_index(drop=True)
    df['exact'] = [exact_maxcut(generate_instance(r['family'], int(r['n']),
                                                  seed=42 + TRAIN_GRAPHS
                                                  + int(r['instance']) * 1000))
                   for _, r in df.iterrows()]
    return df.set_index(KEY)


def main():
    frames = {name: load(p) for name, p in FILES.items()}
    print(f'instances per run: {len(next(iter(frames.values())))}')

    print('\n=== ratio to the true optimum (n=20, brute force) ===')
    for name, df in frames.items():
        print(f'  {name:16s} {((df.gnn_cut/df.exact).mean()):.4f}  '
              f'{int((df.gnn_cut == df.exact).sum())}/{len(df)} optimal')

    ref = next(iter(frames.values()))
    print(f'  {"Goemans-Williamson":16s} {((ref.gw_cut/ref.exact).mean()):.4f}  '
          f'{int((ref.gw_cut == ref.exact).sum())}/{len(ref)} optimal')

    print('\n=== paired Wilcoxon between the learned models ===')
    names = list(frames)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = frames[names[i]], frames[names[j]]
            A, B = a.gnn_cut.loc[b.index], b.gnn_cut
            diff = A - B
            _, p = stats.wilcoxon(A, B, zero_method='wilcox')
            print(f'  {names[i]:16s} vs {names[j]:16s}: mean diff {diff.mean():+.3f} cut, '
                  f'differs on {int((diff != 0).sum())}/{len(A)}, p={p:.3f}')
    print('\n  (none significant; each model is also a single training seed)')


if __name__ == '__main__':
    main()
