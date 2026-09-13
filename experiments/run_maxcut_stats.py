"""Statistics for the Max-Cut comparison (three-way results, ratios, CIs).

Reads results/analysis/maxcut_comparison.csv (canonical run) and
results/analysis/maxcut_ablation.csv (controls, bounds). Prints:

  - mean cut / num_edges per method
  - win / tie / loss of each method against every other method
  - pairwise GNN head-to-heads: mean ratio, 95% paired-bootstrap CI,
    Wilcoxon signed-rank p
  - ratio to the SDP relaxation objective (upper-bound proxy)
  - ratio to the exact optimum for n <= 20 (rigorous certificate)
  - in-distribution (ER n=100) vs out-of-distribution split
  - multi-seed spread if seed-suffixed CSVs are present

Run: python3 experiments/run_maxcut_stats.py
"""

import sys, glob, warnings
from pathlib import Path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats

METHODS = ['random_cut', 'rand_nols', 'rand50_nols', 'rand_ls', 'rand50_ls',
           'greedy', 'gnn_nols', 'gnn_ls', 'spectral_norefine', 'spectral',
           'gw', 'gw_plus_ls']


def boot_ci(vals, n_boot=10000, seed=0, alpha=0.05):
    rng = np.random.RandomState(seed)
    vals = np.asarray(vals)
    n = len(vals)
    means = np.array([vals[rng.randint(0, n, n)].mean() for _ in range(n_boot)])
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def three_way(df, col):
    """Win/tie/loss of `col` against every other method (higher = better)."""
    others = [c for c in METHODS if c != col and c in df.columns]
    other_max = df[others].max(axis=1)
    return ((df[col] > other_max).mean(),
            (df[col] == other_max).mean(),
            (df[col] < other_max).mean())


def main():
    df = pd.read_csv('results/analysis/maxcut_comparison.csv')
    ab = pd.read_csv('results/analysis/maxcut_ablation.csv')
    d = df.merge(ab, on=['family', 'n', 'instance', 'm'], suffixes=('', '_ab'))
    # The comparison CSV is the canonical run; the ablation CSV's gnn_ls
    # alias must be identical to it (fail loudly rather than overwrite).
    assert (d['gnn_ls'] == d['gnn_cut']).all(), \
        "ablation gnn_ls disagrees with the canonical comparison CSV"
    print(f'instances: {len(d)}')

    print('\n=== mean cut / num_edges (unweighted) ===')
    for c in METHODS:
        if c in d.columns:
            print(f'  {c:18s} {(d[c]/d.m).mean():.4f}')
    print(f'  {"sdp_bound":18s} {(d.sdp_bound/d.m).mean():.4f}  (upper-bound proxy)')
    nn = d[d.exact_opt.notna()]
    print(f'  {"exact_opt (n<=20)":18s} {(nn.exact_opt/nn.m).mean():.4f}  ({len(nn)} instances)')

    print('\n=== win / tie / loss vs every other method ===')
    for c in METHODS:
        if c in d.columns:
            w, t, l = three_way(d, c)
            print(f'  {c:18s} {w:.1%} / {t:.1%} / {l:.1%}')

    print('\n=== sample-budget confound (raw output) ===')
    gap1 = (d.gnn_nols / d.m).mean() - (d.rand_nols / d.m).mean()
    gap50 = (d.gnn_nols / d.m).mean() - (d.rand50_nols / d.m).mean()
    print(f'  raw GNN minus 1-sample coin flip : {gap1:.4f} of edges')
    print(f'  raw GNN minus 50-sample coin flip: {gap50:.4f} of edges')
    print(f'  -> {1 - gap50 / gap1:.0%} of the raw gap is the control\'s sample budget')

    print('\n=== pairwise GNN head-to-heads (mean ratio, 95% bootstrap CI, Wilcoxon) ===')
    for other in ['rand_ls', 'rand50_ls', 'spectral', 'gw', 'gw_plus_ls']:
        ratio = (d.gnn_ls / d[other]).values
        lo, hi = boot_ci(ratio, seed=1)
        n_zero = int((d.gnn_ls == d[other]).sum())
        n_nonzero = len(d) - n_zero
        _, p = stats.wilcoxon(d.gnn_ls, d[other], zero_method='wilcox')
        print(f'  vs {other:10s}: {ratio.mean():.4f} [{lo:.4f}, {hi:.4f}]  '
              f'p={p:.3g}  (Wilcoxon drops {n_zero} exact ties, n={n_nonzero})')

    print('\n=== ratio to SDP bound (upper-bound proxy; mean, 95% CI) ===')
    for c in ['gnn_ls', 'spectral', 'gw', 'rand_ls', 'gw_plus_ls']:
        r = (d[c] / d.sdp_bound).values
        lo, hi = boot_ci(r, seed=2)
        print(f'  {c:12s} {r.mean():.4f} [{lo:.4f}, {hi:.4f}]')

    print('\n=== ratio to EXACT optimum, n<=20 (rigorous; 1.0 = optimal) ===')
    for c in ['gnn_ls', 'spectral', 'gw', 'rand_ls', 'gw_plus_ls']:
        sub = nn.dropna(subset=[c])
        r = (sub[c] / sub.exact_opt).values
        lo, hi = boot_ci(r, seed=3)
        print(f'  {c:12s} {r.mean():.4f} [{lo:.4f}, {hi:.4f}]')

    print('\n=== in-distribution vs out-of-distribution (canonical GNN) ===')
    for label, sub in [('ID  (erdos_renyi n=100)', d[(d.family == 'erdos_renyi') & (d.n == 100)]),
                       ('OOD (everything else)', d[~((d.family == 'erdos_renyi') & (d.n == 100))])]:
        w, t, l = three_way(sub, 'gnn_ls')
        print(f'  {label} ({len(sub)}): rel-to-best {(sub.gnn_ls/sub.best_classical_cut).mean():.3f}, '
              f'ratio-to-bound {(sub.gnn_ls/sub.sdp_bound).mean():.3f}, '
              f'W/T/L {w:.1%}/{t:.1%}/{l:.1%}')

    print('\n=== multi-seed spread ===')
    for f in sorted(glob.glob('results/analysis/maxcut_comparison_seed*.csv')):
        s = pd.read_csv(f)
        print(f"  {Path(f).name}: wins {int(s.gnn_wins.sum())}/{len(s)} "
              f"({s.gnn_wins.mean():.1%}), rel-to-best "
              f"{(s.gnn_cut/s.best_classical_cut).mean():.3f}")


if __name__ == '__main__':

    main()
