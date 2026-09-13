"""Max-Cut ablation study: what actually produces the performance?

Controls and instruments, all on the same 600 evaluation instances as
results/analysis/maxcut_comparison.csv (seed-offset evaluation):

  - random coin-flip partition with and without the shared 1-opt local
    search (the negative control: how much of any pipeline's score is
    just the local search?)
  - spectral with and without its local search
  - Goemans-Williamson with and without the local search
  - sdp_bound: objective of the (approximately solved) SDP relaxation —
    an upper-bound PROXY for the true max cut; exact optima for n<=20
    come from brute force below
  - exact_opt: exact maximum cut by full enumeration for n <= 20
  - gnn_nols: the trained GNN's 50-sample decoding WITHOUT the local
    search (isolates the refinement's contribution)
  - gnn_ls: from the committed comparison run (samples + local search)

Output: results/analysis/maxcut_ablation.csv
Run:    python3 experiments/run_maxcut_ablation.py
Needs:  results/analysis/maxcut_comparison.csv and
        results/analysis/maxcut_gnn_weights.pt (the trained model)
"""

import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import cvxpy as cp
import networkx as nx
import torch

from src.graphs.generators import GRAPH_FAMILIES, generate_instance
from src.classical.maxcut.spectral import spectral_maxcut
from src.classical.maxcut.goemans_williamson import _cut_value
from src.gnn.models.gin import GINMaxCut
from src.gnn.training.maxcut_trainer import nx_to_pyg, local_search_refine as local_search_1opt

LOCAL_SEARCH_NOTE = ("1-opt local search: repeatedly move any node whose "
                     "move strictly increases the cut, until no move helps "
                     "(identical implementation for every method).")


def spectral_norefine(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    L = nx.laplacian_matrix(G, nodelist=nodes).astype(float).toarray()
    _, evecs = np.linalg.eigh(L)
    v_max = evecs[:, -1]
    S = {nodes[i] for i in range(n) if v_max[i] >= 0}
    if len(S) == 0:
        S.add(nodes[0])
    elif len(S) == n:
        S.discard(nodes[0])
    return S, _cut_value(G, S)


def gw_with_sdp_value(G, n_roundings=50, seed=0):
    """One SDP solve: (objective, best-rounded partition, its cut)."""
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    node_idx = {v: i for i, v in enumerate(nodes)}
    W = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        w = data.get("weight", 1.0)
        W[i, j] = W[j, i] = w
    L_w = np.diag(W.sum(axis=1)) - W

    X = cp.Variable((n, n), symmetric=True)
    prob = cp.Problem(cp.Maximize(0.25 * cp.trace(L_w @ X)),
                      [X >> 0, cp.diag(X) == 1])
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
    except cp.SolverError:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=10000, eps=1e-6)
    sdp_val = prob.value if prob.status in ("optimal", "optimal_inaccurate") else np.nan

    X_val = X.value
    X_val = (X_val + X_val.T) / 2
    eigs, vecs = np.linalg.eigh(X_val)
    eigs = np.maximum(eigs, 0)
    X_val = vecs @ np.diag(eigs) @ vecs.T
    V = vecs * np.sqrt(eigs + 1e-10)[np.newaxis, :]

    rng = np.random.RandomState(seed)
    best_S, best_cut = None, -1
    for _ in range(n_roundings):
        r = rng.randn(n)
        r = r / np.linalg.norm(r)
        signs = V @ r
        S = {nodes[i] for i in range(n) if signs[i] >= 0}
        if len(S) == 0:
            S.add(nodes[0])
        elif len(S) == n:
            S.discard(nodes[0])
        cut = _cut_value(G, S)
        if cut > best_cut:
            best_cut, best_S = cut, S
    return sdp_val, best_S, best_cut


def exact_maxcut_bruteforce(G):
    """Exact maximum cut by full enumeration (n <= 20 only). Vectorized."""
    n = G.number_of_nodes()
    if n > 20:
        return np.nan
    nodes = sorted(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v], d.get("weight", 1.0)) for u, v, d in G.edges(data=True)]
    w = np.array([e[2] for e in edges], dtype=float)
    us = np.array([e[0] for e in edges])
    vs = np.array([e[1] for e in edges])
    masks = np.arange(1 << n, dtype=np.uint32)
    bu = ((masks[:, None] >> us[None, :]) & 1).astype(bool)
    bv = ((masks[:, None] >> vs[None, :]) & 1).astype(bool)
    cuts = (bu != bv) @ w
    return float(cuts.max())


def main():
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = GINMaxCut(input_dim=5, hidden_dim=128, n_layers=5, logit_init_std=1.0)
    model.load_state_dict(torch.load('results/analysis/maxcut_gnn_weights.pt',
                                     map_location='cpu'))
    model.eval()

    df = pd.read_csv('results/analysis/maxcut_comparison.csv')

    rows = []
    for k, (_, r) in enumerate(df.iterrows()):
        seed_i = 42 + 2000 + int(r['instance']) * 1000
        G = generate_instance(r['family'], int(r['n']), seed=seed_i)
        m = G.number_of_edges()
        assert m == int(r['m']), (m, r['m'])

        # --- coin-flip controls (no GNN involved), per-instance seeded ---
        # 1-sample and 50-sample versions, so the control is budget-matched
        # to the GNN's 50-sample decoding (and to GW's 50 roundings).
        crng = np.random.RandomState(seed_i)
        nodes = sorted(G.nodes())
        S1 = {v for v in nodes if crng.random() < 0.5}
        if not S1:
            S1.add(nodes[0])
        rand_nols = _cut_value(G, S1)
        rand_ls = _cut_value(G, local_search_1opt(G, set(S1)))
        rand50_nols, best50 = -1.0, None
        for _ in range(50):
            S = {v for v in nodes if crng.random() < 0.5}
            if not S:
                S.add(nodes[0])
            elif len(S) == len(nodes):
                S.discard(nodes[0])
            c = _cut_value(G, S)
            if c > rand50_nols:
                rand50_nols, best50 = c, S
        rand50_ls = _cut_value(G, local_search_1opt(G, set(best50)))

        # --- spectral without its refinement (threshold only) ---
        spec_nr = spectral_norefine(G)[1]
        # (spectral_maxcut(refine=True) — the committed CSV's spectral_cut —
        #  already applies the same 1-opt loop; greedy_maxcut likewise starts
        # from a coin flip and runs the same loop.)

        # --- GW with the shared refinement ---
        sdp_val, gwS, gw_cut = gw_with_sdp_value(G, n_roundings=50, seed=seed_i)
        gw_ls = _cut_value(G, local_search_1opt(G, set(gwS)))

        # --- GNN without refinement (samples only) ---
        model_cpu = model.cpu(); model_cpu.eval()
        with torch.no_grad():
            p = model_cpu(nx_to_pyg(G, feature_seed=seed_i)).cpu().numpy()
        model.to(dev)
        nodes = sorted(G.nodes())
        grng = np.random.RandomState(seed_i)
        gnn_nols = -1.0
        for _ in range(50):
            S = {nodes[i] for i in range(len(nodes)) if grng.random() < p[i]}
            if len(S) == 0:
                S.add(nodes[0])
            elif len(S) == len(nodes):
                S.discard(nodes[0])
            gnn_nols = max(gnn_nols, _cut_value(G, S))

        rows.append({
            'family': r['family'], 'n': int(r['n']), 'instance': int(r['instance']),
            'm': m,
            'rand_nols': rand_nols, 'rand_ls': rand_ls,
            'rand50_nols': rand50_nols, 'rand50_ls': rand50_ls,
            'spectral_norefine': spec_nr,
            'gw_plus_ls': gw_ls, 'sdp_bound': sdp_val,
            'exact_opt': exact_maxcut_bruteforce(G),
            'gnn_nols': gnn_nols,
        })
        if (k + 1) % 50 == 0:
            print(f'{k+1}/600', flush=True)

    out = pd.DataFrame(rows)
    # committed-comparison columns carried over as aliases: gnn_ls (samples +
    # local search), and greedy/spectral/gw for convenience
    out = out.merge(df[['family', 'n', 'instance', 'gnn_cut', 'greedy_cut',
                        'spectral_cut', 'gw_cut']]
                    .rename(columns={'gnn_cut': 'gnn_ls', 'greedy_cut': 'greedy',
                                     'spectral_cut': 'spectral', 'gw_cut': 'gw'}),
                    on=['family', 'n', 'instance'])
    out.to_csv('results/analysis/maxcut_ablation.csv', index=False)
    print('saved results/analysis/maxcut_ablation.csv')

    m = out['m']
    print('\n=== mean cut / num_edges (unweighted) ===')
    for c in ['rand_nols', 'rand50_nols', 'rand_ls', 'rand50_ls', 'gnn_nols', 'gnn_ls', 'spectral_norefine',
              'greedy', 'spectral', 'gw', 'gw_plus_ls']:
        print(f'  {c:18s} {(out[c]/m).mean():.4f}')
    print(f'  {"sdp_bound":18s} {(out.sdp_bound/m).mean():.4f}  (upper-bound proxy)')
    nn = out[out.exact_opt.notna()]
    print(f'  {"exact_opt (n<=20)":18s} {(nn.exact_opt/nn.m).mean():.4f}  ({len(nn)} instances)')


if __name__ == '__main__':

    main()
