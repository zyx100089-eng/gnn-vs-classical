"""The single 1-opt local-search refinement used across this project.

Every method that is allowed post-processing — the GNN pipeline, the
spectral baseline, and the ablation's controls — calls this function, so
"identical implementation for every method" is enforced in one place
rather than copy-pasted.
"""


def local_search_refine(G, S: set) -> set:
    """Move any node whose move strictly increases the cut, until none does."""
    nodes = sorted(G.nodes())
    improved = True
    while improved:
        improved = False
        for v in nodes:
            in_S = v in S
            gain = 0.0
            for u in G.neighbors(v):
                w = G[v][u].get("weight", 1.0)
                if (u in S) == in_S:
                    gain += w
                else:
                    gain -= w
            if gain > 1e-10:
                if in_S:
                    S.remove(v)
                else:
                    S.add(v)
                improved = True
    return S
