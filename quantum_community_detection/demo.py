"""
Quantum Community Detection - Main Demo
========================================
Run all 4 algorithms on multiple benchmark graphs and print results.
"""

import sys
import json
import time
import numpy as np

sys.path.insert(0, '.')
from quantum_community_detection import (
    QAOACommunityDetection, QuantumWalkCommunityDetection,
    VQESpectralPartitioning, QTDC,
    run_benchmark, small_test_graph, ring_of_cliques,
    planted_partition, stochastic_block_model,
    modularity_score, normalized_mutual_information, conductance
)

COLORS = {
    'header': '\033[95m',
    'blue':   '\033[94m',
    'green':  '\033[92m',
    'yellow': '\033[93m',
    'red':    '\033[91m',
    'bold':   '\033[1m',
    'end':    '\033[0m',
}

def c(text, color): 
    return f"{COLORS[color]}{text}{COLORS['end']}"

def print_banner():
    print()
    print(c("╔══════════════════════════════════════════════════════════════╗", 'blue'))
    print(c("║   QUANTUM COMMUNITY DETECTION — Algorithm Benchmark Suite   ║", 'blue'))
    print(c("║                                                              ║", 'blue'))
    print(c("║  Algorithms:                                                 ║", 'blue'))
    print(c("║   [1] QAOA       — Modularity optimization (Farhi 2014)     ║", 'blue'))
    print(c("║   [2] QW         — Quantum walk similarity (Childs 2002)    ║", 'blue'))
    print(c("║   [3] VQE        — Spectral bisection (Peruzzo 2014)        ║", 'blue'))
    print(c("║   [4] QTDC ★     — Thermal Diffusion (Novel)                ║", 'yellow'))
    print(c("╚══════════════════════════════════════════════════════════════╝", 'blue'))
    print()

def print_table(results, labels_true, adj):
    algo_names = list(results.keys())
    print(c(f"\n  {'Algorithm':<22} {'Modularity':>11} {'NMI':>8} {'Accuracy':>10} {'Conductance':>13} {'Time(s)':>9}", 'bold'))
    print("  " + "─" * 78)
    for name, r in results.items():
        sym = " ★" if "QTDC" in name else "  "
        color = 'yellow' if "QTDC" in name else 'end'
        mod = r['modularity']
        nmi = r['nmi']
        acc = r['accuracy']
        cond = r['conductance']
        t = r['time_sec']
        line = f"  {name+sym:<22} {mod:>11.4f} {nmi:>8.4f} {acc:>10.4f} {cond:>13.4f} {t:>9.3f}"
        print(c(line, color) if "QTDC" in name else line)

def demo_graph(title, adj, labels_true, n_communities, verbose=False):
    n = adj.shape[0]
    m = int(np.sum(adj > 0) // 2)
    print(c(f"\n{'═'*65}", 'blue'))
    print(c(f"  {title}", 'bold'))
    print(c(f"  {n} nodes  |  {m} edges  |  {n_communities} communities  |  Hilbert dim = {2**n}", 'blue'))
    print(c(f"{'═'*65}", 'blue'))
    print(f"  Ground truth: {labels_true}")

    algorithms = {
        'QAOA':          QAOACommunityDetection(n_layers=1),
        'Quantum Walk':  QuantumWalkCommunityDetection(n_clusters=n_communities),
        'VQE Spectral':  VQESpectralPartitioning(n_communities=n_communities),
        'QTDC (Novel)':  QTDC(n_communities=n_communities),
    }

    results = run_benchmark(algorithms, adj, labels_true, verbose=verbose)
    print_table(results, labels_true, adj)

    for name, r in results.items():
        print(f"  {name:<22} labels: {r['labels']}")

    return results


def main():
    print_banner()
    np.random.seed(42)

    all_results = {}

    # ── Benchmark 1: 8-node SBM ──────────────────────────────────────────────
    adj1, lt1 = small_test_graph(8)
    r1 = demo_graph("Benchmark 1: 8-node Stochastic Block Model", adj1, lt1, 2)
    all_results['8-node SBM'] = r1

    # ── Benchmark 2: Ring of Cliques (9 nodes, 3 communities) ───────────────
    adj2, lt2 = ring_of_cliques(clique_size=3, n_cliques=3)
    r2 = demo_graph("Benchmark 2: Ring of Cliques (9 nodes)", adj2, lt2, 3)
    all_results['Ring of Cliques'] = r2

    # ── Benchmark 3: Planted Partition (12 nodes) ───────────────────────────
    adj3, lt3 = planted_partition(12, k=3, p_in=0.75, p_out=0.08, seed=42)
    r3 = demo_graph("Benchmark 3: Planted Partition (12 nodes, 3 comm)", adj3, lt3, 3)
    all_results['Planted Partition'] = r3

    # ── Aggregate performance table ─────────────────────────────────────────
    print()
    print(c("\n╔══════════════════════════════════════════════════════════════╗", 'green'))
    print(c("║              AGGREGATE PERFORMANCE SUMMARY                  ║", 'green'))
    print(c("╚══════════════════════════════════════════════════════════════╝", 'green'))
    algo_names = ['QAOA', 'Quantum Walk', 'VQE Spectral', 'QTDC (Novel)']
    print(c(f"\n  {'Algorithm':<22} {'Avg NMI':>10} {'Avg Modularity':>16} {'Avg Accuracy':>14}", 'bold'))
    print("  " + "─" * 66)

    for aname in algo_names:
        nmis = []
        mods = []
        accs = []
        for gname, gres in all_results.items():
            for rname, r in gres.items():
                if aname in rname:
                    nmis.append(r['nmi'])
                    mods.append(r['modularity'])
                    accs.append(r['accuracy'])
        if nmis:
            sym = " ★" if "QTDC" in aname else "  "
            line = f"  {aname+sym:<22} {np.mean(nmis):>10.4f} {np.mean(mods):>16.4f} {np.mean(accs):>14.4f}"
            print(c(line, 'yellow') if "QTDC" in aname else line)

    print()
    print(c("\n  ★ = Novel algorithm (QTDC — Thermal Diffusion Community Detection)", 'yellow'))
    print(c("  QTDC key idea: thermal diffusion and entanglement entropy across bipartitions", 'yellow'))
    print(c("  detect community boundaries via quantum mutual information of the graph state |ψ(G)⟩.", 'yellow'))
    print()

    # Save JSON results
    out = {}
    for gname, gres in all_results.items():
        out[gname] = {aname: {'modularity': r['modularity'], 'nmi': r['nmi'],
                               'accuracy': r['accuracy'], 'time': r['time_sec']}
                      for aname, r in gres.items()}
    with open('benchmark_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(c("  Results saved to benchmark_results.json", 'green'))
    print()


if __name__ == '__main__':
    main()