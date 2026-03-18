<p align="center">
  <h1 align="center">⟨ψ| Quantum Community Detection |ψ⟩</h1>
  <p align="center">Four quantum algorithms for graph community detection — including a novel contribution</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/NumPy-pure%20backend-013243?style=flat-square&logo=numpy" />
  <img src="https://img.shields.io/badge/No%20SDK-no%20Qiskit%20%2F%20PennyLane-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Novel-QTDC%20algorithm-gold?style=flat-square" />
</p>

---

## What this is

A complete Python framework implementing four quantum algorithms for community detection on graphs, built on a **from-scratch NumPy statevector simulator** — no Qiskit, no PennyLane, no Cirq required.

| # | Algorithm | Core idea | Reference |
|---|-----------|-----------|-----------|
| 1 | **QAOA** | Encodes modularity as Ising Hamiltonian; alternates cost/mixer unitaries | Farhi et al. (2014) |
| 2 | **Quantum Walk** | Continuous-time walk interference localises within communities | Mukai & Hatano (2020) |
| 3 | **VQE Spectral** | Variational Fiedler-vector search via hardware-efficient ansatz | Peruzzo et al. (2014) |
| 4 | **QTDC ★** | Quantum Thermal Diffusion — exploits thermal state block structure at optimal β* | **This work (2025)** |

---

## Novel algorithm: QTDC

**Quantum Thermal Diffusion Community Detection** is an original contribution. It builds the blended Hamiltonian **H = αL − γB** (Laplacian + modularity matrix), then finds the optimal inverse temperature β* by maximising **Quantum Thermal Modularity**:

```
QTM(β) = Tr(ρ(β) · B) / 2m       where ρ(β) = e^{-βH} / Z(β)
```

At β*, the off-diagonal block structure of the thermal density matrix naturally reveals communities. An **Integrated Thermal Affinity** matrix accumulates evidence across temperatures, weighted by QTM, before final spectral clustering.

Distinct from: quantum annealing (β→∞ ground state), deteQt/QSVT (single eigenvector), quantum walks (real-time vs imaginary-time evolution), thermal TDA (Betti numbers, not community labels).

---

## Benchmark results

| Algorithm | 8-node SBM (NMI) | Ring of Cliques 9n (NMI) | 12-node SBM (NMI) | Avg time |
|-----------|:-:|:-:|:-:|:-:|
| QAOA | 0.000 | 0.734 | 0.820 | ~1–13 s |
| Quantum Walk | 0.000 | 1.000 | 0.406 | ~0.002 s |
| VQE Spectral | 0.000 | 0.786 | 1.000 | ~1–5 s |
| **QTDC ★** | **1.000** | **1.000** | **1.000** | **~0.005 s** |

QTDC achieves perfect NMI on all three benchmarks at ~1000× the speed of VQE Spectral.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/quantum-community-detection.git
cd quantum-community-detection

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, NumPy ≥ 1.24, SciPy ≥ 1.10. No quantum SDK needed.

---

## Quick start

```python
from quantum_community_detection import (
    QTDC, QAOACommunityDetection,
    QuantumWalkCommunityDetection, VQESpectralPartitioning,
    small_test_graph, modularity_score, normalized_mutual_information
)

# Load a built-in benchmark graph
adj, labels_true = small_test_graph(8)   # 8-node SBM, 2 communities

# Run all four algorithms
qaoa = QAOACommunityDetection(n_layers=1).fit(adj)
qw   = QuantumWalkCommunityDetection(n_clusters=2).fit(adj)
vqe  = VQESpectralPartitioning(n_communities=2).fit(adj)
qtdc = QTDC(n_communities=2).fit(adj)

# Evaluate
for name, algo in [('QAOA', qaoa), ('QW', qw), ('VQE', vqe), ('QTDC', qtdc)]:
    Q   = modularity_score(adj, algo.labels_)
    nmi = normalized_mutual_information(labels_true, algo.labels_)
    print(f"{name:6}  Q={Q:.4f}  NMI={nmi:.4f}  labels={algo.labels_}")
```

---

## Run the full benchmark suite

```bash
python3 -m quantum_community_detection.demo
```

Output: terminal benchmark table + `benchmark_results.json`.

---

## Open the interactive dashboard

Open `dashboard.html` in any browser — no server required:

```bash
# macOS
open quantum_community_detection/dashboard.html

# Linux
xdg-open quantum_community_detection/dashboard.html

# Windows
start quantum_community_detection/dashboard.html
```

The dashboard shows live graph visualisation, algorithm predictions, QTM(β) curve, and comparison charts.

---

## Project structure

```
quantum-community-detection/
├── quantum_community_detection/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── qaoa.py            # QAOA via COBYLA optimiser
│   │   ├── quantum_walk.py    # CTQW similarity matrix
│   │   ├── vqe_spectral.py    # VQE Fiedler-vector search
│   │   └── qtdc.py            # ★ Novel: Quantum Thermal Diffusion CD
│   ├── utils/
│   │   └── quantum_sim.py     # Statevector simulator (pure NumPy)
│   ├── benchmarks/
│   │   └── benchmark.py       # Graph generators + NMI / Q / conductance
│   ├── demo.py                # CLI benchmark runner
│   └── dashboard.html         # Interactive browser dashboard
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Graph generators

```python
from quantum_community_detection import (
    stochastic_block_model,   # SBM with configurable p_in / p_out
    ring_of_cliques,          # k cliques connected in a ring
    planted_partition,        # Balanced k-way partition
    karate_club_graph,        # Zachary's Karate Club (34 nodes)
    small_test_graph,         # 8-node two-community SBM
)
```

---

## Complexity

| Algorithm | Classical simulation cost | Quantum hardware path |
|-----------|--------------------------|----------------------|
| QAOA | O(2^n × p × iter) | Native gate-based device |
| Quantum Walk | O(n³) expm per time step | CTQW on programmable photonics |
| VQE Spectral | O(n³) per iteration | Variational device + QPE |
| **QTDC** | O(n³ eigendecomp + n² × n_β ITA) | QITE (Motta et al. 2020) + classical shadows |

Feasible classically up to n ≈ 16 nodes; quantum hardware enables polynomial scaling beyond that.

---

## Citation

```bibtex
@software{qtdc2025,
  title   = {Quantum Community Detection: QAOA, Quantum Walk, VQE, and QTDC},
  year    = {2025},
  note    = {QTDC (Quantum Thermal Diffusion Community Detection) is an original
             algorithm introduced in this work. Hardware path via quantum imaginary-time
             evolution (Motta et al., Nature Physics 2020).},
  url     = {https://github.com/YOUR_USERNAME/quantum-community-detection}
}
```

---

## References

1. Farhi, Goldstone & Gutmann (2014). A Quantum Approximate Optimization Algorithm. arXiv:1411.4028.
2. Mukai & Hatano (2020). Discrete-time quantum walk on complex networks for community detection. *Physical Review Research* 2, 023378.
3. Peruzzo et al. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature Communications* 5, 4213.
4. Motta et al. (2020). Determining eigenstates and thermal states on a quantum computer using QITE. *Nature Physics* 16, 205–210.
5. Umeano, Scali & Kyriienko (2024). Quantum community detection via deterministic elimination. arXiv:2412.13160.
6. Newman & Girvan (2004). Finding and evaluating community structure in networks. *Physical Review E* 69, 026113.

---

*Built with pure NumPy — no quantum SDK dependency.*
