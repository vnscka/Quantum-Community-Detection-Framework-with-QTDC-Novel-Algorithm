"""
Quantum Thermal Diffusion Community Detection (QTDC)
=====================================================

NOVEL ALGORITHM — Original Contribution
-----------------------------------------

Motivation & Gap in Literature
--------------------------------
Existing quantum community detection divides into four families:
  1. QAOA / QUBO    — variational modularity optimization (Farhi 2014; D-Wave 2017–2024)
  2. Quantum Walk   — walk localization similarity (Childs 2002; Mukai 2020;
                      Fourier/Grover coin comparison, Phys. Rev. Research 2020)
  3. VQE Spectral   — Fiedler-vector variational search (Peruzzo 2014;
                      deteQt / QSVT approach, Umeano et al. arXiv:2412.13160, 2024)
  4. Quantum Anneal — D-Wave QUBO hierarchical annealing (Wierzbiński 2023–2024)

None of these exploit the intermediate-temperature block structure of the
quantum Gibbs thermal state ρ(β) = e^{-βH}/Z as a community detector.

QTDC fills this gap. It is distinct from every known approach:

  ┌──────────────────────┬────────────────────────────────────────────────────┐
  │ Prior work           │ How QTDC differs                                   │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Quantum annealing    │ Annealing drives to β→∞ ground state.              │
  │                      │ QTDC uses the finite β* intermediate-T block        │
  │                      │ structure, NOT the ground state.                   │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ deteQt / QSVT        │ Extracts only the leading eigenvector of B.        │
  │ (Umeano 2024)        │ QTDC sums ALL eigenmodes weighted by e^{-βλ},      │
  │                      │ capturing multi-mode community information.         │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Quantum Walk CTQW    │ Walk uses real-time evolution e^{-iLt} (unitary).  │
  │                      │ QTDC uses imaginary-time e^{-βH} (non-unitary,     │
  │                      │ thermal). Imaginary-time cannot be cast as a walk. │
  ├──────────────────────┼────────────────────────────────────────────────────┤
  │ Thermal TDA          │ Scali, Kyriienko (2024) extracts Betti numbers      │
  │ (Scali 2024)         │ from thermal states. QTDC extracts community        │
  │                      │ labels via QTM maximization — entirely different.  │
  └──────────────────────┴────────────────────────────────────────────────────┘

Core Idea
----------
Build the blended graph Hamiltonian:

    H = α · L  −  γ · B

where L = D − A (Laplacian, diffusion operator) and B = A − kk^T/2m
(Newman-Girvan modularity matrix). The minus on B means the ground state
of H aligns with high-modularity partitions.

The Gibbs thermal state at inverse temperature β is:

    ρ(β) = e^{−βH} / Tr(e^{−βH})

Its eigenexpansion is:

    ρ(β)_{ij} = Σ_k  [e^{−β λ_k} / Z(β)]  φ_k(i) φ_k(j)

where (λ_k, φ_k) are eigenpairs of H. The thermal weight e^{−β λ_k}/Z(β) acts
as a spectral filter:

  · β → 0  (hot):  all modes equally weighted → uniform mixture, no structure
  · β → ∞  (cold): only ground mode → effectively single eigenvector (like deteQt)
  · β = β* (optimal): low-frequency intra-community modes dominate; high-frequency
                       inter-community modes suppressed → ρ(β*) is block-diagonal

Novel Contribution (I): Quantum Thermal Modularity (QTM)
---------------------------------------------------------
We define QTM, a density-matrix generalization of Newman-Girvan modularity:

    QTM(β) = Tr(ρ(β) · B) / 2m

Classical modularity Q = (1/2m) Σ_{ij} B_{ij} δ(c_i,c_j) is recovered when
ρ → diagonal community indicator matrix. QTM is maximized at the β* where
the thermal coherences align most strongly with community structure.

Novel Contribution (II): Integrated Thermal Affinity (ITA)
-----------------------------------------------------------
Rather than using ρ(β*) alone, we build a QTM-weighted integral over the
full β-spectrum:

    T_{ij} = Σ_β  softmax[τ · QTM(β)]_β  · |ρ(β)_{ij}|

This accumulates evidence from all temperatures, not just β*, and is more
robust to noisy or irregular graphs. The softmax concentration parameter τ
controls how sharply the integral focuses around β*. In the τ→∞ limit the
ITA reduces to the single-β* thermal affinity.

Novel Contribution (III): The β* Phase Transition
--------------------------------------------------
The position of β* identifies a genuine phase-transition-like point in the
spectral filter: below β* the thermal state is too delocalized (hot phase,
all nodes mixed), above β* it collapses to the ground-state cut (cold phase).
β* is the community-detection phase boundary. This connects QTDC conceptually
to quantum phase transitions while remaining an operationally practical method.

Algorithm Steps
---------------
1. Build H = α·L − γ·B
2. Eigendecompose H: H = V Λ V^T
3. Scan β ∈ [β_min, β_max]; compute QTM(β) for each
4. β* = argmax QTM(β)
5. Build Integrated Thermal Affinity T = Σ_β softmax_weight(β) · |ρ(β)|
6. Spectral-cluster T (normalised cut) → k community labels

Hardware path
-------------
Classical simulation: exact eigendecomposition, O(n³).
On real quantum hardware: use quantum imaginary-time evolution (QITE,
Motta et al. 2020, Nature Physics) or variational thermal state preparation
(Andrews & Bhatt 2024) to prepare ρ(β*) in a quantum register.
The ITA can then be estimated via classical shadows (O(n log n) shots).

References
----------
Temme et al. (2011). Quantum Metropolis sampling. Nature 471, 87–90.
Motta et al. (2020). Determining eigenstates and thermal states on a quantum
  computer using QITE. Nature Physics 16, 205–210.
Scali, Umeano, Kyriienko (2024). The topology of data hides in quantum thermal
  states. APL Quantum 1, 036106.
Umeano, Scali, Kyriienko (2024). Quantum community detection via deterministic
  elimination. arXiv:2412.13160.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Optional, Dict, Any, List, Tuple


class QTDC:
    """
    Quantum Thermal Diffusion Community Detection (QTDC).

    Novel algorithm: finds communities via the block structure of
    ρ(β*) = e^{-β*H}/Z at the optimal inverse temperature β* that
    maximises Quantum Thermal Modularity QTM(β) = Tr(ρ(β)·B)/2m.

    Parameters
    ----------
    n_communities : int
        Number of communities to detect.
    alpha : float
        Weight of Laplacian in H = alpha*L - gamma*B.
        Higher α → more diffusive regularisation.
    gamma : float
        Weight of modularity matrix. Higher γ → modularity drives β*.
    beta_min, beta_max : float
        Range of inverse temperature scan.
    n_beta : int
        Grid resolution for β scan.
    integration_tau : float
        Softmax concentration for Integrated Thermal Affinity.
        Higher τ → integral focuses tightly around β*.
        τ=0 → uniform integration over all β.
    affinity_mode : str
        'abs' | 'real' | 'mixed' — how ρ off-diagonals are extracted.
    """

    def __init__(
        self,
        n_communities: int = 2,
        alpha: float = 1.0,
        gamma: float = 0.5,
        beta_min: float = 0.05,
        beta_max: float = 8.0,
        n_beta: int = 150,
        integration_tau: float = 10.0,
        affinity_mode: str = "abs",
    ):
        self.k = n_communities
        self.alpha = alpha
        self.gamma = gamma
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.n_beta = n_beta
        self.integration_tau = integration_tau
        self.affinity_mode = affinity_mode

    # ── Hamiltonian & matrix construction ────────────────────────────────────

    def _laplacian(self, adj: np.ndarray) -> np.ndarray:
        degree = np.sum(adj, axis=1)
        return np.diag(degree) - adj

    def _modularity_matrix(self, adj: np.ndarray) -> np.ndarray:
        degree = np.sum(adj, axis=1)
        m = np.sum(adj) / 2.0
        return adj - np.outer(degree, degree) / (2 * m + 1e-12)

    def _build_hamiltonian(self, adj: np.ndarray) -> np.ndarray:
        """H = alpha*L - gamma*B  (symmetric, real)."""
        L = self._laplacian(adj)
        B = self._modularity_matrix(adj)
        H = self.alpha * L - self.gamma * B
        return (H + H.T) / 2.0

    # ── Thermal state ─────────────────────────────────────────────────────────

    def _thermal_state(
        self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, beta: float
    ) -> np.ndarray:
        """
        ρ(β) = V · diag(softmax(-β·λ)) · V^T
        Numerically stable via shift by minimum eigenvalue.
        """
        shifted = eigenvalues - eigenvalues[0]
        weights = np.exp(-beta * shifted)
        weights /= weights.sum()
        return eigenvectors @ np.diag(weights) @ eigenvectors.T

    # ── Quantum Thermal Modularity ────────────────────────────────────────────

    def _qtm(self, rho: np.ndarray, B: np.ndarray, m: float) -> float:
        """QTM(β) = Tr(ρ(β)·B) / 2m"""
        return float(np.real(np.trace(rho @ B))) / (2 * m + 1e-12)

    # ── Thermal affinity extraction ───────────────────────────────────────────

    def _affinity_from_rho(self, rho: np.ndarray) -> np.ndarray:
        """Extract non-negative node-pair affinity from ρ off-diagonals."""
        rho_sym = (rho + rho.T) / 2.0
        if self.affinity_mode == "abs":
            S = np.abs(rho_sym)
        elif self.affinity_mode == "real":
            S = np.maximum(0.0, np.real(rho_sym))
        else:  # mixed
            S = 0.7 * np.abs(rho_sym) + 0.3 * np.maximum(0.0, np.real(rho_sym))
        np.fill_diagonal(S, 0.0)
        return S

    # ── Integrated Thermal Affinity (ITA) ────────────────────────────────────

    def _integrated_thermal_affinity(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        B: np.ndarray,
        m: float,
        betas: np.ndarray,
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Build ITA = Σ_β  w(β) · |ρ(β)|

        where w(β) = softmax(τ · QTM(β)) focuses the integral near β*.

        Also returns (beta_star, qtm_curve) for diagnostics.
        """
        n = eigenvalues.shape[0]
        qtm_values = np.zeros(len(betas))
        rho_cache = []

        for i, beta in enumerate(betas):
            rho = self._thermal_state(eigenvalues, eigenvectors, beta)
            rho_cache.append(rho)
            qtm_values[i] = self._qtm(rho, B, m)

        beta_star = float(betas[np.argmax(qtm_values)])

        # Softmax weights over QTM curve
        if self.integration_tau > 0:
            shifted_qtm = qtm_values - qtm_values.max()
            weights = np.exp(self.integration_tau * shifted_qtm)
            weights /= weights.sum()
        else:
            weights = np.ones(len(betas)) / len(betas)

        # Accumulate ITA
        T = np.zeros((n, n))
        for i, rho in enumerate(rho_cache):
            T += weights[i] * self._affinity_from_rho(rho)

        np.fill_diagonal(T, 0.0)
        # Normalise to [0, 1]
        max_val = T.max()
        if max_val > 1e-10:
            T /= max_val

        return T, beta_star, qtm_values.tolist()

    # ── Spectral clustering ───────────────────────────────────────────────────

    def _spectral_cluster(self, S: np.ndarray, k: int) -> np.ndarray:
        """Normalised spectral clustering on affinity S."""
        d = np.sum(S, axis=1) + 1e-10
        D_inv_sqrt = np.diag(1.0 / np.sqrt(d))
        L_norm = D_inv_sqrt @ S @ D_inv_sqrt

        eigenvalues, eigenvectors = eigh(L_norm)
        idx = np.argsort(eigenvalues)[::-1]
        V = eigenvectors[:, idx[:k]]

        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / (norms + 1e-10)
        return self._kmeans(V, k)

    def _kmeans(self, X: np.ndarray, k: int, max_iter: int = 300) -> np.ndarray:
        n = X.shape[0]
        rng = np.random.default_rng(42)
        centers = [X[rng.integers(n)].copy()]
        for _ in range(k - 1):
            dists = np.min([np.sum((X - c) ** 2, axis=1) for c in centers], axis=0)
            probs = dists / (dists.sum() + 1e-10)
            centers.append(X[rng.choice(n, p=probs)].copy())
        centers = np.array(centers)
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            dists = np.array([np.sum((X - c) ** 2, axis=1) for c in centers])
            new_labels = np.argmin(dists, axis=0)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            for j in range(k):
                if np.any(labels == j):
                    centers[j] = X[labels == j].mean(axis=0)
        return labels

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_modularity(self, adj: np.ndarray, labels: np.ndarray) -> float:
        degree = np.sum(adj, axis=1)
        m = np.sum(adj) / 2.0
        Q = 0.0
        n = len(labels)
        for i in range(n):
            for j in range(n):
                if labels[i] == labels[j]:
                    Q += adj[i, j] - degree[i] * degree[j] / (2 * m + 1e-10)
        return Q / (2 * m + 1e-10)

    def _thermal_spectral_gap(self, eigenvalues: np.ndarray, beta: float) -> float:
        """Thermal gap: difference in Boltzmann weights of modes 1 and 2."""
        shifted = eigenvalues - eigenvalues[0]
        weights = np.exp(-beta * shifted)
        weights /= weights.sum()
        return float(weights[1] - weights[2]) if len(weights) > 2 else 0.0

    # ── Main API ──────────────────────────────────────────────────────────────

    def fit(self, adj: np.ndarray, verbose: bool = False) -> "QTDC":
        """
        Run QTDC on adjacency matrix adj.

        Steps:
          1. Build blended Hamiltonian H = α·L − γ·B
          2. Eigendecompose H
          3. Scan β; compute QTM(β) for each; find β*
          4. Build Integrated Thermal Affinity (ITA) via QTM-softmax weights
          5. Spectral-cluster ITA → community labels
        """
        n = adj.shape[0]
        self.n_nodes_ = n

        if verbose:
            print(f"QTDC: {n} nodes, k={self.k}")
            print(f"  H = {self.alpha}·L − {self.gamma}·B")
            print(f"  β ∈ [{self.beta_min}, {self.beta_max}] × {self.n_beta} pts")

        # Step 1–2: Hamiltonian + eigendecomposition
        B = self._modularity_matrix(adj)
        m = np.sum(adj) / 2.0
        H = self._build_hamiltonian(adj)
        self.eigenvalues_, self.eigenvectors_ = eigh(H)
        self.B_ = B
        self.m_ = m

        if verbose:
            gap = self.eigenvalues_[1] - self.eigenvalues_[0]
            print(f"  Spectral gap λ₁−λ₀ = {gap:.4f}")

        # Steps 3–4: QTM scan + ITA
        betas = np.linspace(self.beta_min, self.beta_max, self.n_beta)
        (self.thermal_affinity_,
         self.beta_star_,
         self.qtm_curve_) = self._integrated_thermal_affinity(
            self.eigenvalues_, self.eigenvectors_, B, m, betas
        )
        self.betas_ = betas
        self.qtm_star_ = float(max(self.qtm_curve_))

        if verbose:
            print(f"  β* = {self.beta_star_:.3f},  QTM(β*) = {self.qtm_star_:.5f}")
            print(f"  Thermal gap at β* = "
                  f"{self._thermal_spectral_gap(self.eigenvalues_, self.beta_star_):.4f}")

        # Step 5: Spectral clustering
        self.labels_ = self._spectral_cluster(self.thermal_affinity_, self.k)
        self.modularity_ = self._compute_modularity(adj, self.labels_)
        self.thermal_gap_ = self._thermal_spectral_gap(
            self.eigenvalues_, self.beta_star_
        )

        if verbose:
            print(f"  Modularity Q = {self.modularity_:.4f}")
            print(f"  Labels: {self.labels_}")

        return self

    def get_circuit_stats(self) -> Dict[str, Any]:
        return {
            "algorithm": "QTDC (Novel)",
            "full_name": "Quantum Thermal Diffusion Community Detection",
            "key_novelty": "β* thermal state block structure + QTM maximisation + ITA",
            "hamiltonian": f"H = {self.alpha}·L − {self.gamma}·B",
            "beta_star": round(float(getattr(self, "beta_star_", 0)), 4),
            "qtm_star": round(float(getattr(self, "qtm_star_", 0)), 5),
            "thermal_gap": round(float(getattr(self, "thermal_gap_", 0)), 4),
            "integration_tau": self.integration_tau,
            "complexity": "O(n³) eigendecomp + O(n² × n_beta) ITA accumulation",
            "hardware_path": "QITE (Motta 2020) + classical shadows readout",
        }
