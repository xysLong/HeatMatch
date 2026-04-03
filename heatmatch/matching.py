import numpy as np
from dataclasses import dataclass

"""
HeatMatch Similarity

Implements the similarity scores from §3.3 of the paper:

- Locational similarity S_loc — §3.3.1, Eq. (5): Pearson correlation of density
  maps ρ, rescaled to [0, 1].
- Directional similarity S_dir — §3.3.2, Eq. (6): density- and coherence-weighted
  sum of per-point axial angular similarities, using geometric-mean overlap weights,
  with a density-coherence trade-off parameter ∈ [0, 1] (§ Limitations).

Public API : compute_similarity, SimilarityResult
Internal   : _locational_similarity, _directional_similarity
"""


@dataclass
class SimilarityResult:
    """
    HeatMatch similarity scores between two saccade patterns — §3.3.

    Attributes
    ----------
    s_loc : float in [0, 1] — locational similarity S_loc, Eq. (5)
    s_dir : float in [0, 1] — directional similarity S_dir, Eq. (6)
    """
    s_loc: float
    s_dir: float


def _locational_similarity(rho_a, rho_b):
    """
    S_loc(A, B) = (corr(ρ^A, ρ^B) + 1) / 2  ∈ [0, 1]  — Eq. (5).

    Pearson correlation of the two density maps, shifted to [0, 1].
    S_loc = 1: identical spatial layout; S_loc = 0: maximally anti-correlated.
    """
    corr = np.corrcoef(rho_a.ravel(), rho_b.ravel())[0, 1]
    return float((corr + 1.0) / 2.0)


def _directional_similarity(omega_a, omega_b, R_a, R_b, rho_a, rho_b,
                             density_coherence_tradeoff=0.5):
    """
    S_dir(A, B) = Σ_i w_i · axial-sim(ω̄_i^A, ω̄_i^B) / Σ_i w_i  ∈ [0, 1]  — Eq. (6).

    where w_i = ρ̄_i^dct · R̄_i^(1−dct),  dct = density_coherence_tradeoff.

    ρ̄_i = √(ρ_i^A · ρ_i^B) and R̄_i = √(R_i^A · R_i^B) are the geometric-mean
    overlap weights for density and coherence.

    density_coherence_tradeoff=1 weights by density only — at this extreme S_dir
    becomes statistically redundant with S_loc since angular coherence has no
    influence on the score. Recommended value ~0.5.

    axial-sim(ω^A, ω^B) = 1 − (2/π) · δ  where δ = min(Δ, π−Δ), Δ = |ω^A − ω^B|.

    Grid points with no kernel support have ω̄ = NaN and ρ = 0; their contribution
    is zero via the ρ̄ weight. axial-sim is set to 0 for NaN inputs as a safety guard.
    """
    dct = density_coherence_tradeoff

    Delta     = np.abs(omega_a - omega_b)
    delta     = np.minimum(Delta, np.pi - Delta)
    axial_sim = 1.0 - (2.0 / np.pi) * delta
    axial_sim = np.where(np.isnan(axial_sim), 0.0, axial_sim)

    rho_bar = np.sqrt(rho_a * rho_b)
    r_bar   = np.sqrt(R_a   * R_b)
    w = rho_bar ** dct * r_bar ** (1.0 - dct)
    W = float(w.sum())
    return float(np.sum(w * axial_sim) / W) if W > 0.0 else 0.0


def compute_similarity(field_a, field_b, density_coherence_tradeoff=0.5):
    """
    Compute HeatMatch similarity between two saccade patterns — §3.3.

    Parameters
    ----------
    field_a, field_b : OrientationField
        Precomputed orientation fields from make_orientation_field.
    density_coherence_tradeoff : float in [0, 1]
        Controls how much S_dir weights grid points by density vs. coherence (§ Limitations).
        At 1.0 the weight is purely density-based and angular coherence has no influence,
        making S_dir statistically redundant with S_loc. Recommended ~0.5.

    Returns
    -------
    SimilarityResult with s_loc (Eq. 5) and s_dir (Eq. 6)
    """
    s_loc = _locational_similarity(field_a.rho, field_b.rho)
    s_dir = _directional_similarity(field_a.omega_mean, field_b.omega_mean,
                                    field_a.R, field_b.R,
                                    field_a.rho, field_b.rho,
                                    density_coherence_tradeoff=density_coherence_tradeoff)
    return SimilarityResult(s_loc=s_loc, s_dir=s_dir)