import numpy as np
from dataclasses import dataclass

"""
## HeatMatch Similarity

Implements the similarity scores from §3.3 of the paper:

- **Locational similarity** S_loc — §3.3.1, Eq. (5): Pearson correlation of density
  maps ρ, rescaled to [0, 1].
- **Directional similarity** S_dir — §3.3.2, Eq. (6): density- and coherence-weighted
  sum of per-point axial angular similarities, using geometric-mean overlap weights,
  with a trade-off parameter t ∈ [0, 1] (§ Limitations).

Public API : similarity, SimilarityResult
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


def _directional_similarity(omega_a, omega_b, R_a, R_b, rho_a, rho_b, t=0.5):
    """
    S_dir(A, B) = Σ_i ρ̄_i^t · R̄_i^(1−t) · axial-sim(ω̄_i^A, ω̄_i^B)  ∈ [0, 1]  — Eq. (6).

    ρ̄_i = √(ρ_i^A · ρ_i^B) and R̄_i = √(R_i^A · R_i^B) are the geometric-mean
    overlap weights for density and coherence. The parameter t ∈ [0, 1] trades off
    between them (§ Limitations): t=1 weights by density only, t=0 by coherence only,
    t=0.5 by their geometric mean.

    axial-sim(ω^A, ω^B) = 1 − (2/π) · δ  where δ = min(Δ, π−Δ), Δ = |ω^A − ω^B|.

    Grid points with no kernel support have ω̄ = NaN and ρ = 0; their contribution
    is zero via the ρ̄ weight. axial-sim is set to 0 for NaN inputs as a safety guard.
    """
    Delta     = np.abs(omega_a - omega_b)
    delta     = np.minimum(Delta, np.pi - Delta)
    axial_sim = 1.0 - (2.0 / np.pi) * delta
    axial_sim = np.where(np.isnan(axial_sim), 0.0, axial_sim)

    rho_bar = np.sqrt(rho_a * rho_b)
    r_bar   = np.sqrt(R_a   * R_b)
    return float(np.sum(rho_bar ** t * r_bar ** (1.0 - t) * axial_sim))


def compute_similarity(field_a, field_b, t=0.5):
    """
    Compute HeatMatch similarity between two saccade patterns — §3.3.

    Parameters
    ----------
    field_a, field_b : OrientationField
        Precomputed orientation fields from make_orientation_field.
    t : float in [0, 1]
        Density–coherence trade-off for S_dir (§ Limitations):
        t=1 → density only, t=0 → coherence only, t=0.5 → geometric mean.

    Returns
    -------
    SimilarityResult with s_loc (Eq. 5) and s_dir (Eq. 6)
    """
    s_loc = _locational_similarity(field_a.rho, field_b.rho)
    s_dir = _directional_similarity(field_a.omega_mean, field_b.omega_mean,
                                    field_a.R, field_b.R,
                                    field_a.rho, field_b.rho, t=t)
    return SimilarityResult(s_loc=s_loc, s_dir=s_dir)
