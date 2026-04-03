import numpy as np
from dataclasses import dataclass

"""
Field Computation

Implements the core equations from §3.1 of the paper:

- Orientation — unsigned saccade orientation ω_j ∈ [0, π).
- Point-Saccade Distance — §3.1.1, Eq. (1): minimal Euclidean distance from a
  reference point p_i to a saccade segment s_j.
- Gaussian Kernel — §3.1.2, Eq. (2): kernel K(p_i, s_j) and kernel mass m_i.
- Mean Orientation and Coherence — §3.1.3, Eq. (3): weighted mean orientation ω̄_i
  and mean resultant length R_i via double-angle embedding.
- Density — §3.1.4, Eq. (4): normalized kernel mass ρ_i.

Public API : make_reference_grid, make_orientation_field, OrientationField
Internal   : _compute_orientations, _point_to_segment_distances, _gaussian_kernel
"""


@dataclass
class OrientationField:
    """
    Precomputed orientation field over a reference grid — §3.1.

    Holds the three per-point statistics returned by make_orientation_field
    together with the grid shape so that callers need not carry it separately.

    Attributes
    ----------
    omega_mean : (P,) array in [0, π), NaN where m_i = 0
    R          : (P,) array in [0, 1] — mean resultant length (coherence)
    rho        : (P,) array in [0, 1] — normalized kernel mass (density), sums to 1
    grid_shape : (ny, nx) tuple — reshape arrays with .reshape(grid_shape)
    """
    omega_mean: np.ndarray
    R:          np.ndarray
    rho:        np.ndarray
    grid_shape: tuple


def _compute_orientations(onset, offset):
    """
    Unsigned saccade orientations ω_j ∈ [0, π) — §3.1.

    ω_j = atan2(y_{j1} − y_{j0}; x_{j1} − x_{j0})  mod  π

    Orientations are unsigned: ω and ω+π are treated as equivalent, so that
    back-and-forth saccades along the same axis reinforce rather than cancel.

    Used internally by make_orientation_field; not part of the public API.
    """
    d = offset - onset
    return np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)


def _point_to_segment_distances(points, onset, offset):
    """
    Minimal Euclidean distance d(p_i, s_j) from each reference point p_i to each
    saccade segment s_j = [s_{j0}, s_{j1}] — Eq. (1).

    Used internally by make_orientation_field; not part of the public API.
    """
    p  = points[:, None, :]   # (P, 1, 2)
    a  = onset[None,  :, :]   # (1, J, 2)
    b  = offset[None, :, :]   # (1, J, 2)

    ab      = b - a                                                       # (1, J, 2)
    ap      = p - a                                                       # (P, J, 2)
    ab_len2 = np.sum(ab * ab, axis=-1, keepdims=True)                     # (1, J, 1)
    ab_len2 = np.where(ab_len2 == 0.0, 1.0, ab_len2)                     # guard zero-length

    t = np.clip(np.sum(ap * ab, axis=-1, keepdims=True) / ab_len2, 0.0, 1.0)  # (P, J, 1)
    closest = a + t * ab                                                  # (P, J, 2)
    return np.linalg.norm(p - closest, axis=-1)                           # (P, J)


def _gaussian_kernel(d, sigma):
    """
    Gaussian kernel K(p_i, s_j) = exp(−½ (d / σ)²)  — Eq. (2).

    Used internally by make_orientation_field; not part of the public API.
    """
    return np.exp(-0.5 * (d / sigma) ** 2)



def make_reference_grid(w, h, grid_resolution=200):
    """
    Uniform reference grid of positions {p_i} ⊂ [1..W] × [1..H] — §3.1.

    Parameters
    ----------
    w, h             : stimulus width and height in pixels
    grid_resolution  : int or (ny, nx) tuple
                       Number of grid points along each axis. A single integer
                       produces a square grid; a tuple sets rows and columns
                       independently. Follows numpy's (rows, cols) = (ny, nx)
                       convention, matching the shape of the returned YY array.
                       Default: 200 (i.e. 200×200 = 40 000 points).

    Returns
    -------
    pts : (ny*nx, 2) array — flattened (x, y) positions, input to make_orientation_field
    XX  : (ny, nx) meshgrid of x-coordinates (use for reshaping field outputs)
    YY  : (ny, nx) meshgrid of y-coordinates (use for reshaping field outputs)
    """
    if isinstance(grid_resolution, int):
        ny = nx = grid_resolution
    else:
        ny, nx = grid_resolution
    xs = np.linspace(1, w, nx, dtype=np.float64)
    ys = np.linspace(1, h, ny, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return pts, xx, yy


def make_orientation_field(points, onset, offset, sigma=50.0,
                           grid_shape=None,
                           points_per_chunk=4096, segs_per_chunk=512):
    """
    Compute the orientation field (ω̄, R, ρ) over all reference points — Eqs. (2)–(4).

    For each reference point p_i, accumulates:
      sin_i = Σ_j K(p_i,s_j) · sin(2ω_j)     (unnormalized; proportional to paper's s̄_i · m_i)
      cos_i = Σ_j K(p_i,s_j) · cos(2ω_j)     (unnormalized; proportional to paper's c̄_i · m_i)
      m_i   = Σ_j K(p_i,s_j)                  (kernel mass)

    Then derives:
      ω̄_i = ½ · (atan2(sin_i, cos_i) mod 2π) ∈ [0, π)   mean orientation, Eq. (3)
      R_i  = √(sin_i² + cos_i²) / m_i          ∈ [0, 1]  coherence (MRL), Eq. (3)
      ρ_i  = m_i / Σ_k m_k                     ∈ [0, 1]  density, Eq. (4)

    Points with no kernel support (m_i = 0) get ω̄_i = NaN, R_i = 0, ρ_i = 0.

    Parameters
    ----------
    points           : (P, 2)  reference grid positions {p_i}
    onset            : (J, 2)  saccade starting points  {s_{j0}}
    offset           : (J, 2)  saccade end points       {s_{j1}}
    sigma            : float   Gaussian bandwidth σ > 0, in pixels — Eq. (2)
    grid_shape       : (ny, nx) tuple — stored on the returned OrientationField for
                       convenient reshaping; pass yy.shape from make_reference_grid
    points_per_chunk : int     chunk size over grid points (memory/speed trade-off)
    segs_per_chunk   : int     chunk size over saccades   (memory/speed trade-off)

    Returns
    -------
    OrientationField with omega_mean, R, rho (each (P,)) and grid_shape
    """
    n_pts = points.shape[0]
    n_sac = onset.shape[0]

    omega = _compute_orientations(onset, offset)

    sin2 = np.sin(2.0 * omega)  # (J,)
    cos2 = np.cos(2.0 * omega)  # (J,)

    sin_acc  = np.zeros(n_pts, dtype=np.float64)
    cos_acc  = np.zeros(n_pts, dtype=np.float64)
    mass_acc = np.zeros(n_pts, dtype=np.float64)

    for p0 in range(0, n_pts, points_per_chunk):
        p1    = min(n_pts, p0 + points_per_chunk)
        p_blk = points[p0:p1]

        sin_blk  = np.zeros(p1 - p0, dtype=np.float64)
        cos_blk  = np.zeros(p1 - p0, dtype=np.float64)
        mass_blk = np.zeros(p1 - p0, dtype=np.float64)

        for s0 in range(0, n_sac, segs_per_chunk):
            s1    = min(n_sac, s0 + segs_per_chunk)
            d_blk = _point_to_segment_distances(p_blk, onset[s0:s1], offset[s0:s1])
            k_blk = _gaussian_kernel(d_blk, sigma)

            mass_blk += k_blk.sum(axis=1)
            sin_blk  += k_blk @ sin2[s0:s1]
            cos_blk  += k_blk @ cos2[s0:s1]

        sin_acc[p0:p1]  = sin_blk
        cos_acc[p0:p1]  = cos_blk
        mass_acc[p0:p1] = mass_blk

    # ω̄_i = ½ · (atan2(sin_i, cos_i) mod 2π)  — scale-invariant, so unnormalized sums work
    omega_mean = 0.5 * np.mod(np.arctan2(sin_acc, cos_acc), 2.0 * np.pi)
    omega_mean = np.where(mass_acc > 0.0, omega_mean, np.nan)

    # R_i = √(s̄_i² + c̄_i²)  where s̄_i = sin_i/m_i, c̄_i = cos_i/m_i
    R = np.where(mass_acc > 0.0, np.sqrt(sin_acc**2 + cos_acc**2) / mass_acc, 0.0)

    # ρ_i = m_i / Σ_k m_k
    total_mass = mass_acc.sum()
    rho = mass_acc / total_mass if total_mass > 0 else np.zeros_like(mass_acc)

    return OrientationField(omega_mean=omega_mean, R=R, rho=rho, grid_shape=grid_shape)
