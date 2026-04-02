import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
import math

"""
## Visualization

Implements the OOI-coded heatmap pipeline from §3.2 of the paper.

Public API : Heatmap
Internal   : _make_cyclic_colormap, _opacity_from_confidence
"""


def _make_cyclic_colormap(base_cmap):
    """
    Cyclic diverging colormap for unsigned orientation display — §3.2.

    Concatenates the forward and reverse passes of a diverging colormap so that
    ω = 0 and ω = π map to the same color, creating a seamless cycle over [0, π).
    """
    base = plt.get_cmap(base_cmap)
    n = 256
    forward  = base(np.linspace(0, 1, n))
    backward = base(np.linspace(1, 0, n))
    colors = np.vstack([forward[:-1], backward[:-1]])   # avoid seam duplicate
    return ListedColormap(colors, name=f"{base_cmap}_cyclic")


def _opacity_from_confidence(confidence, degree=2.0, quantile=0.8, eps=1e-12):
    """
    "Gentle nonlinearity" (§3.2) mapping a confidence signal to opacity ∈ [0, 1].

    Uses a logistic function centered at the quantile-th percentile so that
    low-confidence grid points fade toward transparency.
    """
    cmin, cmax = float(np.min(confidence)), float(np.max(confidence))
    if cmax <= cmin + eps:
        return np.zeros_like(confidence)
    c = (confidence - cmin) / (cmax - cmin)
    center = np.quantile(c, quantile)
    c = np.clip(c, eps, 1.0 - eps)
    return 1.0 / (1.0 + (center * (1.0 - c) / ((1.0 - center) * c)) ** degree)


class Heatmap:
    """
    OOI-coded heatmap renderer with two-level caching — §3.2.

    Wraps an OrientationField and caches intermediate results to avoid redundant
    computation when only display parameters change:

    - Cyclic colormap  → rebuilt only when `cmap` changes.
    - Opacity array    → rebuilt only when `density_weight`, `degree`, or `quantile` changes.
    - OOI rotation     → always cheap (one np.mod call), never cached.

    Parameters
    ----------
    field : OrientationField
        Precomputed orientation field from make_orientation_field.
    w, h  : stimulus width and height in pixels
    """

    def __init__(self, field, w, h):
        self._field = field
        self._w     = w
        self._h     = h

        self._cmap_name    = None
        self._cyclic_cmap  = None

        self._opacity_key  = None   # (density_weight, degree, quantile)
        self._opacity      = None

    def draw(self, ax, ooi=0.0, density_weight=1.0, cmap='coolwarm', degree=2.0, quantile=0.8):
        """
        Render the OOI-coded saccade heatmap onto a matplotlib Axes — §3.2.

        Hue encodes mean orientation ω̄_i relative to the chosen OOI.
        Opacity encodes local confidence ρ^density_weight · R^(1−density_weight), passed
        through a gentle nonlinearity to balance visibility and noise suppression.

        Parameters
        ----------
        ax             : matplotlib Axes
        ooi            : float — Orientation of Interest in degrees, compass convention:
                         0=east, 90=north, 180=west, 270=south (default 0.0 = horizontal).
                         Orientations are unsigned so ooi and ooi±180 are equivalent.
        density_weight : float in [0, 1] — blend between density and coherence for opacity:
                         1.0 → density only, 0.0 → coherence only, 0.5 → geometric mean
        cmap           : str — base diverging colormap name (default 'coolwarm')
        degree         : float — sharpness of the opacity nonlinearity
        quantile       : float — center of the opacity nonlinearity
        """
        # -- colormap cache (invalidated on cmap change) -----------------------
        if cmap != self._cmap_name:
            self._cyclic_cmap = _make_cyclic_colormap(cmap)
            self._cmap_name   = cmap

        # -- opacity cache (invalidated on density_weight / degree / quantile change) --
        opacity_key = (density_weight, degree, quantile)
        if opacity_key != self._opacity_key:
            confidence      = self._field.rho ** density_weight * self._field.R ** (1.0 - density_weight)
            self._opacity   = _opacity_from_confidence(confidence,
                                                       degree=degree,
                                                       quantile=quantile)
            self._opacity_key = opacity_key

        # -- OOI rotation (always cheap) ---------------------------------------
        omega_display = np.mod(
            self._field.omega_mean - np.deg2rad(ooi), math.pi
        )

        # -- compose RGBA and render -------------------------------------------
        norm = Normalize(vmin=0.0, vmax=math.pi)
        sm   = cm.ScalarMappable(norm=norm, cmap=self._cyclic_cmap)

        shape = self._field.grid_shape
        rgba  = sm.to_rgba(omega_display.reshape(shape))
        alpha = np.where(np.isnan(self._field.omega_mean.reshape(shape)),
                         0.0,
                         self._opacity.reshape(shape))
        rgba[..., -1] = alpha

        ax.imshow(rgba, extent=(1, self._w, self._h, 1),
                  origin="upper", interpolation="gaussian")
        ax.set_xlim(1, self._w)
        ax.set_ylim(self._h, 1)
        ax.set_xticks([])
        ax.set_yticks([])