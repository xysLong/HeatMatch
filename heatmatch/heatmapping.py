import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image
import math

"""
Visualization

Implements the OOI-coded heatmap pipeline from §3.2 of the paper.

Public API : Heatmap
Internal   : _make_cyclic_colormap, _opacity_from_confidence, _load_grayscale
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


def _load_grayscale(path):
    """
    Load an image from path and return a float32 grayscale array in [0, 1].

    Uses PIL for format detection, so the actual file format takes precedence
    over the file extension (handles e.g. JPEG files named .png).
    """
    img = Image.open(str(path)).convert('L')   # 'L' = 8-bit grayscale
    return np.asarray(img, dtype=np.float32) / 255.0


class Heatmap:
    """
    OOI-coded heatmap renderer with two-level caching — §3.2.

    Wraps an OrientationField and caches intermediate results to avoid redundant
    computation when only display parameters change:

    - Background image  → loaded and converted to grayscale once at construction.
    - Cyclic colormap   → rebuilt only when `cmap` changes.
    - Opacity array     → rebuilt only when `opacity_density_weight`, `degree`, or `quantile` changes.
    - OOI rotation      → always cheap (one np.mod call), never cached.

    Parameters
    ----------
    field : OrientationField
        Precomputed orientation field from make_orientation_field.
    w, h  : stimulus width and height in pixels
    image : str, path-like, or None
        Optional path to a stimulus image. When provided it is converted to
        grayscale once and rendered as a background; the heatmap is composited
        on top with opacity scaled by `base_opacity`.
    """

    def __init__(self, field, w, h, image=None):
        self._field = field
        self._w     = w
        self._h     = h

        self._image_gray   = _load_grayscale(image) if image is not None else None

        self._cmap_name    = None
        self._cyclic_cmap  = None

        self._opacity_key  = None   # (opacity_density_weight, degree, quantile)
        self._opacity      = None

    def draw(self, ax, ooi=0.0, opacity_density_weight=1.0, base_opacity=0.85,
             cmap='coolwarm', degree=2.0, quantile=0.8):
        """
        Render the OOI-coded saccade heatmap onto a matplotlib Axes — §3.2.

        Hue encodes mean orientation ω̄_i relative to the chosen OOI.
        Opacity encodes local confidence ρ^opacity_density_weight · R^(1−opacity_density_weight),
        passed through a gentle nonlinearity and then scaled by base_opacity.

        Parameters
        ----------
        ax                    : matplotlib Axes
        ooi                   : float — Orientation of Interest in degrees, compass convention:
                                0=east, 90=north, 180=west, 270=south (default 0.0 = horizontal).
                                Orientations are unsigned so ooi and ooi±180 are equivalent.
        opacity_density_weight : float in [0, 1] — blend between density and coherence for opacity:
                                1.0 → density only, 0.0 → coherence only. Recommended 0.5–1.0.
        base_opacity          : float in [0, 1] — global opacity ceiling applied on top of the
                                data-derived opacity; useful when a background image is shown.
                                The final per-point alpha = data_opacity × base_opacity.
        cmap                  : str — base diverging colormap name (default 'coolwarm')
        degree                : float — sharpness of the opacity nonlinearity
        quantile              : float — center of the opacity nonlinearity
        """
        # -- colormap cache (invalidated on cmap change) -----------------------
        if cmap != self._cmap_name:
            self._cyclic_cmap = _make_cyclic_colormap(cmap)
            self._cmap_name   = cmap

        # -- opacity cache (invalidated on opacity_density_weight / degree / quantile change) --
        opacity_key = (opacity_density_weight, degree, quantile)
        if opacity_key != self._opacity_key:
            confidence        = self._field.rho ** opacity_density_weight * self._field.R ** (1.0 - opacity_density_weight)
            self._opacity     = _opacity_from_confidence(confidence, degree=degree, quantile=quantile)
            self._opacity_key = opacity_key

        # -- background image (grayscale) --------------------------------------
        if self._image_gray is not None:
            ax.imshow(self._image_gray, extent=(1, self._w, self._h, 1),
                      origin='upper', cmap='gray', vmin=0.0, vmax=1.0,
                      interpolation='bilinear')

        # -- OOI rotation (always cheap) ---------------------------------------
        omega_display = np.mod(self._field.omega_mean - np.deg2rad(ooi), math.pi)

        # -- compose RGBA and render -------------------------------------------
        norm = Normalize(vmin=0.0, vmax=math.pi)
        sm   = cm.ScalarMappable(norm=norm, cmap=self._cyclic_cmap)

        shape = self._field.grid_shape
        rgba  = sm.to_rgba(omega_display.reshape(shape))
        alpha = np.where(np.isnan(self._field.omega_mean.reshape(shape)),
                         0.0,
                         self._opacity.reshape(shape) * base_opacity)
        rgba[..., -1] = alpha

        ax.imshow(rgba, extent=(1, self._w, self._h, 1),
                  origin='upper', interpolation='gaussian')
        ax.set_xlim(1, self._w)
        ax.set_ylim(self._h, 1)
        ax.set_xticks([])
        ax.set_yticks([])
