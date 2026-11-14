"""Synthetic alveolar-like images with exactly known Lm.

Walls are line segments (Voronoi ridges or a square grid) thickened to width t:
a pixel is tissue iff its centre lies within t/2 of a segment (a "capsule").

Ground truth is computed from the geometry, not from pixels:
  * surface truth: along each test line, the capsule intervals are computed
    analytically, merged, and their boundaries counted -> T_true, Lm = 2L/T.
    This is what an ideal counter on this exact shape should report.
  * septal truth: number of crossings of the wall centre-lines -> Lm = L/C.
    This differs from the surface truth where walls meet: a line passing
    through a junction crosses 2 centre-lines but enters tissue only once.
Pixel coordinates: pixel (r, c) has its centre at (x=c, y=r).
"""
import numpy as np
from scipy.spatial import Voronoi
from scipy.stats import qmc

from mli import line_positions  # noqa: E402


# ---------------------------------------------------------------- geometry
def poisson_seeds(W, H, radius, rng, pad):
    """Blue-noise seeds (min distance `radius`) over the image plus a padding band."""
    size = max(W, H) + 2 * pad
    eng = qmc.PoissonDisk(d=2, radius=radius / size, seed=rng)
    pts = eng.fill_space() * size - pad
    return pts


def voronoi_segments(seeds, W, H, margin):
    """Finite Voronoi ridges whose bounding box touches the (expanded) image."""
    vor = Voronoi(seeds)
    segs = []
    for a, b in vor.ridge_vertices:
        if a < 0 or b < 0:
            continue
        (x1, y1), (x2, y2) = vor.vertices[a], vor.vertices[b]
        if max(x1, x2) < -margin or min(x1, x2) > W - 1 + margin:
            continue
        if max(y1, y2) < -margin or min(y1, y2) > H - 1 + margin:
            continue
        segs.append((x1, y1, x2, y2))
    return np.array(segs, float)


def grid_segments(W, H, period):
    """Square grid of walls, centred between pixel rows/cols, extending past the image."""
    xs = np.arange(period / 2 - 0.5, W, period)
    ys = np.arange(period / 2 - 0.5, H, period)
    segs = [(x, -period, x, H + period) for x in xs]
    segs += [(-period, y, W + period, y) for y in ys]
    return np.array(segs, float)


def drop_segments(segs, fraction, rng):
    """Remove a random fraction of walls (septal destruction)."""
    keep = rng.random(len(segs)) >= fraction
    return segs[keep]


# ------------------------------------------------------------- rasterising
def half_widths(segs, thickness):
    """Per-segment half-width; `thickness` is a scalar or one value per segment."""
    return np.broadcast_to(np.asarray(thickness, float) / 2, (len(segs),))


def rasterize(segs, W, H, thickness):
    """Boolean mask: True where the pixel centre is within thickness/2 of any segment.

    thickness: scalar, or an array with one wall thickness per segment.
    """
    mask = np.zeros((H, W), bool)
    for (x1, y1, x2, y2), h in zip(segs, half_widths(segs, thickness)):
        c0 = int(max(0, np.floor(min(x1, x2) - h)))
        c1 = int(min(W - 1, np.ceil(max(x1, x2) + h)))
        r0 = int(max(0, np.floor(min(y1, y2) - h)))
        r1 = int(min(H - 1, np.ceil(max(y1, y2) + h)))
        if c0 > c1 or r0 > r1:
            continue
        yy, xx = np.mgrid[r0:r1 + 1, c0:c1 + 1].astype(float)
        dx, dy = x2 - x1, y2 - y1
        d2 = dx * dx + dy * dy
        u = np.clip(((xx - x1) * dx + (yy - y1) * dy) / d2, 0, 1)
        dist2 = (xx - x1 - u * dx) ** 2 + (yy - y1 - u * dy) ** 2
        mask[r0:r1 + 1, c0:c1 + 1] |= dist2 <= h * h
    return mask


def add_fragments(mask, segs, n, radius, clearance, rng):
    """Add n small discs in airspace, at least `clearance` px from any wall.

    Returns (new_mask, n_added). Fragments are NOT walls: ground truth ignores them.
    """
    from scipy import ndimage as ndi
    from skimage.draw import disk

    H, W = mask.shape
    dist_to_wall = ndi.distance_transform_edt(~mask)
    out = mask.copy()
    added = 0
    tries = 0
    while added < n and tries < n * 50:
        tries += 1
        r, c = rng.integers(0, H), rng.integers(0, W)
        if dist_to_wall[r, c] < clearance + radius:
            continue
        rr, cc = disk((r, c), radius, shape=mask.shape)
        out[rr, cc] = True
        added += 1
    return out, added


def add_pinholes(mask, n, radius, rng):
    """Punch n small air holes fully inside tissue (should NOT change true Lm)."""
    from scipy import ndimage as ndi
    from skimage.draw import disk

    depth = ndi.distance_transform_edt(mask)
    cand = np.argwhere(depth > radius + 1.5)
    out = mask.copy()
    if len(cand) == 0:
        return out, 0
    pick = cand[rng.choice(len(cand), size=min(n, len(cand)), replace=False)]
    for r, c in pick:
        rr, cc = disk((r, c), radius, shape=mask.shape)
        out[rr, cc] = False
    return out, len(pick)


def render_rgb(mask, rng, blur=1.2, noise=4.0):
    """Render a mask as an H&E-like pink-on-white RGB image."""
    from skimage import filters

    tissue = np.array([225, 140, 190], float)
    glass = np.array([255, 255, 255], float)
    m = filters.gaussian(mask.astype(float), sigma=blur)
    img = glass * (1 - m[..., None]) + tissue * m[..., None]
    # mild stain texture inside tissue + sensor noise
    img -= 25 * m[..., None] * rng.random(mask.shape)[..., None]
    img += rng.normal(0, noise, img.shape)
    return img.clip(0, 255).astype(np.uint8)


# ----------------------------------------------------------- ground truth
def _capsule_interval(seg, y0, h):
    """Interval [a, b] of x where the horizontal line y=y0 is inside the capsule, or None."""
    x1, y1, x2, y2 = seg
    pieces = []
    for xi, yi in ((x1, y1), (x2, y2)):
        dy = y0 - yi
        if abs(dy) <= h:
            w = np.sqrt(h * h - dy * dy)
            pieces.append((xi - w, xi + w))
    dx, dy = x2 - x1, y2 - y1
    d2 = dx * dx + dy * dy
    lo, hi = -np.inf, np.inf
    # 0 <= ((x-x1)dx + (y0-y1)dy) / d2 <= 1
    k = (y0 - y1) * dy
    if dx != 0:
        a, b = (-k) / dx + x1, (d2 - k) / dx + x1
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    elif not (0 <= k / d2 <= 1):
        lo, hi = 1, 0
    # |(x-x1)dy - (y0-y1)dx| <= h*sqrt(d2)
    m = (y0 - y1) * dx
    r = h * np.sqrt(d2)
    if dy != 0:
        a, b = (m - r) / dy + x1, (m + r) / dy + x1
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    elif abs(m) > r:
        lo, hi = 1, 0
    if lo <= hi:
        pieces.append((lo, hi))
    if not pieces:
        return None
    return min(p[0] for p in pieces), max(p[1] for p in pieces)


def _truth_on_lines(segs, n_along, positions, hs):
    """Return (T_surface, T_pixel, C_centreline) summed over horizontal lines y = positions.

    T_pixel is the surface count after discarding what pixel sampling cannot
    resolve: tissue intervals that contain no pixel centre, and air gaps that
    contain no pixel centre (centres are at integer x). A correct counter on
    the rasterised mask must reproduce T_pixel exactly.
    """
    lo_dom, hi_dom = -0.5, n_along - 0.5
    T = 0
    Tp = 0
    C = 0
    ymin = np.minimum(segs[:, 1], segs[:, 3])
    ymax = np.maximum(segs[:, 1], segs[:, 3])
    for y0 in positions:
        sel = (ymin - hs <= y0) & (ymax + hs >= y0)
        near, near_h = segs[sel], hs[sel]
        # centre-line crossings
        s = near[(near[:, 1] - y0) * (near[:, 3] - y0) < 0]
        if len(s):
            t = (y0 - s[:, 1]) / (s[:, 3] - s[:, 1])
            xc = s[:, 0] + t * (s[:, 2] - s[:, 0])
            C += int(np.count_nonzero((xc > lo_dom) & (xc < hi_dom)))
        # surface boundaries of the union of capsules
        ivs = [iv for iv in (_capsule_interval(sg, y0, hh) for sg, hh in zip(near, near_h))
               if iv is not None]
        ivs = [(max(a, lo_dom), min(b, hi_dom)) for a, b in ivs if b > lo_dom and a < hi_dom]
        ivs.sort()
        merged = []
        for a, b in ivs:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            T += int(a > lo_dom) + int(b < hi_dom)
        # pixel-resolvable version: keep intervals covering >=1 pixel centre,
        # then merge neighbours whose gap holds no pixel centre.
        px = [(a, b) for a, b in merged if np.ceil(a) <= np.floor(b)]
        pm = []
        for a, b in px:
            if pm and not _gap_has_centre(pm[-1][1], a):
                pm[-1][1] = b
            else:
                pm.append([a, b])
        for a, b in pm:
            # a boundary is seen by pixels only if a pixel centre lies outside the interval
            Tp += int(np.ceil(a) > 0) + int(np.floor(b) < n_along - 1)
    return T, Tp, C


def _gap_has_centre(b, a):
    """True if the open gap (b, a) contains an integer (a pixel centre)."""
    k = np.floor(b) + 1
    return k < a


def ground_truth(segs, W, H, thickness, spacing=8, directions="hv"):
    """Exact Lm for the thickened-segment geometry along the same lines measure_lm uses.

    thickness: scalar, or an array with one wall thickness per segment.
    """
    h = np.array(half_widths(segs, thickness))
    L = T = Tp = C = 0
    if "h" in directions:
        rows = line_positions(H, spacing)
        t, tp, c = _truth_on_lines(segs, W, rows, h)
        L += W * len(rows)
        T += t
        Tp += tp
        C += c
    if "v" in directions:
        cols = line_positions(W, spacing)
        swapped = segs[:, [1, 0, 3, 2]]
        t, tp, c = _truth_on_lines(swapped, H, cols, h)
        L += H * len(cols)
        T += t
        Tp += tp
        C += c
    return {
        "L": L,
        "T": T,
        "T_pixel": Tp,
        "Lm_surface": 2 * L / T if T else float("inf"),
        "C": C,
        "Lm_septal": L / C if C else float("inf"),
    }


# ------------------------------------------------- realistic rendering (v2)
def render_rgb_real(mask, rng, profile, large_sigma=8.0, small_sigma=1.2, noise=2.0):
    """Render a mask with stain statistics measured on a real image.

    profile (from measure_real.render_profile):
      od_quantiles  101 quantiles (0..100%) of tissue OD (sum over RGB) to draw from
      colour        OD direction over R,G,B (sums to 1)
      blur_sigma    Gaussian blur (px) applied to transmitted intensity, i.e. optics
    Tissue OD = quantile-map of a spatially correlated field: a slow component
    (pale vs dense stretches of wall) plus a fast one (nucleus-sized spots).
    """
    from scipy.stats import norm
    from skimage import filters

    def field(s):
        f = filters.gaussian(rng.standard_normal(mask.shape), sigma=s)
        return (f - f.mean()) / f.std()

    z = (field(large_sigma) + field(small_sigma)) / np.sqrt(2)
    q = np.asarray(profile["od_quantiles"], float)
    od = np.interp(norm.cdf(z) * 100, np.linspace(0, 100, len(q)), q) * mask
    od_rgb = od[..., None] * np.asarray(profile["colour"], float)
    intensity = 256.0 * np.exp(-od_rgb) - 1.0  # inverse of OD = -ln((I+1)/256)
    for c in range(3):
        intensity[..., c] = filters.gaussian(intensity[..., c], sigma=profile["blur_sigma"],
                                             preserve_range=True)
    intensity += rng.normal(0, noise, intensity.shape)
    return intensity.clip(0, 255).round().astype(np.uint8)
