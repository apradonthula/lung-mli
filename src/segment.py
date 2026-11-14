"""Step 1 of the MLI pipeline: separate tissue from airspace in an H&E lung image.

Usage:
    python -I src/segment.py <image> <out_dir> [--method halfmax|global]
                             [--low 0.10] [--frac 0.5]

Writes <stem>_<method>_mask.png (white = tissue) and <stem>_<method>_overlay.png
to out_dir, and prints basic statistics.

Methods
  global   (v1, approved overlay 2026-11-13) Gaussian-smoothed OD (sigma 1.5) >
           background + `low`. Finds pale tissue, but on blurred edges the
           boundary sits at ~10% of the wall's stain, so every wall is
           thickened (~3 px/side on synthetic renders).
  halfmax  (v2, default) same low threshold to *find* tissue (candidates), then
           each wall edge is placed where OD crosses half of the local tissue
           level:  OD >= bg + frac * (local_mean - bg).  local_mean is the mean OD
           of current tissue pixels in a Gaussian window (normalised
           convolution), re-estimated n_iter times as the fringe is trimmed.
           A blurred step edge crosses its half height at the true edge, so
           this undoes the blur-induced thickening for walls wider than the blur.
           Using a local *mean* (not max) keeps cytoplasm next to dark nuclei.

Dense holes (halfmax default, 2026-11-14): enclosed holes whose fine OD never
drops to bg + low (the threshold used to find tissue) are filled; they are pale
tissue, not air.

Free pieces (halfmax default, 2026-11-14): connected tissue pieces smaller than
FREE_PIECE_PX are dropped as free cells (alveolar macrophages etc.), which are not
septa. Pieces touching the image edge are kept (they may be walls cut by the
edge). 300 px is set from the hand traces of Emphysema_low_mag.jpg (see
compare_walls.py): the 15 small pieces the user traced came out as automatic
pieces of 34-172 px each, except two cells joined by a faint untraced strand
(207 px). 300 px clears that by ~45%, and the walls-only comparison is the same
for cutoffs 300 and 400. Size alone cannot separate everything: some detached,
elongated wall fragments are 200-300 px and some round free cells exceed 300 px
(outputs/real/walls_only/small_pieces.png). The value is in native pixels of
that image and must be rescaled for other magnifications.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, io, morphology, segmentation


def od_sum(rgb):
    rgb = rgb[..., :3].astype(np.float64)
    return -np.log((rgb + 1.0) / 256.0).sum(axis=2)


def background_od(od):
    """Background (airspace/glass) level: the mode of the OD histogram.

    Airspace dominates lung fields, so the tallest histogram bin is the glass.
    """
    hist, edges = np.histogram(od, bins=512)
    i = hist.argmax()
    return 0.5 * (edges[i] + edges[i + 1])


def _cleanup(mask, min_object_px, max_hole_px):
    # Scale cleanup sizes to the image so the defaults work across resolutions.
    n = mask.size
    if min_object_px is None:
        min_object_px = max(20, int(n * 2e-5))
    if max_hole_px is None:
        max_hole_px = max(10, int(n * 5e-6))
    mask = morphology.remove_small_objects(mask, max_size=min_object_px - 1)
    mask = morphology.remove_small_holes(mask, max_size=max_hole_px - 1)
    return mask


FREE_PIECE_PX = 300


def drop_free_pieces(mask, min_px=FREE_PIECE_PX, keep_border=False):
    """Remove connected pieces smaller than min_px (8-connectivity, the same
    as ndi.label with a full structure). keep_border=True keeps pieces touching
    the array edge: in a crop, those may be attached to walls outside it."""
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask.copy()
    size = np.bincount(lab.ravel())
    keep = size >= min_px
    if keep_border:
        edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        keep[edge] = True
    keep[0] = False
    return keep[lab]


def fill_dense_holes(mask, od, level):
    """Fill enclosed holes whose OD never falls to airspace level (min OD > level).

    Half-max edges can open a gap inside a wall where a pale core sits next to
    dark nuclei (crop A junction: OD 0.15-0.24 inside, 0.64 around). A real air
    gap reaches background OD somewhere, so it stays open. Returns (mask, n_filled).
    """
    holes, n = ndi.label(ndi.binary_fill_holes(mask) & ~mask)
    if n == 0:
        return mask, 0
    lo = ndi.minimum(od, holes, index=np.arange(1, n + 1))
    fill = np.concatenate([[False], np.asarray(lo) > level])
    return mask | fill[holes], int(fill.sum())


def tissue_mask_global(rgb, sigma=1.5, offset=0.10, min_object_px=None, max_hole_px=None):
    """v1: smoothed OD > background + offset. Returns (mask, threshold, od_smoothed)."""
    od = filters.gaussian(od_sum(rgb), sigma=sigma)
    thr = background_od(od) + offset
    return _cleanup(od > thr, min_object_px, max_hole_px), thr, od


def tissue_mask_halfmax(rgb, low=0.10, frac=0.5, cand_sigma=1.5, edge_sigma=0.7,
                        window=3.0, n_iter=3, min_object_px=None, max_hole_px=None,
                        free_piece_px=FREE_PIECE_PX, fill_dense=True):
    """v2: low threshold finds tissue, local half-maximum places the edges.

    free_piece_px: drop pieces smaller than this (free cells); None keeps them.
    fill_dense: fill enclosed holes whose OD stays above the low threshold
    (bg + low) everywhere, i.e. never reaches airspace density.
    Returns (mask, low_threshold, od_fine).
    """
    od = od_sum(rgb)
    od_c = filters.gaussian(od, sigma=cand_sigma)
    bg = background_od(od_c)
    cand = od_c > bg + low
    od_f = filters.gaussian(od, sigma=edge_sigma)
    m = cand
    for _ in range(n_iter):
        w = filters.gaussian(m.astype(float), sigma=window)
        local = filters.gaussian(od_f * m, sigma=window) / np.maximum(w, 1e-6)
        m = cand & (od_f >= bg + frac * (local - bg))
    m = _cleanup(m, min_object_px, max_hole_px)
    if fill_dense:
        m, _ = fill_dense_holes(m, od_f, bg + low)
    if free_piece_px:
        m = drop_free_pieces(m, free_piece_px, keep_border=True)
    return m, bg + low, od_f


def tissue_mask(rgb, method="halfmax", **kw):
    """Dispatch to the chosen method. Returns (mask, threshold, od)."""
    if method == "halfmax":
        return tissue_mask_halfmax(rgb, **kw)
    if method == "global":
        return tissue_mask_global(rgb, **kw)
    raise ValueError(method)


def overlay(rgb, mask, color=(0, 200, 255), alpha=0.35):
    """Original image with tissue tinted and its boundary drawn in a solid color."""
    out = rgb[..., :3].astype(np.float64).copy()
    c = np.array(color, dtype=np.float64)
    out[mask] = (1 - alpha) * out[mask] + alpha * c
    edges = segmentation.find_boundaries(mask, mode="inner")
    edges = ndi.binary_dilation(edges, iterations=max(1, rgb.shape[1] // 1500))
    out[edges] = (20, 20, 20)
    return out.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("out_dir")
    ap.add_argument("--method", default="halfmax", choices=["halfmax", "global"])
    ap.add_argument("--low", type=float, default=0.10, help="OD above background to find tissue")
    ap.add_argument("--frac", type=float, default=0.5, help="halfmax: edge at this fraction of local level")
    a = ap.parse_args()
    img_path, out_dir = Path(a.image), Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = io.imread(img_path)
    if a.method == "halfmax":
        mask, thr, _ = tissue_mask_halfmax(rgb, low=a.low, frac=a.frac)
    else:
        mask, thr, _ = tissue_mask_global(rgb, offset=a.low)

    stem = f"{img_path.stem}_{a.method}"
    io.imsave(out_dir / f"{stem}_mask.png", (mask * 255).astype(np.uint8), check_contrast=False)
    io.imsave(out_dir / f"{stem}_overlay.png", overlay(rgb, mask), check_contrast=False)
    print(f"image        {img_path.name}  {rgb.shape[1]}x{rgb.shape[0]} px  method={a.method}")
    print(f"low thr (OD) {thr:.3f}")
    print(f"tissue frac  {mask.mean():.3f}")
    print(f"objects      {ndi.label(mask)[1]}")
    print(f"wrote        {out_dir / (stem + '_mask.png')}")


if __name__ == "__main__":
    main()
