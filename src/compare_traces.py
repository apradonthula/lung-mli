"""Compare automatic tissue masks with hand traces on small crops.

Usage:
    python -I src/compare_traces.py packet <image> <traces_dir>
        write crop_<K>.png (native) and crop_<K>_x4.png (for tracing) + README
    python -I src/compare_traces.py check <image> <traces_dir>
        show the 1x mask read from each trace_<K>.png (traces_dir/check/)
    python -I src/compare_traces.py compare <image> <traces_dir>
        read trace_<K>.png for each crop present, print + save comparison.csv
    python -I src/compare_traces.py selftest <render_profile_calibrated.json>
        run the same comparison on a synthetic render with its ground truth as
        the "trace" (checks the metrics agree with validate_synthetic.py)

Trace format: trace_<K>.png at 1x (200x200) or 4x (800x800). Walls either
painted pure red (R>200, G<70, B<70) over the crop, or a black/white mask
(white = wall). Red traces: soft brush edges are weighted by how much red they
contain (estimated against the original crop), then 4x traces are reduced to
1x by mean coverage >= 0.5 per 4x4 block.

Metrics, automatic mask vs trace, per crop and pooled:
  edge offset   (area_auto - area_trace) / perimeter_trace: mean outward shift
                of the auto edge in px per side (+ = auto walls thicker)
  bnd median    median signed distance of auto boundary pixels from the trace
                boundary (+ = outside the traced wall)
  width         median wall width on the skeleton, 2*EDT - 1 (px)
  Lm            test lines every 4 px, h+v, min_tissue=3; pooled over crops
  dice          overlap of the two masks
All pieces are kept here (free cells included, as traced). For the comparison
with free pieces removed from both, see compare_walls.py.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import io, measure, morphology, segmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crops  # noqa: E402
import mli  # noqa: E402
import segment  # noqa: E402

SPACING, MIN_TISSUE = 4, 3
METHODS = [
    ("v1 global 0.10", dict(method="global", offset=0.10)),
    ("v2 halfmax 0.5", dict(method="halfmax", frac=0.5, free_piece_px=None)),
    ("v2 halfmax 0.6", dict(method="halfmax", frac=0.6, free_piece_px=None)),
    ("v2 halfmax 0.7", dict(method="halfmax", frac=0.7, free_piece_px=None)),
]

README = """# Hand-tracing crops

Three 200x200 px crops from Emphysema_low_mag.jpg (Nephron, CC BY-SA 3.0).
Boxes (x0, y0, size in original pixels): {boxes}

For each crop K in A, B, C:
1. Open crop_K_x4.png (4x enlarged, nearest neighbour; each original pixel is a
   4x4 block). crop_K.png is the same crop at native size if you prefer it.
2. Paint every alveolar wall / tissue region in pure red (255, 0, 0) over the
   image, filling the full wall width out to where you judge the edge to be.
   Leave airspace unpainted. Free-floating cells: paint them only if you would
   count them as tissue. Alternatively export a black/white mask (white = wall).
3. Save as trace_K.png in this folder, same size as the image you traced on
   (800x800 or 200x200).
Check what the script reads:  python -I src/compare_traces.py check data/raw/commons/Emphysema_low_mag.jpg {dir}
Then run:  python -I src/compare_traces.py compare data/raw/commons/Emphysema_low_mag.jpg {dir}
"""


def _paint_coverage(rgb, ref):
    """Per-pixel brush coverage in [0, 1]: rgb = ref + a * (paint - ref).

    Soft or anti-aliased brushes blend the paint with the crop at stroke edges;
    a strict colour test drops those pixels and shrinks the trace slightly.
    The paint colour is the median of the clearly red pixels."""
    strict = (rgb[..., 0] > 200) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)
    paint = np.median(rgb[strict], axis=0)
    d = paint - ref
    a = ((rgb - ref) * d).sum(-1) / np.maximum((d * d).sum(-1), 1.0)
    return np.clip(a, 0, 1), strict, paint


def read_trace(path, size, ref=None, details=False):
    """Trace -> boolean 1x mask. ref: the crop at the trace's resolution (RGB),
    used to measure brush coverage at soft stroke edges when walls are painted red."""
    t = io.imread(path)
    info = {"mode": "mask"}
    if t.ndim == 3:
        rgb = t[..., :3].astype(float)
        strict = (rgb[..., 0] > 200) & (rgb[..., 1] < 70) & (rgb[..., 2] < 70)
        if strict.any() and ref is not None and ref.shape[:2] == rgb.shape[:2]:
            cov, strict, paint = _paint_coverage(rgb, ref[..., :3].astype(float))
            info = {"mode": "red, coverage-weighted", "paint": paint.round().astype(int).tolist()}
        elif strict.any():
            cov = strict.astype(float)
            info = {"mode": "red, strict colour test"}
        else:
            cov = (rgb.mean(-1) > 127).astype(float)
        info["strict"] = strict if strict.any() else None
    else:
        cov = (t > 127).astype(float)
    info["coverage"] = cov
    f = cov.shape[0] // size
    if cov.shape != (size * f, size * f):
        raise ValueError(f"{path}: shape {cov.shape} is not a multiple of {size}x{size}")
    m = cov.reshape(size, f, size, f).mean(axis=(1, 3)) >= 0.5 if f > 1 else cov >= 0.5
    return (m, info) if details else m


def _ref_for(rgb, k, trace_path):
    """The crop at the resolution the trace was drawn on (1x or 4x)."""
    c = crops.cut(rgb, k)
    n = io.imread(trace_path).shape[0]
    f = n // c.shape[0]
    return np.kron(c, np.ones((f, f, 1), np.uint8)) if f > 1 else c


def widths(mask):
    sk = morphology.skeletonize(mask)
    d = ndi.distance_transform_edt(mask)
    w = 2 * d[sk] - 1
    return w[w > 0]


def compare_pair(auto, trace):
    per = measure.perimeter(trace)
    inside = ndi.distance_transform_edt(trace)
    outside = ndi.distance_transform_edt(~trace)
    b = segmentation.find_boundaries(auto, mode="inner")
    signed = np.where(trace[b], -(inside[b] - 0.5), outside[b] + 0.5)
    inter = (auto & trace).sum()
    return {
        "edge_offset": (auto.sum() - trace.sum()) / per,
        "bnd_median": float(np.median(signed)) if signed.size else float("nan"),
        "width_auto": float(np.median(widths(auto))),
        "width_trace": float(np.median(widths(trace))),
        "dice": 2 * inter / (auto.sum() + trace.sum()),
    }


def run_comparison(pairs):
    """pairs: {crop: (trace, {method_label: auto_mask})}. Returns rows."""
    rows = []
    for label, _ in METHODS:
        pooled = {"auto": [0, 0], "trace": [0, 0]}
        for k, (trace, autos) in pairs.items():
            auto = autos[label]
            r = compare_pair(auto, trace)
            ma = mli.measure_lm(auto, SPACING, min_tissue=MIN_TISSUE)
            mt = mli.measure_lm(trace, SPACING, min_tissue=MIN_TISSUE)
            pooled["auto"][0] += ma["L"]; pooled["auto"][1] += ma["T"]
            pooled["trace"][0] += mt["L"]; pooled["trace"][1] += mt["T"]
            rows.append({"method": label, "crop": k, **{x: round(v, 3) for x, v in r.items()},
                         "Lm_auto": round(ma["Lm"], 1), "Lm_trace": round(mt["Lm"], 1),
                         "Lm_err_pct": round(100 * (ma["Lm"] / mt["Lm"] - 1), 1)})
        la, ta = pooled["auto"]
        lt, tt = pooled["trace"]
        rows.append({"method": label, "crop": "pooled", "Lm_auto": round(2 * la / ta, 1),
                     "Lm_trace": round(2 * lt / tt, 1),
                     "Lm_err_pct": round(100 * ((2 * la / ta) / (2 * lt / tt) - 1), 1)})
    return rows


def print_rows(rows):
    cols = ["method", "crop", "edge_offset", "bnd_median", "width_auto", "width_trace", "dice",
            "Lm_auto", "Lm_trace", "Lm_err_pct"]
    print("  ".join(f"{c:>14s}" if i > 1 else f"{c:15s}" for i, c in enumerate(cols)))
    for r in rows:
        print("  ".join((f"{str(r.get(c, '')):>14s}" if i > 1 else f"{str(r.get(c, '')):15s}")
                        for i, c in enumerate(cols)))


def packet(img_path, d):
    d.mkdir(parents=True, exist_ok=True)
    rgb = io.imread(img_path)[..., :3]
    for k in crops.CROPS:
        c = crops.cut(rgb, k)
        io.imsave(d / f"crop_{k}.png", c, check_contrast=False)
        io.imsave(d / f"crop_{k}_x4.png", np.kron(c, np.ones((4, 4, 1), np.uint8)), check_contrast=False)
    (d / "README.md").write_text(README.format(boxes=crops.CROPS, dir=d))
    print("wrote tracing packet to", d)


def compare(img_path, d):
    rgb = io.imread(img_path)[..., :3]
    full = {label: segment.tissue_mask(rgb, **kw)[0] for label, kw in METHODS}
    pairs = {}
    for k, (_, _, size) in crops.CROPS.items():
        p = d / f"trace_{k}.png"
        if p.exists():
            pairs[k] = (read_trace(p, size, _ref_for(rgb, k, p)), {lab: crops.cut(m, k) for lab, m in full.items()})
    if not pairs:
        sys.exit(f"no trace_<K>.png found in {d}")
    rows = run_comparison(pairs)
    print_rows(rows)
    with open(d / "comparison.csv", "w", newline="") as f:
        fields = list(dict.fromkeys(c for r in rows for c in r))  # ordered union of keys
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def check(img_path, d):
    """Show what the script reads from each trace present, before comparing."""
    from skimage import segmentation
    rgb = io.imread(img_path)[..., :3]
    out = d / "check"
    out.mkdir(exist_ok=True)
    for k, (_, _, size) in crops.CROPS.items():
        p = d / f"trace_{k}.png"
        if not p.exists():
            continue
        ref = _ref_for(rgb, k, p)
        m, info = read_trace(p, size, ref, details=True)
        strict = info.get("strict")
        print(f"trace_{k}: {info['mode']}" + (f", paint colour {info['paint']}" if "paint" in info else ""))
        if strict is not None:
            cov = info["coverage"]
            soft = (cov > 0.02) & ~strict
            m_strict = read_trace_strict(strict, size)
            print(f"  fully red px {strict.sum()}, soft-edge px {soft.sum()} "
                  f"(median coverage {np.median(cov[soft]) if soft.any() else 0:.2f})")
            print(f"  soft edges add {m.sum() - m_strict.sum()} px at 1x = "
                  f"{(m.sum() - m_strict.sum()) / measure.perimeter(m_strict):+.3f} px/side vs strict test")
        lab, n = ndi.label(m)
        holes = ndi.binary_fill_holes(m) & ~m
        w = widths(m)
        print(f"  1x mask: {m.sum()} tissue px ({m.mean():.3f}), {n} pieces "
              f"(sizes {sorted(np.bincount(lab.ravel())[1:].tolist(), reverse=True)}), "
              f"{ndi.label(holes)[1]} enclosed gaps, median wall width {np.median(w):.0f} px")
        Z = 4
        big = np.kron(crops.cut(rgb, k), np.ones((Z, Z, 1), np.uint8))
        mb = np.kron(m.astype(np.uint8), np.ones((Z, Z), np.uint8)).astype(bool)
        mask_img = np.repeat(np.where(mb, 255, 0).astype(np.uint8)[..., None], 3, -1)
        ov = big.copy()
        ov[segmentation.find_boundaries(mb, mode="inner")] = [0, 150, 255]
        gap = np.full((big.shape[0], 12, 3), 255, np.uint8)
        io.imsave(out / f"trace_{k}_read.png", np.hstack([mask_img, gap, ov]), check_contrast=False)
        io.imsave(out / f"trace_{k}_mask_1x.png", (m * 255).astype(np.uint8), check_contrast=False)
        print(f"  wrote {out / f'trace_{k}_read.png'}")


def read_trace_strict(strict, size):
    f = strict.shape[0] // size
    return strict.reshape(size, f, size, f).mean(axis=(1, 3)) >= 0.5 if f > 1 else strict


def selftest(profile_path):
    import synthetic as syn

    profile = json.loads(Path(profile_path).read_text())["profile"]
    W = 1024
    rng = np.random.default_rng(7)
    seeds = syn.poisson_seeds(W, W, radius=60, rng=rng, pad=180)
    segs = syn.voronoi_segments(seeds, W, W, margin=120)
    truth = syn.rasterize(segs, W, W, rng.uniform(3, 12, len(segs)))
    rgb = syn.render_rgb_real(truth, rng, profile)
    full = {label: segment.tissue_mask(rgb, **kw)[0] for label, kw in METHODS}
    pairs = {}
    for k, (x0, y0) in {"S1": (100, 100), "S2": (500, 300), "S3": (300, 700)}.items():
        sl = (slice(y0, y0 + 200), slice(x0, x0 + 200))
        pairs[k] = (truth[sl], {lab: m[sl] for lab, m in full.items()})
    print_rows(run_comparison(pairs))
    # whole-image reference, same metric as validate_synthetic.py
    for lab, m in full.items():
        print(f"whole-image edge offset {lab}: {(m.sum() - truth.sum()) / measure.perimeter(truth):+.2f} px/side")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode = sys.argv[1]
    if mode == "packet" and len(sys.argv) == 4:
        packet(sys.argv[2], Path(sys.argv[3]))
    elif mode == "check" and len(sys.argv) == 4:
        check(sys.argv[2], Path(sys.argv[3]))
    elif mode == "compare" and len(sys.argv) == 4:
        compare(sys.argv[2], Path(sys.argv[3]))
    elif mode == "selftest" and len(sys.argv) == 3:
        selftest(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
