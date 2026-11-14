"""Calibrate synthetic rendering so it measures like the real image.

Usage:
    python -I src/calibrate_render.py <real_stats.json> <out_profile.json>

The raw measured profile (interior OD quantiles, blur sigma) cannot be used
as-is: blur lowers the stain of thin walls and the edge fit itself has a
geometry-dependent bias. So the render parameters (stain gain g, blur b) are
adjusted until the *same measurement procedure* (measure_real.describe) gives
the same tissue OD median and edge sigma on the render as on the real image.
Walls get thicknesses spread over 3-12 px, cells ~60 px, so that both thin
septa and thick tissue (with plateaus for the edge fit) are present.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_real as mr  # noqa: E402
import synthetic as syn  # noqa: E402

W = H = 1024


def calibration_scene(seed=11, cell=60, tmin=3.0, tmax=12.0):
    rng = np.random.default_rng(seed)
    seeds = syn.poisson_seeds(W, H, radius=cell, rng=rng, pad=3 * cell)
    segs = syn.voronoi_segments(seeds, W, H, margin=2 * cell)
    t = rng.uniform(tmin, tmax, len(segs))
    return segs, t, syn.rasterize(segs, W, H, t)


def measure(profile, mask, seed=0):
    rgb = syn.render_rgb_real(mask, np.random.default_rng(seed), profile)
    d = mr.describe(rgb, np.random.default_rng(0))
    return d["tissue_od_raw"], d["edge_blur"]["sigma_median"]


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    real = json.loads(Path(sys.argv[1]).read_text())
    base = real["render_profile"]
    target_p50 = real["tissue_od_raw"]["p50"]
    target_sig = real["edge_blur"]["sigma_median"]
    _, _, mask = calibration_scene()

    g, b = 1.0, base["blur_sigma"]
    log = []
    for it in range(6):
        prof = {**base, "od_quantiles": (np.asarray(base["od_quantiles"]) * g).tolist(), "blur_sigma": b}
        od, sig = measure(prof, mask)
        log.append({"iter": it, "gain": g, "blur": b, "p10": od["p10"], "p50": od["p50"],
                    "p90": od["p90"], "edge_sigma": sig})
        print(f"iter {it}: gain {g:.3f} blur {b:.3f} -> p10 {od['p10']:.3f} p50 {od['p50']:.3f} "
              f"p90 {od['p90']:.3f} edge sigma {sig:.3f}")
        if abs(od["p50"] / target_p50 - 1) < 0.02 and abs(sig / target_sig - 1) < 0.03:
            break
        g *= target_p50 / od["p50"]
        b *= target_sig / sig
    out = {"profile": prof, "gain": g, "blur_sigma_applied": b,
           "target": {"p10": real["tissue_od_raw"]["p10"], "p50": target_p50,
                      "p90": real["tissue_od_raw"]["p90"], "edge_sigma": target_sig},
           "achieved": log[-1], "log": log}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=2))
    print("target:", out["target"])


if __name__ == "__main__":
    main()
