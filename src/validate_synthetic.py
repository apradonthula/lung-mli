"""Validate the line-counting MLI on synthetic images with exactly known Lm.

Usage:
    python -I src/validate_synthetic.py <out_dir> [render_profile_calibrated.json]

Three kinds of check:
  exact     counter on an exact rasterised mask must reproduce the
            pixel-resolvable ground-truth transition count T_pixel exactly.
  cleanup   artifacts (fragments / pinholes) are added; the cleaned count must
            match the same cleanup applied to the artifact-free mask (within tol).
  e2e       render RGB (calibrated to the real image) -> segment.tissue_mask
            -> count; compared with the geometric truth (within tol).
Every row also reports error vs the continuous geometric truth (Lm_surface),
which includes pixel-resolution effects.

Writes results.csv and example images to out_dir. Exit code 1 if any check fails.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from skimage import io, measure

sys.path.insert(0, str(Path(__file__).resolve().parent))  # src/, since -I drops it

import mli  # noqa: E402
import segment  # noqa: E402
import synthetic as syn  # noqa: E402

W = H = 1024
SPACING = 8


def foam(cell, t, seed, drop=0.0):
    rng = np.random.default_rng(seed)
    seeds = syn.poisson_seeds(W, H, radius=cell, rng=rng, pad=3 * cell)
    segs = syn.voronoi_segments(seeds, W, H, margin=2 * cell)
    if drop:
        segs = syn.drop_segments(segs, drop, rng)
    return segs, syn.rasterize(segs, W, H, t), rng


def save_mask(path, m):
    io.imsave(path, (m * 255).astype(np.uint8), check_contrast=False)


def result(case, kind, truth, meas, ok, detail, expected_bad=False):
    return {
        "case": case,
        "kind": kind,
        "Lm_true_surface": round(truth["Lm_surface"], 3),
        "Lm_true_septal": round(truth["Lm_septal"], 3),
        "Lm_measured": round(meas["Lm"], 3),
        "err_vs_geometry_pct": round(100 * (meas["Lm"] / truth["Lm_surface"] - 1), 2),
        "detail": detail,
        "pass": "n/a (expected bad)" if expected_bad else bool(ok),
    }


def exact_check(case, segs, m, t, w=W, valid=None):
    truth = syn.ground_truth(segs, w, H, t, SPACING)
    meas = mli.measure_lm(m, SPACING, valid=valid)
    return result(case, "exact", truth, meas, meas["T"] == truth["T_pixel"],
                  f"T={meas['T']} T_pixel={truth['T_pixel']} T_geom={truth['T']}")


def cleanup_check(case, truth, dirty, clean, tol_pct, damage_tol_pct=1.5, expected_bad=False, **kw):
    """Pass only if (a) artifacts are removed: dirty ~= clean after the same cleanup, and
    (b) the cleanup does not damage real structure: clean-after-cleanup ~= geometric truth
    (damage_tol covers the ~1% pixel-resolution bias seen in the exact checks)."""
    meas = mli.measure_lm(dirty, SPACING, **kw)
    ref = mli.measure_lm(clean, SPACING, **kw)
    removal = 100 * (meas["Lm"] / ref["Lm"] - 1)
    damage = 100 * (ref["Lm"] / truth["Lm_surface"] - 1)
    ok = abs(removal) <= tol_pct and abs(damage) <= damage_tol_pct
    return result(case, "cleanup", truth, meas, ok,
                  f"artifact residual {removal:+.2f}% (tol {tol_pct}); "
                  f"cleanup damage {damage:+.2f}% (tol {damage_tol_pct})", expected_bad)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    profile = None
    if len(sys.argv) == 3:
        import json
        profile = json.loads(Path(sys.argv[2]).read_text())["profile"]
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    # 1. Regular square grid (includes test lines that run inside horizontal walls).
    for p in (24, 48, 96):
        segs = syn.grid_segments(W, H, p)
        rows.append(exact_check(f"grid p={p} t=4", segs, syn.rasterize(segs, W, H, 4), 4))

    # 2. Voronoi foams across cell sizes and wall thicknesses.
    for cell in (20, 40, 80):
        for t in (3, 5, 8):
            for seed in (1, 2):
                segs, m, _ = foam(cell, t, seed)
                if seed == 1 and t == 5:
                    save_mask(out / f"foam_cell{cell}_t{t}_mask.png", m)
                rows.append(exact_check(f"foam cell={cell} t={t} seed={seed}", segs, m, t))

    # 3. Septal destruction (walls removed -> enlarged airspaces).
    for drop in (0.3, 0.6):
        segs, m, _ = foam(20, 5, 3, drop=drop)
        save_mask(out / f"foam_cell20_drop{int(drop * 100)}_mask.png", m)
        rows.append(exact_check(f"foam cell=20 t=5 drop={drop}", segs, m, 5))

    # 4. Valid region: count only the left half (edge of valid region = line end).
    segs, m, _ = foam(40, 5, 6)
    valid = np.zeros_like(m)
    valid[:, : W // 2] = True
    rows.append(exact_check("valid=left half", segs, m, 5, w=W // 2, valid=valid))

    # 5. Floating fragments (3 px discs in airspace, walls 5 px): sweep min_tissue.
    segs, m, rng = foam(40, 5, 4)
    truth = syn.ground_truth(segs, W, H, 5, SPACING)
    mf, nfrag = syn.add_fragments(m, segs, n=1500, radius=1.5, clearance=3, rng=rng)
    save_mask(out / "foam_fragments_mask.png", mf)
    for mt in (0, 2, 4, 5, 6):
        rows.append(cleanup_check(f"fragments n={nfrag} min_tissue={mt}", truth, mf, m, 0.5,
                                  expected_bad=mt <= 3, min_tissue=mt))

    # 6. Pinholes (3 px) inside thick walls (t=8): sweep min_air.
    segs, m, rng = foam(40, 8, 5)
    truth = syn.ground_truth(segs, W, H, 8, SPACING)
    mh, nh = syn.add_pinholes(m, n=3000, radius=1.0, rng=rng)
    for ma in (0, 3, 4):
        rows.append(cleanup_check(f"pinholes n={nh} min_air={ma}", truth, mh, m, 0.5,
                                  expected_bad=ma < 3, min_air=ma))

    # 7. End-to-end on renders calibrated to the real image:
    #    render RGB -> segmentation (v1 global vs v2 halfmax) -> count, vs geometric truth.
    #    v1 is reported for reference only; v2 must be within 2%.
    if profile is not None:
        cases = [(20, 4), (40, 5), (80, 6), (160, 6), (60, "3-12")]
        for cell, t in cases:
            rng = np.random.default_rng(7)
            seeds = syn.poisson_seeds(W, H, radius=cell, rng=rng, pad=3 * cell)
            segs = syn.voronoi_segments(seeds, W, H, margin=2 * cell)
            tt = rng.uniform(3, 12, len(segs)) if t == "3-12" else t
            m = syn.rasterize(segs, W, H, tt)
            rgb = syn.render_rgb_real(m, rng, profile)
            io.imsave(out / f"render_real_cell{cell}_t{t}.png", rgb, check_contrast=False)
            save_mask(out / f"render_real_cell{cell}_t{t}_truth.png", m)
            truth = syn.ground_truth(segs, W, H, tt, SPACING)
            for method in ("global", "halfmax"):
                seg_mask, _, _ = segment.tissue_mask(rgb, method=method)
                if cell == 40:
                    save_mask(out / f"render_real_cell{cell}_t{t}_{method}.png", seg_mask)
                meas = mli.measure_lm(seg_mask, SPACING, min_tissue=3)
                err = 100 * (meas["Lm"] / truth["Lm_surface"] - 1)
                offset = (seg_mask.sum() - m.sum()) / measure.perimeter(m)
                rows.append(result(f"e2e-real {method} cell={cell} t={t}", "e2e", truth, meas,
                                   abs(err) <= 2.0,
                                   f"mean edge offset {offset:+.2f} px/side; tissue frac "
                                   f"{seg_mask.mean():.3f} vs true {m.mean():.3f} (tol 2%)",
                                   expected_bad=method == "global"))

    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{'case':34s} {'kind':7s} {'true':>7s} {'meas':>7s} {'err%':>6s}  {'pass':5s} detail")
    print("-" * 110)
    for r in rows:
        print(f"{r['case']:34s} {r['kind']:7s} {r['Lm_true_surface']:7.2f} {r['Lm_measured']:7.2f} "
              f"{r['err_vs_geometry_pct']:6.2f}  {str(r['pass']):5s} {r['detail']}")
    checked = [r for r in rows if isinstance(r["pass"], bool)]
    failed = [r for r in checked if not r["pass"]]
    print(f"\n{len(checked)} checks, {len(failed)} failed: {[r['case'] for r in failed]}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
