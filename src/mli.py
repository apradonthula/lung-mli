"""Mean linear intercept (Lm) by test-line counting on a binary tissue mask.

Definition used (septal-intercept convention):
    Lm = L / I,   I = T / 2
L = total test-line length inside the valid region (px)
T = number of air<->tissue transitions along the lines
Each septum crossed completely gives 2 transitions (one intercept). A wall cut
by the end of a line segment (image edge or edge of the valid region) gives 1
transition, i.e. half an intercept, which keeps edge handling unbiased on average.

Cleanup along each line (applied before counting):
    min_air:    interior air runs shorter than this are filled (pinholes in walls)
    min_tissue: tissue runs shorter than this are removed (floating fragments, specks)
min_tissue must stay below the shortest real wall chord (about the thinnest
wall thickness in px), or real septa get removed.
"""
import numpy as np


def line_positions(n, spacing):
    """Row (or column) indices of test lines across a dimension of size n."""
    return np.arange(spacing // 2, n, spacing)


def _runs(p):
    """Run-length encode a 1D bool array -> (starts, lengths, values)."""
    n = len(p)
    if n == 0:
        return np.array([], int), np.array([], int), np.array([], bool)
    change = np.flatnonzero(p[1:] != p[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, n]
    return starts, ends - starts, p[starts]


def clean_profile(p, min_tissue=0, min_air=0):
    """Fill short interior air gaps, then drop short tissue runs."""
    p = np.asarray(p, bool).copy()
    if min_air > 0:
        s, l, v = _runs(p)
        for i in range(1, len(s) - 1):  # interior runs only: both neighbours are tissue
            if not v[i] and l[i] < min_air:
                p[s[i]:s[i] + l[i]] = True
    if min_tissue > 0:
        s, l, v = _runs(p)
        for i in range(len(s)):
            if v[i] and l[i] < min_tissue:
                p[s[i]:s[i] + l[i]] = False
    return p


def _measure_profiles(profiles, valids, min_tissue, min_air):
    L = 0
    T = 0
    for prof, val in zip(profiles, valids):
        s, l, v = _runs(val)
        for start, length, ok in zip(s, l, v):
            if not ok:
                continue
            seg = clean_profile(prof[start:start + length], min_tissue, min_air)
            L += length
            T += int(np.count_nonzero(seg[1:] != seg[:-1]))
    return L, T


def measure_lm(mask, spacing=8, valid=None, min_tissue=0, min_air=0, directions="hv"):
    """Measure Lm (px) on a boolean tissue mask with horizontal and/or vertical lines.

    valid: optional boolean mask; lines are only counted inside it (use it to
    exclude vessels, airways, pleura, background). Transitions at the edge of
    the valid region are not counted.
    Returns a dict with L, T, intercepts, Lm and per-direction values.
    """
    mask = np.asarray(mask, bool)
    if valid is None:
        valid = np.ones_like(mask)
    valid = np.asarray(valid, bool)
    H, W = mask.shape
    out = {}
    L_all = T_all = 0
    if "h" in directions:
        rows = line_positions(H, spacing)
        L, T = _measure_profiles(mask[rows], valid[rows], min_tissue, min_air)
        out["h"] = _summary(L, T)
        L_all += L
        T_all += T
    if "v" in directions:
        cols = line_positions(W, spacing)
        L, T = _measure_profiles(mask[:, cols].T, valid[:, cols].T, min_tissue, min_air)
        out["v"] = _summary(L, T)
        L_all += L
        T_all += T
    out.update(_summary(L_all, T_all))
    return out


def _summary(L, T):
    I = T / 2
    return {"L": L, "T": T, "intercepts": I, "Lm": (L / I) if I > 0 else float("inf")}
