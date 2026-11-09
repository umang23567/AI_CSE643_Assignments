"""
ARC-AGI-1 solver  -  FILE 1/3: the DSL (the hard-coded PRIORS).

A grid is a 2-D numpy array of small ints (colours 0-9).  A *primitive* is a function
grid -> grid (or None when it does not apply).  A *program* is a sequence of primitives.

Every primitive below encodes one of Chollet's "Core Knowledge" priors.  This file IS the
prior knowledge of the system, so it is the part that most needs to be justified:

    OBJECTNESS      connected components, crop / keep the largest or smallest object,
                    remove isolated noise pixels, recolour the largest / smallest object
    GEOMETRY        rotations, flips, transpose, scaling, tiling, mirroring, symmetry completion
    TOPOLOGY        fill enclosed holes, connect two same-coloured cells with a line,
                    gravity (things fall until blocked)
    COUNTING/LOGIC  split a grid in two halves and combine them with AND / OR / XOR ...
    COLOUR          remove / keep / replace a colour, crop to a colour, draw a border

Assumption hard-coded everywhere: the background colour is 0 (black).  Tasks whose
background is another colour are therefore mostly out of reach (see the failure analysis).
"""
import numpy as np

BG = 0
MAX_SIDE = 30


# =====================================================================================
# helpers
# =====================================================================================
def valid(g):
    return (g is not None and g.ndim == 2 and 0 < g.shape[0] <= MAX_SIDE and 0 < g.shape[1] <= MAX_SIDE)


def components(g, same_color=True, diag=True):
    """Connected components of non-background cells (OBJECTNESS prior).
    same_color=True : a component has a single colour;  False : any non-bg cells touch.
    Returns a list of lists of (row, col), ordered by first cell (top-to-bottom, left-to-right)."""
    h, w = g.shape
    seen = np.zeros((h, w), bool)
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)] + ([(-1, -1), (-1, 1), (1, -1), (1, 1)] if diag else [])
    comps = []
    for r in range(h):
        for c in range(w):
            if g[r, c] == BG or seen[r, c]:
                continue
            col = g[r, c]
            stack, cells = [(r, c)], []
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny, nx] and g[ny, nx] != BG \
                            and (not same_color or g[ny, nx] == col):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(cells)
    return comps


def _bbox(cells):
    rs = [y for y, _ in cells]
    cs = [x for _, x in cells]
    return min(rs), max(rs), min(cs), max(cs)


def _pick(g, which):
    """The largest / smallest multicolour object (None when the grid has no object)."""
    comps = components(g, same_color=False)
    if not comps:
        return None
    key = (lambda c: -len(c)) if which == "largest" else (lambda c: len(c))
    return sorted(comps, key=key)[0]          # stable: ties -> first in reading order


# =====================================================================================
# GEOMETRY
# =====================================================================================
def rot90(g): return np.rot90(g, 1)
def rot180(g): return np.rot90(g, 2)
def rot270(g): return np.rot90(g, 3)
def flipud(g): return g[::-1, :]
def fliplr(g): return g[:, ::-1]
def transpose(g): return g.T
def antitranspose(g): return np.rot90(g, 2).T


def up2(g): return np.kron(g, np.ones((2, 2), g.dtype))
def up3(g): return np.kron(g, np.ones((3, 3), g.dtype))
def down2(g): return g[::2, ::2] if g.shape[0] % 2 == 0 and g.shape[1] % 2 == 0 else None
def tile22(g): return np.tile(g, (2, 2))
def tile33(g): return np.tile(g, (3, 3))
def tile_h2(g): return np.tile(g, (1, 2))
def tile_v2(g): return np.tile(g, (2, 1))
def mirror_h(g): return np.hstack([g, g[:, ::-1]])
def mirror_v(g): return np.vstack([g, g[::-1, :]])
def mirror_h_rev(g): return np.hstack([g[:, ::-1], g])
def mirror_v_rev(g): return np.vstack([g[::-1, :], g])
def mirror_4(g):
    top = np.hstack([g, g[:, ::-1]])
    return np.vstack([top, top[::-1, :]])


def sym_lr(g): return np.where(g == BG, g[:, ::-1], g)          # fill bg from the mirror image
def sym_ud(g): return np.where(g == BG, g[::-1, :], g)
def sym_both(g):
    g = sym_lr(g)
    return sym_ud(g)


def roll(axis, k):
    return lambda g: np.roll(g, k, axis=axis)


# =====================================================================================
# OBJECTNESS / CROPPING
# =====================================================================================
def crop_bbox(g):
    ys, xs = np.nonzero(g != BG)
    if len(ys) == 0:
        return None
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def crop_obj(which):
    def f(g):
        o = _pick(g, which)
        if o is None:
            return None
        r0, r1, c0, c1 = _bbox(o)
        return g[r0:r1 + 1, c0:c1 + 1]
    return f


def keep_obj(which):
    def f(g):
        o = _pick(g, which)
        if o is None:
            return None
        out = np.zeros_like(g)
        for y, x in o:
            out[y, x] = g[y, x]
        return out
    return f


def remove_singletons(g):
    out = g.copy()
    for cells in components(g, same_color=True):
        if len(cells) == 1:
            out[cells[0]] = BG
    return out


def recolor_obj(which, color):
    """Recolour the largest / smallest SINGLE-COLOUR object."""
    def f(g):
        comps = components(g, same_color=True)
        if not comps:
            return None
        key = (lambda c: -len(c)) if which == "largest" else (lambda c: len(c))
        target = sorted(comps, key=key)[0]
        out = g.copy()
        for y, x in target:
            out[y, x] = color
        return out
    return f


# =====================================================================================
# COLOUR
# =====================================================================================
def remove_color(c):
    return lambda g: np.where(g == c, BG, g)


def keep_color(c):
    return lambda g: np.where(g == c, g, BG)


def replace_color(a, b):
    return lambda g: np.where(g == a, b, g)


def crop_color(c):
    def f(g):
        ys, xs = np.nonzero(g == c)
        if len(ys) == 0:
            return None
        return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return f


def draw_border(c):
    def f(g):
        out = g.copy()
        out[0, :] = c; out[-1, :] = c; out[:, 0] = c; out[:, -1] = c
        return out
    return f


# =====================================================================================
# TOPOLOGY
# =====================================================================================
def fill_holes(color):
    """Background cells NOT connected (4-neighbourhood) to the border are 'enclosed': paint them."""
    def f(g):
        h, w = g.shape
        reach = np.zeros((h, w), bool)
        stack = [(y, x) for y in range(h) for x in (0, w - 1) if g[y, x] == BG] + \
                [(y, x) for x in range(w) for y in (0, h - 1) if g[y, x] == BG]
        for y, x in stack:
            reach[y, x] = True
        while stack:
            y, x = stack.pop()
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not reach[ny, nx] and g[ny, nx] == BG:
                    reach[ny, nx] = True
                    stack.append((ny, nx))
        out = g.copy()
        out[(g == BG) & ~reach] = color
        return out
    return f


def _connect(g, horizontal, vertical):
    out = g.copy()
    h, w = g.shape
    lines = []
    if horizontal:
        lines += [[(r, c) for c in range(w)] for r in range(h)]
    if vertical:
        lines += [[(r, c) for r in range(h)] for c in range(w)]
    for line in lines:
        last, last_col = None, None
        for k, (r, c) in enumerate(line):
            v = g[r, c]
            if v == BG:
                continue
            if last is not None and last_col == v and k - last > 1:
                for (rr, cc) in line[last + 1:k]:
                    out[rr, cc] = v
            last, last_col = k, v
    return out


def connect_h(g): return _connect(g, True, False)
def connect_v(g): return _connect(g, False, True)
def connect_hv(g): return _connect(g, True, True)


def gravity(direction):
    """TOPOLOGY/physics prior: every non-background cell slides until it is blocked."""
    def f(g):
        out = np.zeros_like(g)
        h, w = g.shape
        if direction in ("down", "up"):
            for c in range(w):
                vals = [v for v in g[:, c] if v != BG]
                if vals:
                    if direction == "down":
                        out[h - len(vals):, c] = vals
                    else:
                        out[:len(vals), c] = vals
        else:
            for r in range(h):
                vals = [v for v in g[r, :] if v != BG]
                if vals:
                    if direction == "right":
                        out[r, w - len(vals):] = vals
                    else:
                        out[r, :len(vals)] = vals
        return out
    return f


def dedup_rows(g):
    keep = [0] + [i for i in range(1, g.shape[0]) if not np.array_equal(g[i], g[i - 1])]
    return g[keep, :]


def dedup_cols(g):
    keep = [0] + [i for i in range(1, g.shape[1]) if not np.array_equal(g[:, i], g[:, i - 1])]
    return g[:, keep]


# =====================================================================================
# COUNTING / LOGIC: split into halves, combine
# =====================================================================================
def _halves(g, axis):
    """Two halves of the grid. An odd size means the middle line is a separator and is dropped."""
    n = g.shape[axis]
    if n < 2:
        return None
    a_sl = slice(0, n // 2)
    b_sl = slice(n - n // 2, n)
    if axis == 0:
        return g[a_sl, :], g[b_sl, :]
    return g[:, a_sl], g[:, b_sl]


def half(axis, first):
    def f(g):
        ab = _halves(g, axis)
        return None if ab is None else ab[0 if first else 1]
    return f


def logic(axis, op):
    """Combine the two halves cell by cell. Output uses colour 1 for 'true' (a final colour
    map turns it into whatever colour the task wants)."""
    def f(g):
        ab = _halves(g, axis)
        if ab is None:
            return None
        a, b = ab[0] != BG, ab[1] != BG
        r = {"and": a & b, "or": a | b, "xor": a ^ b, "nor": ~(a | b), "a_not_b": a & ~b}[op]
        return r.astype(g.dtype)
    return f


# =====================================================================================
# building the action set for ONE task (colour-parameterised primitives need the task's colours)
# =====================================================================================
def build_primitives(in_colors, out_colors):
    """Returns a list of (name, function).  Colour parameters are restricted to the colours that
    actually occur in this task's inputs (to read / remove) or outputs (to paint), which keeps
    the branching factor small."""
    P = [
        ("rot90", rot90), ("rot180", rot180), ("rot270", rot270), ("flipud", flipud),
        ("fliplr", fliplr), ("transpose", transpose), ("antitranspose", antitranspose),
        ("up2", up2), ("up3", up3), ("down2", down2), ("tile2x2", tile22), ("tile3x3", tile33),
        ("tile_h2", tile_h2), ("tile_v2", tile_v2),
        ("mirror_h", mirror_h), ("mirror_v", mirror_v), ("mirror_h_rev", mirror_h_rev),
        ("mirror_v_rev", mirror_v_rev), ("mirror4", mirror_4),
        ("sym_lr", sym_lr), ("sym_ud", sym_ud), ("sym_both", sym_both),
        ("crop_bbox", crop_bbox), ("crop_largest", crop_obj("largest")), ("crop_smallest", crop_obj("smallest")),
        ("keep_largest", keep_obj("largest")), ("keep_smallest", keep_obj("smallest")),
        ("remove_singletons", remove_singletons),
        ("connect_h", connect_h), ("connect_v", connect_v), ("connect_hv", connect_hv),
        ("gravity_down", gravity("down")), ("gravity_up", gravity("up")),
        ("gravity_left", gravity("left")), ("gravity_right", gravity("right")),
        ("dedup_rows", dedup_rows), ("dedup_cols", dedup_cols),
        ("roll_up", roll(0, -1)), ("roll_down", roll(0, 1)), ("roll_left", roll(1, -1)), ("roll_right", roll(1, 1)),
        ("left_half", half(1, True)), ("right_half", half(1, False)),
        ("top_half", half(0, True)), ("bottom_half", half(0, False)),
    ]
    for axis, an in ((1, "lr"), (0, "ud")):
        for op in ("and", "or", "xor", "nor", "a_not_b"):
            P.append((f"{an}_{op}", logic(axis, op)))
    for c in sorted(in_colors):
        if c == BG:
            continue
        P += [(f"remove_color({c})", remove_color(c)), (f"keep_color({c})", keep_color(c)),
              (f"crop_color({c})", crop_color(c))]
    for a in sorted(in_colors):
        for b in sorted(out_colors):
            if a != b:
                P.append((f"replace({a}->{b})", replace_color(a, b)))
    for c in sorted(out_colors):
        if c == BG:
            continue
        P += [(f"fill_holes({c})", fill_holes(c)), (f"border({c})", draw_border(c)),
              (f"recolor_largest({c})", recolor_obj("largest", c)),
              (f"recolor_smallest({c})", recolor_obj("smallest", c))]
    return P
