"""
ARC-AGI-1 solver  -  FILE 2/3: program search over the DSL (classical AI, no learning).

PROBLEM FORMULATION  (state-space search, exactly like assignments 1 and 2)
--------------------------------------------------------------------------
State    the grids obtained by running ONE candidate program on every demonstration input
         AND on the test input(s):   (g_1, ..., g_n, test_1, ...).
         Carrying the test inputs along means two programs that agree on the demos but differ
         on the test are NOT merged by duplicate detection.
Initial  the untouched inputs (the empty program).
Action   apply one DSL primitive to every grid of the state  (about 90-110 actions per task).
Goal     the first n grids equal the n demonstration OUTPUTS, either exactly or after one
         consistent colour re-mapping (the "final colour map", learned from the demos).
Path cost  1 per primitive, so a shortest solution is the SIMPLEST program (Occam's razor,
         which is also our only defence against programs that fit the demos by accident).
There is no stated goal to discover (unlike ARC-AGI-3): the demos define it.

SEARCH STRATEGIES (same state space, different frontier order)
    bfs     breadth first    complete, returns shortest program, slow: ~b^d states
    greedy  best first on h  fast when h is informative, no optimality guarantee
    astar   f = g + LAMBDA*h like A*, but h (pixel mismatch) is NOT admissible, so no guarantee

DUPLICATE DETECTION: a state is a tuple of grids; if two different programs produce the same
tuple, only the first is kept (this is what makes depth 3 affordable: many primitives are
no-ops or undo each other).

PREDICTION: ARC allows two attempts per test input, so we return up to 2 DIFFERENT predictions
coming from programs that solve all demos.
"""
import heapq
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from dsl import BG, build_primitives, valid

LAMBDA = 3.0          # weight of the heuristic in the A*-style search


# =====================================================================================
# state, goal test, heuristic
# =====================================================================================
def key_of(grids):
    return tuple((g.shape, g.tobytes()) for g in grids)


def colormap_fit(grids, outputs):
    """Is there ONE mapping colour -> colour that turns every grid into its target output?
    Returns the mapping (identity entries omitted) or None.  Exact equality = empty mapping."""
    mapping = {}
    for g, o in zip(grids, outputs):
        if g.shape != o.shape:
            return None
        for a, b in zip(g.ravel().tolist(), o.ravel().tolist()):
            if mapping.setdefault(a, b) != b:
                return None
    return {a: b for a, b in mapping.items() if a != b}


def apply_map(g, mapping):
    out = g.copy()
    for a, b in mapping.items():
        out[g == a] = b
    return out


def heuristic(grids, outputs):
    """Average 'distance' between the current grids and the demo outputs, in [0, ~2].
    Same shape: fraction of wrong cells (colour-blind version is NOT used, so a pending colour
    map still costs something).  Different shape: 1 plus the relative size difference."""
    tot = 0.0
    for g, o in zip(grids, outputs):
        if g.shape == o.shape:
            tot += float(np.mean(g != o))
        else:
            tot += 1.0 + (abs(g.shape[0] - o.shape[0]) + abs(g.shape[1] - o.shape[1])) / (o.shape[0] + o.shape[1])
    return tot / len(outputs)


# =====================================================================================
# result container
# =====================================================================================
@dataclass
class Solution:
    program: List[str]
    mapping: dict
    predictions: List[np.ndarray]          # one per test input


@dataclass
class SearchResult:
    solutions: List[Solution] = field(default_factory=list)     # programs that fit ALL demos
    expanded: int = 0
    generated: int = 0
    seconds: float = 0.0
    status: str = ""                       # solved / exhausted / timeout / node-limit

    def attempts(self, test_index=0):
        """Up to two DISTINCT predictions for a test input (ARC allows 2 attempts)."""
        out = []
        for s in self.solutions:
            p = s.predictions[test_index]
            if not any(p.shape == q.shape and np.array_equal(p, q) for q in out):
                out.append(p)
            if len(out) == 2:
                break
        return out


# =====================================================================================
# the search
# =====================================================================================
_MISSING = object()


def solve(task, strategy="bfs", max_depth=3, max_nodes=10 ** 9, time_limit=10.0, second_attempt_window=1.0):
    """task = {'train': [{'input','output'}, ...], 'test': [{'input', ...}, ...]} with list-of-list grids.

    Engineering notes (all of them only make the SAME search faster, none changes its result):
      * the goal test is done when a successor is GENERATED, and states at the maximum depth are
        tested but never stored (no one will expand them);
      * a successor is first checked on the first demo only (shape, then colour map); the other grids
        are computed only if that cheap check passes;
      * primitive applications are memoised on (primitive, grid contents).
    """
    t0 = time.perf_counter()
    train_in = [np.array(p["input"], dtype=np.int8) for p in task["train"]]
    train_out = [np.array(p["output"], dtype=np.int8) for p in task["train"]]
    test_in = [np.array(p["input"], dtype=np.int8) for p in task["test"]]
    n = len(train_in)
    out0 = train_out[0]

    in_colors = set(int(v) for g in train_in + test_in for v in np.unique(g))
    out_colors = set(int(v) for g in train_out for v in np.unique(g))
    prims = build_primitives(in_colors, out_colors)

    res = SearchResult()
    cache = {}

    def gkey(g):
        return (g.shape, g.tobytes())

    def run(pi, g, gk):
        k = (pi, gk)
        r = cache.get(k, _MISSING)
        if r is _MISSING:
            try:
                r = prims[pi][1](g)
            except Exception:                              # primitive not applicable to this grid
                r = None
            if not valid(r):
                r = None
            if len(cache) < 400000:
                cache[k] = r
        return r

    def record(prog, grids, mapping):
        res.solutions.append(Solution(prog, mapping, [apply_map(g, mapping) for g in grids[n:]]))

    start = tuple(train_in + test_in)
    start_keys = tuple(gkey(g) for g in start)
    visited = {start_keys}

    m0 = colormap_fit(start[:n], train_out)
    if m0:                                                  # pure recolouring task: empty program + colour map
        record([], start, m0)

    def priority(grids, depth):
        if strategy == "bfs":
            return depth
        h = heuristic(grids[:n], train_out)
        return h if strategy == "greedy" else depth + LAMBDA * h

    counter = 0
    q = deque()
    heap = []
    if strategy == "bfs":
        q.append((start, start_keys, []))
    else:
        heapq.heappush(heap, (priority(start, 0), counter, start, start_keys, []))

    first_found_t = None
    stop = False
    while (q or heap) and not stop:
        if strategy == "bfs":
            grids, keys, prog = q.popleft()
        else:
            _, _, grids, keys, prog = heapq.heappop(heap)
        res.expanded += 1
        last_level = len(prog) + 1 >= max_depth

        for pi, (name, _) in enumerate(prims):
            r0 = run(pi, grids[0], keys[0])
            if r0 is None:
                continue
            # cheap check on the first demo: does it already look like a solution?
            looks_ok = r0.shape == out0.shape and colormap_fit([r0], [out0]) is not None
            if last_level and not looks_ok:
                continue
            new = [r0]
            ok = True
            for i in range(1, len(grids)):
                r = run(pi, grids[i], keys[i])
                if r is None:
                    ok = False
                    break
                new.append(r)
            if not ok:
                continue
            nkeys = tuple([gkey(r0)] + [gkey(r) for r in new[1:]])
            if nkeys in visited:
                continue
            res.generated += 1
            nprog = prog + [name]
            if looks_ok:
                mapping = colormap_fit(new[:n], train_out)
                if mapping is not None:
                    record(nprog, new, mapping)
                    if first_found_t is None:
                        first_found_t = time.perf_counter()
                    if len(res.attempts()) >= 2:
                        stop = True
                        break
            if not last_level:
                visited.add(nkeys)
                counter += 1
                new = tuple(new)
                if strategy == "bfs":
                    q.append((new, nkeys, nprog))
                else:
                    heapq.heappush(heap, (priority(new, len(nprog)), counter, new, nkeys, nprog))
            else:
                visited.add(nkeys)

        now = time.perf_counter()
        if first_found_t is not None and now - first_found_t > min(second_attempt_window, 0.5 * (first_found_t - t0) + 0.2):
            break
        if res.expanded >= max_nodes:
            res.status = res.status or "node-limit"
            break
        if now - t0 > time_limit:
            res.status = res.status or "timeout"
            break

    if res.solutions:
        res.status = "solved"
    elif not res.status:
        res.status = "exhausted"
    res.seconds = time.perf_counter() - t0
    return res
