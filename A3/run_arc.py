"""
ARC-AGI-1 solver  -  FILE 3/3: command line.

    python run_arc.py --suite                       the 8 chosen tasks in ./tasks  (BFS)
    python run_arc.py --suite --compare             same tasks with BFS, greedy and A*
    python run_arc.py --task tasks/<id>.json --show watch one task: demos, program, prediction
    python run_arc.py --dir /path/to/ARC-AGI/data/training --limit 400 --time 4
                                                    batch statistics over a whole directory
Results (CSV) are written to ./results.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from search import solve                                    # noqa: E402

RESULTS = os.path.join(HERE, "results")


def load(path):
    with open(path) as f:
        return json.load(f)


def show_pair(a, b, title_a="", title_b=""):
    """Print two grids side by side."""
    a, b = np.asarray(a), np.asarray(b)
    h = max(a.shape[0], b.shape[0])
    wa = a.shape[1]
    lines = [f"{title_a:<{wa}}    {title_b}"]
    for r in range(h):
        ra = "".join(str(v) for v in a[r]) if r < a.shape[0] else " " * wa
        rb = "".join(str(v) for v in b[r]) if r < b.shape[0] else ""
        lines.append(f"{ra:<{wa}}    {rb}")
    return "\n".join(lines)


def program_text(sol):
    p = " -> ".join(sol.program) if sol.program else "(identity)"
    return p + (f"  + colour map {sol.mapping}" if sol.mapping else "")


def evaluate(task, res):
    """ARC scoring: a task is solved only if EVERY test output is produced exactly
    (up to 2 attempts per test input)."""
    ok = True
    for i, t in enumerate(task["test"]):
        truth = np.array(t["output"])
        att = res.attempts(i)
        if not any(p.shape == truth.shape and np.array_equal(p, truth) for p in att):
            ok = False
    return ok


def classify(task, res):
    if not res.solutions:
        return "no program found (" + res.status + ")"
    return "SOLVED" if evaluate(task, res) else "overfit: fits demos, wrong on test"


def run_one(path, strategy, time_limit, show):
    task = load(path)
    res = solve(task, strategy=strategy, time_limit=time_limit)
    verdict = classify(task, res)
    if show:
        print("=" * 90)
        print(f"{os.path.basename(path)}   strategy={strategy}   demos={len(task['train'])}")
        for i, p in enumerate(task["train"]):
            print(show_pair(p["input"], p["output"], f"demo {i + 1} input", "output"))
            print()
        print(f"expanded={res.expanded}  generated={res.generated}  time={res.seconds:.2f}s  ->  {verdict}")
        for s in res.solutions[:3]:
            print("   program:", program_text(s))
        for i, t in enumerate(task["test"]):
            att = res.attempts(i)
            truth = np.array(t["output"]) if "output" in t else None
            if att:
                print(show_pair(att[0], truth if truth is not None else att[0], f"test {i + 1} prediction", "expected"))
            else:
                print(show_pair(t["input"], truth, f"test {i + 1} input (no prediction)", "expected"))
    return task, res, verdict


def suite(strategies, time_limit, show):
    files = sorted(glob.glob(os.path.join(HERE, "tasks", "*.json")))
    rows = []
    for strat in strategies:
        print(f"\n--- strategy: {strat} ---")
        print(f"{'task':<14}{'result':<40}{'expanded':>9}{'generated':>10}{'sec':>7}   program")
        for f in files:
            task, res, verdict = run_one(f, strat, time_limit, show and strat == strategies[0])
            prog = program_text(res.solutions[0]) if res.solutions else ""
            print(f"{os.path.basename(f)[:-5]:<14}{verdict:<40}{res.expanded:>9}{res.generated:>10}{res.seconds:>7.2f}   {prog}")
            rows.append({"strategy": strat, "task": os.path.basename(f)[:-5], "result": verdict,
                         "expanded": res.expanded, "generated": res.generated, "seconds": round(res.seconds, 3),
                         "program": prog})
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "suite_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def batch(directory, limit, strategy, time_limit, tag):
    files = sorted(glob.glob(os.path.join(directory, "*.json")))[:limit]
    rows, t0 = [], time.time()
    for i, f in enumerate(files, 1):
        task = load(f)
        res = solve(task, strategy=strategy, time_limit=time_limit)
        verdict = classify(task, res)
        plen = len(res.solutions[0].program) if res.solutions else ""
        rows.append({"task": os.path.basename(f)[:-5], "verdict": verdict, "program_len": plen,
                     "expanded": res.expanded, "seconds": round(res.seconds, 2),
                     "program": program_text(res.solutions[0]) if res.solutions else ""})
        if i % 20 == 0:
            solved = sum(r["verdict"] == "SOLVED" for r in rows)
            print(f"  {i}/{len(files)} tasks, solved so far {solved}  ({time.time() - t0:.0f}s)", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"batch_{tag}_{strategy}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    solved = sum(r["verdict"] == "SOLVED" for r in rows)
    over = sum(r["verdict"].startswith("overfit") for r in rows)
    print(f"\n{tag} / {strategy}: {n} tasks | solved {solved} ({100 * solved / n:.1f}%) | "
          f"fit the demos but wrong on test {over} | nothing found {n - solved - over} | "
          f"{time.time() - t0:.0f}s total  -> {out}")
    lens = {}
    for r in rows:
        if r["verdict"] == "SOLVED":
            lens[r["program_len"]] = lens.get(r["program_len"], 0) + 1
    print("solved by program length:", dict(sorted(lens.items())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task")
    ap.add_argument("--suite", action="store_true")
    ap.add_argument("--compare", action="store_true", help="with --suite: bfs vs greedy vs astar")
    ap.add_argument("--dir")
    ap.add_argument("--limit", type=int, default=10 ** 9)
    ap.add_argument("--strategy", default="bfs", choices=["bfs", "greedy", "astar"])
    ap.add_argument("--time", type=float, default=8.0, help="seconds per task")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.suite:
        suite(["bfs", "greedy", "astar"] if a.compare else [a.strategy], a.time, a.show)
    elif a.task:
        run_one(a.task, a.strategy, a.time, True)
    elif a.dir:
        batch(a.dir, a.limit, a.strategy, a.time, os.path.basename(os.path.normpath(a.dir)))
    else:
        ap.print_help()
