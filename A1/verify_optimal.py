"""
Independent brute-force check of optimality on the SAVED benchmark problems.
Does NOT use Problem.successors / truck_details: cost is recomputed from first
principles, so it also catches bugs in the shared model, not just in the search.

  python verify_optimal.py            # n in 4..10 (all saved instances)
  python verify_optimal.py 12         # up to n=12 (slow)
"""
import sys, json, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent


def group_cost(group, w1_unused=None):
    """group: list of (dest, arrival) in LOAD order (last = front). Returns sum of delivery times."""
    dispatch = max(a for _, a in group)
    M = max(d for d, _ in group)
    total, max_above = 0, 0
    for d, a in reversed(group):                    # front -> back
        travel = d if d >= max_above else 2 * M - d  # blocked => return leg
        max_above = max(max_above, d)
        total += (dispatch - a) + travel
    return total


def brute_force_optimum(pkgs, cap, w1, w2):
    """Enumerate every way to split packages (in arrival order) into trucks of size <= cap."""
    n = len(pkgs)
    best = float("inf")
    groups = []

    def rec(i):
        nonlocal best
        if i == n:
            t = sum(group_cost([pkgs[j] for j in g]) for g in groups)
            best = min(best, w1 * len(groups) + (w2 / n) * t)
            return
        for g in groups:
            if len(g) < cap:
                g.append(i); rec(i + 1); g.pop()
        groups.append([i]); rec(i + 1); groups.pop()

    rec(0)
    return best


def main(max_n=10):
    df = pd.read_csv(ROOT / "results" / "benchmark_results.csv")
    ref = df[df.Algorithm == "A* (h_combined)"].set_index("Instance")["Cost"]
    checked = bad = 0
    t0 = time.time()
    for f in sorted((ROOT / "problems" / "benchmark").glob("*.json")):
        d = json.loads(f.read_text())
        n = d["num_packages"]
        if n > max_n:
            continue
        pkgs = [(p["destination"], p["arrival_time"]) for p in d["packages"]]
        key = f"{n}-{d['capacity']}-{f.stem.split('_s')[-1]}"
        bf = brute_force_optimum(pkgs, d["capacity"], d["w1"], d["w2"])
        ok = abs(bf - ref[key]) < 1e-9
        checked += 1; bad += (not ok)
        if not ok:
            print("MISMATCH", f.name, bf, ref[key])
    print(f"Brute force vs A*(h_combined) on {checked} saved benchmark problems (n<={max_n}): "
          f"{bad} mismatches  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
