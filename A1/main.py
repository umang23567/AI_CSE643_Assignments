"""
FILE 3/3: entry point.

  python main.py                          -> full experiment run
  python main.py --solve problems/benchmark/n08_cap3_s0.json
                                          -> re-solve ONE saved problem with every algorithm

Outputs (created next to this file):
  problems/   one JSON file per randomly generated problem
              (benchmark/, weight_sweep/, variations/)
  results/    report.txt, benchmark_results.csv, benchmark_summary.csv,
              weight_sweep.csv, variations.csv,
              benchmark_plots.png
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from experiments import (load_problem, make_plots, run_benchmark, run_variations,
                         run_weight_sweep, self_check, summarize_benchmark)
from truck_search import Problem, get_algorithms, solution_stats

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
PROBLEMS_DIR = ROOT / "problems"


class Tee:
    """Prints to the terminal AND collects everything for results/report.txt."""
    def __init__(self, stream):
        self.stream, self.buf = stream, []

    def write(self, s):
        self.stream.write(s)
        self.buf.append(s)

    def flush(self):
        self.stream.flush()


def main(seeds_per_cell=10, w1=10.0, w2=1.5, max_nodes=30000, seed=42):
    RESULTS_DIR.mkdir(exist_ok=True)
    PROBLEMS_DIR.mkdir(exist_ok=True)
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        self_check()

        # --- main benchmark -------------------------------------------------
        df = run_benchmark([4, 6, 8, 10, 12], [2, 3, 4], seeds_per_cell, w1, w2,
                           max_nodes, seed, PROBLEMS_DIR)
        t = summarize_benchmark(df)
        print("\nSummary (Avg_Expanded counts failed runs at the node cap, i.e. a LOWER bound for them;"
              " Gap only over solved runs):")
        print(t["summary"].round(2).to_string(index=False))
        print("\nSuccess rate by problem size:")
        print(t["by_size"].round(2).to_string())
        print(f"\nNodes expanded on the {t['n_common']} instances solved by every optimal method:")
        print(t["common"].round(1).to_string())

        # --- weight sweep & variations ---------------------------------------
        sweep = run_weight_sweep(w2=1.0, ratios=[0.05, 0.1, 0.2, 0.4, 0.7, 1, 2, 5], n=8, cap=3,
                                 seeds=10, max_nodes=max_nodes, base_seed=seed, problems_dir=PROBLEMS_DIR)
        print("\nWeight sweep (optimal plans, n=8, cap=3):")
        print(sweep.round(2).to_string(index=False))

        var = run_variations(n=8, cap=3, seeds=10, w1=w1, w2=w2, max_nodes=max_nodes,
                             base_seed=seed, problems_dir=PROBLEMS_DIR)
        print("\nModel variations (optimal plans, n=8, cap=3; means over 10 instances):")
        print(var.round(2).to_string(index=False))

        # --- save everything --------------------------------------------------
        df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
        t["summary"].round(3).to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)
        sweep.round(3).to_csv(RESULTS_DIR / "weight_sweep.csv", index=False)
        var.round(3).to_csv(RESULTS_DIR / "variations.csv", index=False)
        make_plots(df, sweep, RESULTS_DIR / "benchmark_plots.png")
        print(f"\nResults  -> {RESULTS_DIR}\nProblems -> {PROBLEMS_DIR}")
    finally:
        sys.stdout = tee.stream
        (RESULTS_DIR / "report.txt").write_text("".join(tee.buf))
    return df


def solve_file(path, max_nodes=30000):
    """Re-run every algorithm on one saved problem file."""
    p, meta = load_problem(path)
    print(f"{meta['name']}: n={p.n}, capacity={p.cap}, w1={p.w1}, w2={p.w2}, "
          f"max_trucks={p.max_trucks}, reorder={p.reorder}")
    print("Packages (id, dest, arrival):", [(q.id, q.destination, q.arrival_time) for q in p.pkgs])
    rows = []
    for name, fn in get_algorithms(max_nodes).items():
        pr = Problem(p.pkgs, p.cap, p.w1, p.w2, p.max_trucks, p.reorder)
        r = fn(pr)
        if r.goal:
            trucks, avg_t = solution_stats(pr, r.goal)
            rows.append((name, r.expanded, round(r.goal.g, 2), trucks, round(avg_t, 2)))
        else:
            rows.append((name, r.expanded, None, None, None))
    print(pd.DataFrame(rows, columns=["Algorithm", "Expanded", "Cost", "Trucks", "AvgDelivery"])
          .to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", metavar="PROBLEM_JSON", help="solve one saved problem and exit")
    args = ap.parse_args()
    solve_file(args.solve) if args.solve else main()
