"""
FILE 2/3: random problem generation (+ saving/loading each problem as a JSON
file), self-check, benchmark, weight sweep, model variations and plots.
"""
import json
import math
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from truck_search import (OPTIMAL_ALGOS, Package, Problem, get_algorithms,
                          h_combined, solution_stats, solve_best_first)


# =====================================================================
# 1. PROBLEM GENERATION + SAVE / LOAD (one JSON file per problem)
# =====================================================================

def generate_random_packages(n: int, max_dest: int, rng: random.Random,
                             dist: str = "uniform") -> List[Package]:
    """Random destinations and arrival times; ids are assigned in arrival order."""
    arrivals = sorted(rng.randint(0, max(1, n // 2)) for _ in range(n))
    pkgs = []
    for i, a in enumerate(arrivals, start=1):
        if dist == "uniform":
            d = rng.randint(1, max_dest)
        elif dist == "near":       # most packages go to nearby stops
            d = min(max_dest, 1 + int(rng.expovariate(0.8)))
        else:
            raise ValueError(dist)
        pkgs.append(Package(i, d, a))
    return pkgs


def save_problem(folder: Path, name: str, pkgs: List[Package], capacity: int,
                 w1: float, w2: float, max_dest: int, dist: str = "uniform",
                 max_trucks: Optional[int] = None, reorder: bool = False,
                 note: str = "") -> Path:
    """Writes one problem instance to <folder>/<name>.json (full config, re-runnable)."""
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name, "note": note,
        "num_packages": len(pkgs), "capacity": capacity,
        "w1": w1, "w2": w2,
        "max_destination": max_dest, "destination_distribution": dist,
        "max_trucks": max_trucks, "reorder": reorder,
        "packages": [{"id": p.id, "destination": p.destination, "arrival_time": p.arrival_time}
                     for p in pkgs],
    }
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def load_problem(path) -> Tuple[Problem, dict]:
    """Rebuilds the exact Problem (incl. fleet limit / reorder flag) from a saved file."""
    data = json.loads(Path(path).read_text())
    pkgs = [Package(d["id"], d["destination"], d["arrival_time"]) for d in data["packages"]]
    p = Problem(pkgs, data["capacity"], data["w1"], data["w2"],
                max_trucks=data["max_trucks"], reorder=data["reorder"])
    return p, data


# =====================================================================
# 2. SELF-CHECK (correctness of g and optimality of UCS / A*)
# =====================================================================

def simulate_deliveries(dests_front_to_back: List[int]) -> List[int]:
    """Independent check of the delivery-time rule: simulate the truck one time unit
    at a time. It moves one stop per tick towards the destination of the package
    currently at the front, and unloads it when it gets there."""
    stack, pos, t, out = list(dests_front_to_back), 0, 0, []
    while stack:
        if pos == stack[0]:
            out.append(t)
            stack.pop(0)
        else:
            pos += 1 if stack[0] > pos else -1
            t += 1
    return out


def self_check(num_instances: int = 60, seed: int = 0) -> None:
    """
    Verifies on tiny instances that
      (1) incremental g == full recomputed cost,
      (2) UCS and all A* variants match a brute-force optimum (=> h admissible, g correct),
      (3) the delivery-time rule equals a step-by-step unloading simulation,
      (4) every action has a non-negative cost (g never decreases).
    """
    rng = random.Random(seed)
    big = 10 ** 9
    configs = [dict(), dict(reorder=True), dict(tight=True)]
    checked = 0
    for _ in range(num_instances):
        n, cap = rng.randint(2, 6), rng.randint(2, 4)
        pkgs = generate_random_packages(n, rng.randint(2, 6), rng)
        for cfg in configs:
            kw = {}
            if cfg.get("reorder"):
                kw["reorder"] = True
            if cfg.get("tight"):
                kw["max_trucks"] = math.ceil(n / cap)
            p = Problem(pkgs, cap, 10.0, 1.5, **kw)

            best = float("inf")
            stack = [p.initial()]
            while stack:                                  # exhaustive enumeration
                s = stack.pop()
                if p.is_goal(s):
                    assert abs(s.g - p.full_cost(s.trucks)) < 1e-9, "g != recomputed cost"
                    for t in s.trucks:                    # (3) independent delivery-time check
                        front_first = [q.destination for q in reversed(p.truck_loaded(t))]
                        _, times = p.truck_details(t)
                        disp = max(q.arrival_time for q in p.truck_loaded(t))
                        mine = [times[q.id] - (disp - q.arrival_time) for q in reversed(p.truck_loaded(t))]
                        assert mine == simulate_deliveries(front_first), "delivery-time rule wrong"
                    best = min(best, s.g)
                else:
                    kids = p.successors(s)
                    assert all(c.g >= s.g - 1e-12 for c in kids), "negative step cost"   # (4)
                    stack.extend(kids)
            for name in OPTIMAL_ALGOS:
                r = get_algorithms(big)[name](p)
                assert abs(r.goal.g - best) < 1e-9, f"{name} not optimal ({r.goal.g} vs {best})"
            checked += 1
    print(f"Self-check passed on {checked} tiny instances: g exact, delivery-time rule matches simulation, step costs >= 0, UCS/A* optimal vs brute force.")


# =====================================================================
# 3. EXPERIMENTS
# =====================================================================

def run_benchmark(scales, capacities, seeds_per_cell, w1, w2, max_nodes, base_seed,
                  problems_dir: Path) -> pd.DataFrame:
    algos = get_algorithms(max_nodes)
    rows = []
    total = len(scales) * len(capacities) * seeds_per_cell
    with tqdm(total=total, desc="Instances", unit="inst") as bar:
        for n in scales:
            for cap in capacities:
                for sd in range(seeds_per_cell):
                    rng = random.Random(f"{base_seed}-{n}-{cap}-{sd}")
                    max_dest = rng.randint(3, 8)
                    pkgs = generate_random_packages(n, max_dest, rng)
                    inst = f"{n}-{cap}-{sd}"
                    save_problem(problems_dir / "benchmark", f"n{n:02d}_cap{cap}_s{sd}",
                                 pkgs, cap, w1, w2, max_dest)
                    for name, fn in algos.items():
                        p = Problem(pkgs, cap, w1, w2)       # fresh cache for fair timing
                        t0 = time.perf_counter()
                        res = fn(p)
                        ms = (time.perf_counter() - t0) * 1000
                        ok = res.goal is not None
                        trucks, avg_t = solution_stats(p, res.goal) if ok else (None, None)
                        rows.append({
                            "Instance": inst, "Packages": n, "Capacity": cap, "Algorithm": name,
                            "Expanded": res.expanded, "Generated": res.generated,
                            "Runtime (ms)": ms, "Success": ok,
                            "Cost": res.goal.g if ok else float("nan"),
                            "Trucks": trucks, "AvgDelivery": avg_t,
                        })
                    bar.update(1)
    df = pd.DataFrame(rows)

    # Optimal cost per instance = value found by any optimal algorithm that finished.
    opt_rows = df[df["Algorithm"].isin(OPTIMAL_ALGOS) & df["Success"]]
    spread = opt_rows.groupby("Instance")["Cost"].agg(["min", "max"])
    bad = (spread["max"] - spread["min"] > 1e-6).sum()
    print(f"\nSanity: optimal algorithms disagree on {bad} instances (should be 0).")
    best_idx = opt_rows.groupby("Instance")["Cost"].idxmin()        # cheapest optimal plan per instance
    ref = opt_rows.loc[best_idx].set_index("Instance")[["Cost", "AvgDelivery"]]
    ref.columns = ["Optimal", "Opt_AvgDelivery"]
    df = df.join(ref, on="Instance")
    # Two gaps, both relative to the optimal plan, and their average:
    df["Gap %"] = (df["Cost"] - df["Optimal"]) / df["Optimal"] * 100                     # total objective
    df["Delay Gap %"] = (df["AvgDelivery"] - df["Opt_AvgDelivery"]) / df["Opt_AvgDelivery"] * 100
    # df["Avg Gap %"] = (df["Gap %"] + df["Delay Gap %"]) / 2
    return df


def summarize_benchmark(df: pd.DataFrame) -> dict:
    """Builds the summary tables printed in the report and saved as CSVs."""
    summary = df.groupby("Algorithm", sort=False).agg(
        Solved=("Success", "mean"),
        Avg_Expanded=("Expanded", "mean"),
        Median_Expanded=("Expanded", "median"),
        Avg_Runtime_ms=("Runtime (ms)", "mean"),
        Cost_Gap_pct=("Gap %", "mean"),
        Delay_Gap_pct=("Delay Gap %", "mean"),
        # Avg_of_Gaps_pct=("Avg Gap %", "mean"),
        Max_Cost_Gap_pct=("Gap %", "max"),
        Pct_Optimal=("Gap %", lambda s: 100 * (s.dropna() < 1e-6).mean() if s.notna().any() else float("nan")),
    ).reset_index()

    by_size = df.pivot_table(index="Algorithm", columns="Packages", values="Success",
                             aggfunc="mean", sort=False)

    # Efficiency comparison fair to everyone: instances where ALL optimal algorithms finished.
    done = df[df["Algorithm"].isin(OPTIMAL_ALGOS)].groupby("Instance")["Success"].all()
    common = df[df["Instance"].isin(done[done].index) & df["Algorithm"].isin(OPTIMAL_ALGOS)]
    common_tbl = common.groupby("Algorithm", sort=False)["Expanded"].agg(["mean", "median", "max"])
    return {"summary": summary, "by_size": by_size, "common": common_tbl, "n_common": int(done.sum())}


def run_weight_sweep(w2, ratios, n, cap, seeds, max_nodes, base_seed, problems_dir: Path) -> pd.DataFrame:
    """How the optimal plan changes as trucks get more/less expensive (w1 vs w2)."""
    rows = []
    for w1 in ratios:
        for sd in range(seeds):
            rng = random.Random(f"sweep-{base_seed}-{sd}")
            max_dest = rng.randint(3, 8)
            pkgs = generate_random_packages(n, max_dest, rng)
            # same packages are reused for every w1 -> save once, with w1 stated per file
            save_problem(problems_dir / "weight_sweep", f"n{n:02d}_cap{cap}_s{sd}_w1_{w1}",
                         pkgs, cap, w1, w2, max_dest)
            p = Problem(pkgs, cap, w1, w2)
            res = solve_best_first(p, h_combined, max_nodes)
            if res.goal:
                trucks, avg_t = solution_stats(p, res.goal)
                rows.append({"w1/w2": w1 / w2, "Trucks": trucks, "AvgDelivery": avg_t})
    return pd.DataFrame(rows).groupby("w1/w2").mean().reset_index()


VARIANTS = {   # key: (label, config)
    "baseline":   ("Baseline (arrival-order stacking)", dict()),
    "reorder":    ("Buffer: load far->near (reorder)", dict(reorder=True)),
    "tight_fleet": ("Tight fleet (max trucks = ceil(n/cap))", dict(tight=True)),
    "skewed_near": ("Skewed destinations (mostly near)", dict(dist="near")),
}


def run_variations(n, cap, seeds, w1, w2, max_nodes, base_seed, problems_dir: Path) -> pd.DataFrame:
    """Real-life variations, each solved optimally with A*."""
    rows = []
    for key, (vname, cfg) in VARIANTS.items():
        for sd in range(seeds):
            rng = random.Random(f"var-{base_seed}-{sd}")
            dist = cfg.get("dist", "uniform")
            max_dest = rng.randint(3, 8)
            pkgs = generate_random_packages(n, max_dest, rng, dist=dist)
            kw = {}
            if cfg.get("reorder"):
                kw["reorder"] = True
            if cfg.get("tight"):
                kw["max_trucks"] = math.ceil(n / cap)
            save_problem(problems_dir / "variations", f"{key}_n{n:02d}_cap{cap}_s{sd}",
                         pkgs, cap, w1, w2, max_dest, dist=dist, note=vname, **kw)
            p = Problem(pkgs, cap, w1, w2, **kw)
            res = solve_best_first(p, h_combined, max_nodes)
            if res.goal:
                trucks, avg_t = solution_stats(p, res.goal)
                rows.append({"Variation": vname, "Cost": res.goal.g, "Trucks": trucks,
                             "AvgDelivery": avg_t, "Expanded": res.expanded})
    return pd.DataFrame(rows).groupby("Variation", sort=False).mean().reset_index()


# =====================================================================
# 4. PLOTS
# =====================================================================

def make_plots(df: pd.DataFrame, sweep: pd.DataFrame, output_file: Path) -> None:
    fig, ax = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle("Truck Loading & Delivery Planning: Search Benchmark", fontsize=15, fontweight="bold")
    algos = list(df["Algorithm"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {a: cmap(i) for i, a in enumerate(algos)}

    def line(axis, col, title, ylabel, log=True, data=df):
        g = data.groupby(["Packages", "Algorithm"])[col].mean().unstack()
        for a in algos:
            if a in g:
                axis.plot(g.index, g[a], marker="o", label=a, color=colors[a])
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Package count")
        axis.set_ylabel(ylabel)
        if log:
            axis.set_yscale("log")
        axis.grid(True, linestyle="--", alpha=0.5)

    line(ax[0, 0], "Expanded", "1. Search effort (avg nodes expanded; failed runs count at the cap)",
         "Nodes expanded [log]")
    ax[0, 0].legend(fontsize=8)
    line(ax[0, 1], "Runtime (ms)", "2. Runtime (avg ms)", "ms [log]")

    df2 = df.assign(S=df["Success"].astype(float))
    line(ax[0, 2], "S", "3. Success rate (finished within node cap)", "Fraction solved", log=False, data=df2)
    ax[0, 2].set_ylim(-0.05, 1.05)

    ok = df[df["Success"] & df["Gap %"].notna()]
    gaps = ok.groupby("Algorithm")[["Gap %", "Delay Gap %"]].mean().reindex(algos)
    # gaps = ok.groupby("Algorithm")[["Gap %", "Delay Gap %", "Avg Gap %"]].mean().reindex(algos)
    
    x = list(range(len(algos)))
    w = 0.35
    series = [("Gap %", "Total-cost gap", "tab:blue"),
              ("Delay Gap %", "Delivery-time gap", "tab:red"),
            #   ("Avg Gap %", "Average of the two", "tab:green")
              ]
    for k, (col, label, colr) in enumerate(series):
        vals = gaps[col].values
        bars = ax[1, 0].bar([i + (k - 1) * w for i in x], vals, w, label=label, color=colr)
        for b_, v in zip(bars, vals):            # value labels, so 0% bars don't look empty
            ax[1, 0].annotate(f"{v:.1f}", (b_.get_x() + b_.get_width() / 2, v), ha="center",
                              va="bottom", xytext=(0, 2), textcoords="offset points", fontsize=7)
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(algos, rotation=35, ha="right", fontsize=8)
    ax[1, 0].set_ylim(0, max(gaps.values.max() * 1.15, 1))
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("4. Solution quality: mean gap to optimal (solved runs)", fontweight="bold")
    ax[1, 0].set_ylabel("Gap to optimal (%)")
    ax[1, 0].grid(True, axis="y", linestyle="--", alpha=0.5)

    for a in algos:
        g = df[df["Algorithm"] == a]
        ax[1, 1].scatter(g["Expanded"], g["Runtime (ms)"], label=a, alpha=0.6, s=25, color=colors[a])
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("5. Nodes vs runtime", fontweight="bold")
    ax[1, 1].set_xlabel("Nodes expanded")
    ax[1, 1].set_ylabel("Runtime (ms)")
    ax[1, 1].grid(True, linestyle="--", alpha=0.5)

    a1 = ax[1, 2]
    a1.plot(sweep["w1/w2"], sweep["Trucks"], "o-", color="tab:blue", label="Avg #trucks")
    a1.set_xscale("log")
    a1.set_xlabel("w1 / w2  (truck cost relative to delay cost)")
    a1.set_ylabel("Avg #trucks in optimal plan", color="tab:blue")
    a2 = a1.twinx()
    a2.plot(sweep["w1/w2"], sweep["AvgDelivery"], "s--", color="tab:red", label="Avg delivery time")
    a2.set_ylabel("Avg delivery time in optimal plan", color="tab:red")
    a1.set_title("6. Trade-off: trucks vs delivery time (optimal plans)", fontweight="bold")
    a1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=200)
    plt.close(fig)
    print(f"Plot saved to '{output_file}'")
