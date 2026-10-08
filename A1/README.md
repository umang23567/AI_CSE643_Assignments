# Truck Loading & Delivery Planning (CSE643 Assignment 1)

State-space search for loading packages into fixed-capacity trucks so as to minimise
`cost = w1 * (#trucks) + w2 * (average delivery time)`.

## Files
| File | Purpose |
|---|---|
| `truck_search.py` | Model, state/successors, 3 heuristics, BFS, DFS, UCS, A*, Weighted A*, Greedy |
| `experiments.py` | Random problem generation, save/load, self-check, benchmark, weight sweep, variations, plots |
| `main.py` | Entry point |

`problems/` and `results/` are created automatically next to `main.py`.

## Run
```
pip install pandas matplotlib tqdm
python main.py                                             # full experiment (~1-2 min)
python main.py --solve problems/benchmark/n08_cap3_s0.json # re-solve one saved problem
```

## Output
- `problems/` : one JSON file per random problem (`benchmark/` 150, `weight_sweep/` 80, `variations/` 40)
- `results/`  : `report.txt` (full console output), `benchmark_results.csv`, `benchmark_summary.csv`,
  `weight_sweep.csv`, `variations.csv`, `benchmark_plots.png`

Everything is seeded (`seed=42`), so nodes expanded, costs and gaps are reproducible.
Only runtime in ms varies slightly between runs.

## Model assumptions
1. Packages arrive in order and are loaded immediately; the last loaded is at the front (LIFO).
2. All trucks have the same capacity; the fleet is unlimited unless `max_trucks` is set.
3. A truck leaves when its last package is loaded, so earlier packages wait.
4. Stop k is k time units from the centre. Packages are unloaded from the FRONT only, so the
   unload order is forced; the truck drives to each package's stop in that order, and the
   travel time of the i-th package is the total distance driven so far
   (`|d1-0| + |d2-d1| + ... + |di-d(i-1)|`).
5. Delivery time = (dispatch - arrival) + travel time.

## Search formulation
- **State:** (index of next package, tuple of trucks)
- **Actions:** put the next package in a non-full truck, or open a new truck
- **Goal:** all packages assigned
- **g(n):** exact cost so far (increments are non-negative)
- **h(n):** admissible lower bounds `h_time`, `h_trucks`, `h_combined` (their sum)

## Key findings (150 random instances, 4-12 packages)
Solution quality is measured primarily by the percentage gap between an algorithm's weighted objective and the optimal weighted objective. The delivery-time gap is also reported separately to show the effect on the delivery-time component of the objective.

| Algorithm | Solved | Avg nodes | Objective gap | Delivery-time gap |
|---|---:|---:|---:|---:|
| BFS | 87% | 5575 | 3.5% | 19.8% |
| DFS | 100% | 9 | 3.7% | 21.1% |
| UCS (h=0) | 100% | 1246 | 0% | 0% |
| A* (h_time) | 100% | 598 | 0% | 0% |
| A* (h_trucks) | 100% | 1075 | 0% | 0% |
| A* (h_combined) | 100% | 308 | 0% | 0% |
| Weighted A* (W=2) | 100% | 94 | 3.5% | 24.2% |
| Greedy | 100% | 9 | 5.4% | 34.2% |

- A* with the combined heuristic is optimal and expands about 4x fewer nodes than UCS.
- DFS and Greedy are near-instant but have larger objective gaps than the optimal methods, with maximum single-instance objective gaps of about 30%.
- Cheap trucks give many trucks and fast delivery; once w1/w2 is about 1 the plan uses the
  minimum number of trucks and delivery slows.
- Reordering at a buffer and skewed (near) destinations lower the average delivery time.

## Verification
`self_check()` runs on 180 tiny instances and checks that (1) the incremental g equals the
recomputed cost, (2) UCS and every A* variant match a brute-force optimum, (3) the delivery-time
rule equals an independent step-by-step unloading simulation, and (4) no action has a negative cost.
