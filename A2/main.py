"""
Robo-Football  -  FILE 6/6: entry point.

    python main.py test                      run the automated rule / agent checks
    python main.py demo                      watch one game (board printed after every turn)
    python main.py trace                     write a hand-traceable alpha-beta trace
    python main.py replot                    redraw all figures from the saved CSV files
    python main.py experiments               run ALL experiments (about 20-25 minutes, 1 core)
    python main.py experiments --quick       small version of everything (about 2-3 minutes)
    python main.py experiments --only pruning,greedy,heuristics,depth,mcts

Outputs go to ./results  (tables as CSV, plots as PNG, report.txt, sample game, trace).
"""
import argparse
import os
import sys
import time
import unittest

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
sys.path.insert(0, ROOT)

import experiments as ex                                    # noqa: E402
from agents import build_agent, play_game                   # noqa: E402
from game import Game                                       # noqa: E402


def run_tests():
    suite = unittest.defaultTestLoader.loadTestsFromName("tests")
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    sys.exit(0 if res.wasSuccessful() else 1)


def run_demo():
    g = Game()
    a = build_agent(ex.ab(3, "field"), 1)
    b = build_agent(ex.greedy("structure"), 2)
    print(f"A = {a.name}   B = {b.name}\n")
    r = play_game(g, a, b, seed=0, record=True)
    log = r["log"]
    for i in range(0, len(log), 2):
        print(log[i])
        if i // 2 < 3 or "GOAL" in log[i] or "tackled" in log[i] or "saved" in log[i]:
            print(log[i + 1])
    print(f"\nFINAL SCORE  A {r['score'][0]} - {r['score'][1]} B   after {r['turns']} turns")
    print(g.render(r["final"]))


def run_trace():
    os.makedirs(RESULTS, exist_ok=True)
    s = ex.build_positions()["attack in the zone"]
    text = ("Alpha-beta trace, depth 3, FIELD heuristic, first 3 moves of every node.\n"
            "Position:\n" + ex.GAME.render(s) + "\n\n" + ex.alphabeta_trace(s, 3, "field", branch_limit=3))
    path = os.path.join(RESULTS, "alphabeta_trace.txt")
    with open(path, "w") as f:
        f.write(text)
    print(text[:3500] + ("\n...\n(full trace in %s)" % path if len(text) > 3500 else ""))


def run_experiments(quick, only):
    os.makedirs(RESULTS, exist_ok=True)
    rep = ex.Report()
    n, nm = (10, 8) if quick else (60, 40)
    chosen = set(only.split(",")) if only else {"pruning", "greedy", "heuristics", "depth", "mcts"}
    t0 = time.time()
    rep(f"Robo-Football experiments | max turns {ex.GAME.max_turns} | games per pairing {n} "
        f"(MCTS pairings {nm}) | quick={quick}")
    games, depth_summary = [], None
    ex.sample_game(os.path.join(RESULTS, "sample_game.txt"))
    if "pruning" in chosen:
        ex.exp_pruning(rep, RESULTS, depths=(2, 3, 4) if quick else (2, 3, 4, 5), extra_depth=5 if quick else 6)
    if "greedy" in chosen:
        ex.exp_greedy(rep, RESULTS, n, games)
    if "heuristics" in chosen:
        ex.exp_heuristics(rep, RESULTS, n, games, depths=(3,) if quick else (3, 4))
    if "depth" in chosen:
        depth_summary = ex.exp_depth(rep, RESULTS, n, games,
                                     field_depths=(1, 2, 3) if quick else (1, 2, 3, 4, 5),
                                     struct_depths=(1, 2, 3) if quick else (1, 2, 3, 4),
                                     ladder_n=n)
    if "mcts" in chosen:
        ex.exp_mcts(rep, RESULTS, nm, games, depth_summary,
                    iterations=(50, 200) if quick else (50, 100, 200, 400, 800))
    if games:
        pd.concat(games).to_csv(os.path.join(RESULTS, "all_games.csv"), index=False)
    rep(f"\nTotal time: {(time.time() - t0) / 60:.1f} min   ->   results in {RESULTS}")
    with open(os.path.join(RESULTS, "report.txt"), "w") as f:
        f.write("\n".join(rep.lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["test", "demo", "trace", "experiments", "replot"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    if args.command == "test":
        run_tests()
    elif args.command == "demo":
        run_demo()
    elif args.command == "trace":
        run_trace()
    elif args.command == "replot":
        ex.replot(RESULTS)
    else:
        run_experiments(args.quick, args.only)
