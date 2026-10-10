"""
Robo-Football  -  FILE 5/6: experiments and analysis (Part 3).

  exp 1  pruning     minimax vs alpha-beta (vs alpha-beta + move ordering): nodes and time,
                     both heuristics, 4 positions (kickoff + 3 mid-game), depths 2-5(6)
  exp 2  greedy      Greedy vs Alpha-Beta tournament (+ Random baseline)
  exp 3  heuristics  field vs structure at identical search settings, plus a "gauntlet"
                     against common opponents and a playing-style profile
  exp 4  depth       strength and cost as the search depth grows
  exp 5  mcts        MCTS vs Random / Greedy / Alpha-Beta; iterations and roll-out depth

All games are SEEDED, so every table can be reproduced exactly (only wall-clock time varies).
Every pairing alternates which side each agent plays: for each seed k two games are played,
X as team A and X as team B, with the same random opening.
"""
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from game import Game
from heuristics import HEURISTICS, MAIN_HEURISTICS, terminal_value
from agents import (AlphaBetaAgent, MinimaxAgent, build_agent, play_game, spec_label)

INF = float("inf")
GAME = Game()                                    # 80 turns


# ========================================================================================
# helpers
# ========================================================================================
class Report:
    """Prints to the terminal and keeps everything for results/report.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, *a):
        text = " ".join(str(x) for x in a)
        print(text, flush=True)
        self.lines.append(text)

    def table(self, df, **kw):
        self(df.to_string(index=False, **kw))


def ab(depth, h="field", ordering=False):
    return {"type": "alphabeta", "depth": depth, "heuristic": h, "ordering": ordering}


def greedy(h="field"):
    return {"type": "greedy", "heuristic": h}


RANDOM = {"type": "random"}


def mcts(it, rd=20, policy="random"):
    return {"type": "mcts", "iterations": it, "rollout_depth": rd, "policy": policy}


def sign_test_p(w, l):
    """Exact two-sided sign test on the DECISIVE games (draws carry no information about
    which agent is stronger).  p = probability of a split this lopsided if both were equal."""
    n = w + l
    if n == 0:
        return 1.0
    k = max(w, l)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


STAT_KEYS = ["possession", "passes", "dribbles", "shots", "goals", "shots_saved", "tackled", "defensive"]


def tournament(specX, specY, n_games, seed0, exp, opening_plies=4):
    """n_games games between X and Y (n_games/2 seeds, both sides each).  Returns a DataFrame
    with one row per game, always from X's point of view."""
    rows = []
    for k in range(n_games // 2):
        seed = seed0 + k
        for x_is_a in (True, False):
            x = build_agent(specX, 2 * seed + 1)
            y = build_agent(specY, 2 * seed + 2)
            a, b = (x, y) if x_is_a else (y, x)
            r = play_game(GAME, a, b, seed=seed, opening_plies=opening_plies)
            gx, gy = r["score"] if x_is_a else r["score"][::-1]
            sx, sy = (r["stats"][0], r["stats"][1]) if x_is_a else (r["stats"][1], r["stats"][0])
            row = {"Experiment": exp, "X": spec_label(specX), "Y": spec_label(specY), "Seed": seed,
                   "X_side": "A" if x_is_a else "B", "X_goals": gx, "Y_goals": gy,
                   "Result": "W" if gx > gy else ("L" if gx < gy else "D"),
                   "X_ms_move": 1000 * x.time / max(1, x.moves), "Y_ms_move": 1000 * y.time / max(1, y.moves),
                   "X_nodes_move": x.nodes / max(1, x.moves), "Y_nodes_move": y.nodes / max(1, y.moves)}
            for key in STAT_KEYS:
                row["X_" + key] = sx[key]
                row["Y_" + key] = sy[key]
            rows.append(row)
    return pd.DataFrame(rows)


def wilson(p, n, z=1.96):
    """95% Wilson interval for a score p over n games. Treating a draw as half a win this is
    conservative (real variance is a bit smaller) but, unlike a plain +/- 1.96*sd/sqrt(n), it never
    collapses to zero width when every game happens to be a draw."""
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def summarize(games: pd.DataFrame) -> pd.DataFrame:
    """One row per (X, Y) pairing: W/D/L, score %, goals, cost per move, sign-test p-value."""
    out = []
    for (x, y), g in games.groupby(["X", "Y"], sort=False):
        n = len(g)
        w, d, l = (g.Result == "W").sum(), (g.Result == "D").sum(), (g.Result == "L").sum()
        sc = g.Result.map({"W": 1.0, "D": 0.5, "L": 0.0})
        lo, hi = wilson(sc.mean(), n)
        out.append({"X": x, "Y": y, "Games": n, "W": w, "D": d, "L": l,
                    "Score_%": round(100 * sc.mean(), 1), "CI95_%": f"{100 * lo:.0f}-{100 * hi:.0f}",
                    "CI_lo": 100 * lo, "CI_hi": 100 * hi,
                    "Goals_for/g": round(g.X_goals.mean(), 2), "Goals_against/g": round(g.Y_goals.mean(), 2),
                    "p_sign": round(sign_test_p(w, l), 4),
                    "X_ms/move": round(g.X_ms_move.mean(), 2), "Y_ms/move": round(g.Y_ms_move.mean(), 2),
                    "X_nodes/move": round(g.X_nodes_move.mean()), "Y_nodes/move": round(g.Y_nodes_move.mean())})
    return pd.DataFrame(out)


def show_pairs(rep, summ, extra=()):
    cols = ["X", "Y", "Games", "W", "D", "L", "Score_%", "CI95_%", "Goals_for/g", "Goals_against/g",
            "p_sign"] + list(extra)
    rep.table(summ[cols])


def style_profile(games: pd.DataFrame) -> pd.DataFrame:
    """Average events per game for every agent that appears in `games` (as X or as Y)."""
    rows = []
    for label in sorted(set(games.X) | set(games.Y)):
        parts = []
        gx = games[games.X == label]
        gy = games[games.Y == label]
        for g, p in ((gx, "X_"), (gy, "Y_")):
            if len(g):
                parts.append(g[[p + k for k in STAT_KEYS]].rename(columns=lambda c: c[2:]))
        allg = pd.concat(parts)
        n = len(allg)
        row = {"Agent": label, "Games": n}
        row["Possession_%"] = round(100 * allg.possession.mean() / GAME.max_turns, 1)
        for k in STAT_KEYS[1:]:
            row[k + "/g"] = round(allg[k].mean(), 2)
        rows.append(row)
    return pd.DataFrame(rows)


# ========================================================================================
# experiment 1: pruning vs no pruning
# ========================================================================================
def build_positions():
    mk = Game.make_state
    return {
        "kickoff": GAME.initial_state(),
        "midfield duel": mk([(1, 2), (2, 0), (2, 2), (3, 4), (0, 1)],
                            [(4, 1), (3, 3), (4, 4), (3, 0), (5, 2)], ball=2, to_move=0, turn=20),
        "attack in the zone": mk([(2, 2), (3, 4), (4, 1), (3, 0), (0, 3)],
                                 [(4, 3), (3, 2), (4, 4), (2, 3), (5, 2)], ball=2, to_move=0, turn=30),
        "defending a counter": mk([(1, 1), (1, 3), (2, 2), (0, 4), (0, 2)],
                                  [(2, 0), (2, 4), (1, 2), (3, 2), (5, 3)], ball=7, to_move=0, turn=40),
    }


def timed_search(cls, depth, heuristic, state, ordering=False, repeats=2):
    best_t, agent, move = INF, None, None
    for _ in range(repeats):
        agent = cls(depth, heuristic, shuffle=False, ordering=ordering)
        t0 = time.perf_counter()
        move = agent.choose(GAME, state)
        best_t = min(best_t, time.perf_counter() - t0)
    return agent, move, best_t * 1000


def plot_pruning(df, out):
    positions = list(dict.fromkeys(df.Position))
    heurs = list(dict.fromkeys(df.Heuristic))
    fig, axes = plt.subplots(len(heurs), len(positions), figsize=(20, 8), sharex=True)
    for r, h in enumerate(heurs):
        for c, pname in enumerate(positions):
            ax = axes[r][c]
            d_ = df[(df.Heuristic == h) & (df.Position == pname)]
            ax.plot(d_.Depth, d_.MM_visited, "o-", label="Minimax")
            ax.plot(d_.Depth, d_.AB_visited, "s-", label="Alpha-beta")
            ax.plot(d_.Depth, d_.ABo_visited, "^-", label="Alpha-beta + ordering")
            ax.set_yscale("log")
            ax.set_xticks(sorted(d_.Depth.unique()))
            ax.set_title(f"{pname}  [{h}]", fontsize=10, fontweight="bold")
            ax.grid(True, ls="--", alpha=.5)
            if c == 0:
                ax.set_ylabel("nodes visited (log)")
            if r == len(heurs) - 1:
                ax.set_xlabel("search depth (plies)")
    axes[0][0].legend()
    fig.suptitle("Experiment 1: nodes visited with and without alpha-beta pruning", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "exp1_pruning.png"), dpi=150)
    plt.close(fig)


def exp_pruning(rep, out, depths=(2, 3, 4, 5), extra_depth=6):
    rep("\n" + "=" * 100 + "\nEXPERIMENT 1  -  alpha-beta pruning vs plain minimax (same depth, same heuristic)\n" + "=" * 100)
    positions = build_positions()
    with open(os.path.join(out, "positions.txt"), "w") as f:
        for name, s in positions.items():
            f.write(f"--- {name} ---\n{GAME.render(s)}\n"
                    f"legal moves for the side to move: {len(GAME.legal_moves(s))}\n\n")
    rows = []
    for h in MAIN_HEURISTICS:
        for pname, s in positions.items():
            branching = len(GAME.legal_moves(s))
            for d in list(depths) + [extra_depth]:
                res = {}
                for key, cls, ordering in (("MM", MinimaxAgent, False), ("AB", AlphaBetaAgent, False),
                                           ("ABo", AlphaBetaAgent, True)):
                    if key == "MM" and d > max(depths):
                        continue                                  # minimax at d=6 is too slow to be worth it
                    a, m, ms = timed_search(cls, d, h, s, ordering, repeats=1 if d >= 5 else 2)
                    res[key] = (a.visited, a.expanded, ms, a.last_value, m)
                row = {"Heuristic": h, "Position": pname, "Depth": d, "Root_moves": branching}
                for key, (vis, exp_, ms, val, m) in res.items():
                    row[f"{key}_visited"], row[f"{key}_expanded"] = vis, exp_
                    row[f"{key}_ms"], row[f"{key}_value"] = round(ms, 2), round(val, 3)
                vals = {round(v[3], 6) for v in res.values()}
                row["Same_value"] = len(vals) == 1
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "exp1_pruning.csv"), index=False)

    def pct(a, b):
        return round(100 * (1 - a / b), 1) if pd.notna(a) and pd.notna(b) else float("nan")
    t = pd.DataFrame({
        "Heuristic": df.Heuristic, "Position": df.Position, "Depth": df.Depth,
        "MM_nodes": df.MM_visited, "AB_nodes": df.AB_visited, "ABord_nodes": df.ABo_visited,
        "AB_saved_%": [pct(a, b) for a, b in zip(df.AB_visited, df.MM_visited)],
        "ABord_saved_%": [pct(a, b) for a, b in zip(df.ABo_visited, df.MM_visited)],
        "MM_ms": df.MM_ms, "AB_ms": df.AB_ms, "ABord_ms": df.ABo_ms,
        "AB_speedup_x": [round(a / b, 1) if pd.notna(a) else float("nan") for a, b in zip(df.MM_ms, df.AB_ms)],
        "Same_value": df.Same_value})
    t.to_csv(os.path.join(out, "exp1_pruning_summary.csv"), index=False)
    for h in MAIN_HEURISTICS:
        rep(f"\nHeuristic = {h}   (nodes = states visited; MM_nodes blank at depth 6: minimax not run)")
        rep.table(t[t.Heuristic == h].drop(columns="Heuristic"))
    ok = df.Same_value.all()
    rep(f"\nAll three searches returned the SAME value on every position/depth: {ok}")
    dd = df[df.Depth <= max(depths)]
    rep("Average nodes saved by alpha-beta over minimax: "
        f"{pct(dd.AB_visited.sum(), dd.MM_visited.sum())}%  (with move ordering: {pct(dd.ABo_visited.sum(), dd.MM_visited.sum())}%)")

    plot_pruning(df, out)


# ========================================================================================
# plotting helper for tournaments
# ========================================================================================
def plot_wdl(summ, title, path):
    labels = [f"{x}\nvs {y}" for x, y in zip(summ.X, summ.Y)]
    fig, ax = plt.subplots(figsize=(11, max(3, 0.75 * len(summ) + 1.5)))
    pos = range(len(summ))
    ax.barh(pos, summ.W, color="tab:green", label="X wins")
    ax.barh(pos, summ.D, left=summ.W, color="lightgrey", label="draws")
    ax.barh(pos, summ.L, left=summ.W + summ.D, color="tab:red", label="X loses")
    for i, (w, d, l) in enumerate(zip(summ.W, summ.D, summ.L)):
        ax.text(w / 2, i, str(w), ha="center", va="center", fontsize=9)
        ax.text(w + d / 2, i, str(d), ha="center", va="center", fontsize=9)
        if l:
            ax.text(w + d + l / 2, i, str(l), ha="center", va="center", fontsize=9)
    ax.set_yticks(list(pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("games")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ========================================================================================
# experiment 2: greedy vs alpha-beta
# ========================================================================================
def exp_greedy(rep, out, n, allgames):
    rep("\n" + "=" * 100 + f"\nEXPERIMENT 2  -  Greedy (no look-ahead) vs Alpha-Beta   [{n} games per pairing, sides alternated]\n" + "=" * 100)
    parts = []
    for h in MAIN_HEURISTICS:
        parts.append(tournament(greedy(h), RANDOM, n, 1000, "exp2"))
    for h in MAIN_HEURISTICS:
        for d in (2, 4):
            parts.append(tournament(ab(d, h), greedy(h), n, 1100 + 10 * d, "exp2"))
    games = pd.concat(parts)
    allgames.append(games)
    summ = summarize(games)
    summ.to_csv(os.path.join(out, "exp2_greedy_vs_search.csv"), index=False)
    show_pairs(rep, summ, ["X_ms/move", "Y_ms/move"])
    plot_wdl(summ, "Experiment 2: Greedy vs Random and Alpha-Beta vs Greedy (X's results)",
             os.path.join(out, "exp2_greedy_vs_search.png"))
    return summ


# ========================================================================================
# experiment 3: heuristic vs heuristic
# ========================================================================================
def exp_heuristics(rep, out, n, allgames, depths=(3, 4)):
    rep("\n" + "=" * 100 + f"\nEXPERIMENT 3  -  heuristic vs heuristic (identical alpha-beta search)   [{n} games per pairing]\n" + "=" * 100)
    head = pd.concat([tournament(ab(d, "field"), ab(d, "structure"), n, 2000 + 10 * d, "exp3") for d in depths])
    allgames.append(head)
    s1 = summarize(head)
    rep("\nHead to head (X = field, Y = structure):")
    show_pairs(rep, s1, ["X_ms/move", "Y_ms/move"])

    d = max(depths)
    opponents = [RANDOM, greedy("field"), greedy("structure")]
    gl = pd.concat([tournament(ab(d, h), o, n, 2200 + 10 * i, "exp3-gauntlet")
                    for h in MAIN_HEURISTICS for i, o in enumerate(opponents)])
    allgames.append(gl)
    s2 = summarize(gl)
    rep(f"\nGauntlet - each heuristic with alpha-beta depth {d} against the same three opponents:")
    show_pairs(rep, s2, ["X_ms/move"])

    prof_games = pd.concat([head[head.X.str.contains(f"d={d}")], gl])
    prof = style_profile(prof_games[prof_games.X.str.startswith("AB")])
    rep(f"\nPlaying style of the two heuristics (alpha-beta depth {d}; averages per game over all their games):")
    rep.table(prof[prof.Agent.str.contains(f"d={d}")])
    pd.concat([s1.assign(Part="head-to-head"), s2.assign(Part="gauntlet")]).to_csv(
        os.path.join(out, "exp3_heuristics.csv"), index=False)
    prof.to_csv(os.path.join(out, "exp3_style_profile.csv"), index=False)
    plot_wdl(pd.concat([s1, s2]), "Experiment 3: field vs structure heuristic (X's results)",
             os.path.join(out, "exp3_heuristics.png"))


# ========================================================================================
# experiment 4: depth vs performance
# ========================================================================================
def plot_depth(summ, out):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for h, colr in (("field", "tab:blue"), ("structure", "tab:red")):
        d_ = summ[summ.Heuristic == h]
        ax[0].errorbar(d_.Depth, d_["Score_%"], yerr=[d_["Score_%"] - d_.CI_lo, d_.CI_hi - d_["Score_%"]],
                       marker="o", color=colr, capsize=3, label=h)
        ax[1].plot(d_.Depth, d_["X_ms/move"], "o-", color=colr, label=h)
        ax[2].plot(d_.Depth, d_["X_nodes/move"], "o-", color=colr, label=h)
    ax[0].set_title("Strength: score % vs Greedy (same heuristic)", fontweight="bold")
    ax[0].set_ylabel("score % (win=1, draw=0.5); bars = 95% interval")
    ax[1].set_title("Cost: time per move", fontweight="bold")
    ax[1].set_ylabel("ms / move (log)")
    ax[2].set_title("Cost: nodes per move", fontweight="bold")
    ax[2].set_ylabel("nodes / move (log)")
    for a in ax:
        a.set_xlabel("alpha-beta depth (plies)")
        a.set_xticks(sorted(summ.Depth.unique()))
        a.grid(True, ls="--", alpha=.5)
        a.legend()
    ax[1].set_yscale("log")
    ax[2].set_yscale("log")
    fig.suptitle("Experiment 4: strength and cost as search depth grows", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "exp4_depth.png"), dpi=150)
    plt.close(fig)


def exp_depth(rep, out, n, allgames, field_depths=(1, 2, 3, 4, 5), struct_depths=(1, 2, 3, 4), ladder_n=None):
    rep("\n" + "=" * 100 + f"\nEXPERIMENT 4  -  depth vs strength and cost   [{n} games per point, reference opponent = Greedy]\n" + "=" * 100)
    parts = []
    for h, depths in (("field", field_depths), ("structure", struct_depths)):
        for d in depths:
            parts.append(tournament(ab(d, h), greedy(h), n, 3000 + 10 * d, "exp4"))
    games = pd.concat(parts)
    allgames.append(games)
    summ = summarize(games)
    summ["Depth"] = [int(x.split("d=")[1].split(",")[0]) for x in summ.X]
    summ["Heuristic"] = [x.split(",")[1].split(")")[0] for x in summ.X]
    summ.to_csv(os.path.join(out, "exp4_depth.csv"), index=False)
    show_pairs(rep, summ, ["X_ms/move", "X_nodes/move"])

    ln = ladder_n or n
    ladder = pd.concat([tournament(ab(d), ab(d - 1), ln, 3500 + 10 * d, "exp4-ladder") for d in (2, 3, 4, 5)])
    allgames.append(ladder)
    ls = summarize(ladder)
    ls.to_csv(os.path.join(out, "exp4_depth_ladder.csv"), index=False)
    rep(f"\nDepth ladder (field heuristic): each depth against the previous depth   [{ln} games]")
    show_pairs(rep, ls, ["X_ms/move", "Y_ms/move"])

    plot_depth(summ, out)
    return summ


# ========================================================================================
# experiment 5: MCTS
# ========================================================================================
def plot_cost_benefit(summ, depth_summary, out):
    # cost-benefit: every agent against the SAME reference (Greedy(field)); x = time per move
    pts = []
    mg = summ[(summ.Y == "Greedy(field)") & summ.X.str.startswith("MCTS") & ~summ.X.str.contains("rd=6|rd=40")]
    for _, r in mg.iterrows():
        pts.append((r.X, r["X_ms/move"], r["Score_%"], "tab:orange" if "guided" in r.X else "tab:green"))
    if depth_summary is not None:
        for _, r in depth_summary[depth_summary.Heuristic == "field"].iterrows():
            pts.append((r.X.replace(",field", ""), max(r["X_ms/move"], 0.01), r["Score_%"], "tab:blue"))
    fig, ax = plt.subplots(figsize=(9, 6))
    for lab, ms, sc, colr in pts:
        ax.scatter(ms, sc, color=colr, s=60)
        ax.annotate(lab.replace("MCTS(it=", "MCTS ").replace(",rd=20)", ""), (ms, sc),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.scatter([], [], color="tab:blue", label="Alpha-beta (field)")
    ax.scatter([], [], color="tab:green", label="MCTS (random roll-outs, depth 20)")
    ax.scatter([], [], color="tab:orange", label="MCTS (heuristic-guided roll-outs, depth 10)")
    ax.set_xscale("log")
    ax.set_xlabel("time per move (ms, log scale)")
    ax.set_ylabel("score % against Greedy(field)")
    ax.set_title("Experiment 5: cost-benefit (all agents vs the same reference opponent)", fontweight="bold")
    ax.axhline(50, color="grey", lw=.8, ls=":")
    ax.grid(True, ls="--", alpha=.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "exp5_cost_benefit.png"), dpi=150)
    plt.close(fig)


def exp_mcts(rep, out, n, allgames, depth_summary=None, iterations=(50, 100, 200, 400, 800)):
    rep("\n" + "=" * 100 + f"\nEXPERIMENT 5  -  MCTS cost-benefit   [{n} games per pairing]\n" + "=" * 100)
    parts = [tournament(mcts(it), greedy("field"), n, 4000 + it, "exp5") for it in iterations]
    full = len(iterations) >= 5
    vs_random = (iterations[1], iterations[3]) if full else (iterations[0],)
    vs_ab = (iterations[1], iterations[3], iterations[4]) if full else tuple(iterations)
    parts += [tournament(mcts(it), RANDOM, n, 4100 + it, "exp5") for it in vs_random]
    parts += [tournament(mcts(it), ab(3), n, 4200 + it, "exp5") for it in vs_ab]
    parts += [tournament(mcts(400, rd), greedy("field"), n, 4300 + rd, "exp5-rollout") for rd in (6, 40)]
    guided = (25, 100, 400) if full else (25,)          # heuristic-guided roll-outs (depth 10)
    parts += [tournament(mcts(it, 10, "guided"), greedy("field"), n, 4400 + it, "exp5-guided") for it in guided]
    parts += [tournament(mcts(it, 10, "guided"), ab(3), n, 4500 + it, "exp5-guided") for it in guided]
    games = pd.concat(parts)
    allgames.append(games)
    summ = summarize(games)
    summ.to_csv(os.path.join(out, "exp5_mcts.csv"), index=False)
    show_pairs(rep, summ, ["X_ms/move", "Y_ms/move", "X_nodes/move"])

    plot_cost_benefit(summ, depth_summary, out)
    exp_tactics(rep, out, iterations=(50, 200, 800, 3200) if full else (50, 200), seeds=20 if full else 6)
    return summ


def tactic_probes():
    """Three positions whose best move is FORCED and needs 1, 2 and 3 plies of look-ahead."""
    from game import SHOOT, MOVE, PASS, COL
    from tests import scenario
    return [
        ("take a free goal (perfect search needs 1 ply)",
         scenario(a={2: (4, 0)}, b={4: (5, 3)}, ball=2),
         lambda m: m[0] == SHOOT),
        ("stop a shot: keeper must step to column 1 (needs 2 plies)",
         scenario(a={2: (4, 0)}, b={0: (3, 4), 1: (3, 3), 2: (4, 4), 3: (4, 3), 4: (5, 2)}, ball=2, to_move=1),
         lambda m: m[0] == MOVE and m[1] == 9 and COL[m[2]] == 1),
        ("pass to the wing, then shoot past the keeper (needs 3 plies)",
         scenario(a={2: (2, 2), 3: (4, 4)}, b={4: (5, 1)}, ball=2),
         lambda m: m[0] == PASS and m[2] == 3),
    ]


def exp_tactics(rep, out, iterations, seeds):
    rep("\nTactics probe: how often is the forced best move found?  Alpha-beta is deterministic.  "
        "'goals' = evaluation that only counts goals (shows the depth really needed);\n'field' = the field heuristic "
        f"(already rewards shot threats, so it needs less depth).  MCTS is run with {seeds} seeds.")
    rows = []
    for name, s, ok in tactic_probes():
        assert GAME.legal_moves(s)
        for hname, d in (("goals", 1), ("goals", 2), ("goals", 3), ("field", 1)):
            a = AlphaBetaAgent(d, hname, shuffle=False)
            t0 = time.perf_counter()
            m = a.choose(GAME, s)
            rows.append({"Probe": name, "Agent": f"AB(d={d},{hname})", "Found_%": 100.0 if ok(m) else 0.0,
                         "ms/move": round(1000 * (time.perf_counter() - t0), 2), "Nodes": a.visited})
        for it in iterations:
            hits, tm, nodes = 0, 0.0, 0
            for sd in range(seeds):
                ag = build_agent(mcts(it), sd)
                t0 = time.perf_counter()
                hits += ok(ag.choose(GAME, s))
                tm += time.perf_counter() - t0
                nodes += ag.last_nodes
            rows.append({"Probe": name, "Agent": f"MCTS(it={it})", "Found_%": round(100 * hits / seeds, 0),
                         "ms/move": round(1000 * tm / seeds, 2), "Nodes": round(nodes / seeds)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "exp5_tactics.csv"), index=False)
    rep.table(df)


# ========================================================================================
# viva helpers: a printable alpha-beta trace and a sample game
# ========================================================================================
def alphabeta_trace(state, depth, heuristic, branch_limit=None, game=GAME):
    """Text trace of alpha-beta (alpha/beta at every node, leaf values, cut-offs).
    `branch_limit` keeps only the first K moves per node so the tree is small enough to
    follow by hand (it changes WHICH tree is searched, not how alpha-beta works)."""
    h = HEURISTICS[heuristic]
    root = state.to_move
    out = []
    fmt = lambda x: "-inf" if x == -INF else ("+inf" if x == INF else f"{x:g}")

    def leaf(s):
        return terminal_value(s, root) if game.is_terminal(s) else h(s, root)

    def rec(s, d, alpha, beta, ind, desc):
        pad = "    " * ind
        if d == 0 or game.is_terminal(s):
            v = leaf(s)
            out.append(f"{pad}{desc}: LEAF  h={v:.1f}")
            return v
        is_max = s.to_move == root
        moves = game.legal_moves(s)
        shown = moves if not branch_limit else moves[:branch_limit]
        out.append(f"{pad}{desc}: {'MAX' if is_max else 'MIN'} node  alpha={fmt(alpha)} beta={fmt(beta)}"
                   f"  ({len(shown)} of {len(moves)} moves considered)")
        best = -INF if is_max else INF
        for i, m in enumerate(shown):
            v = rec(game.apply(s, m), d - 1, alpha, beta, ind + 1, game.describe(s, m))
            if is_max:
                best = max(best, v)
                alpha = max(alpha, best)
            else:
                best = min(best, v)
                beta = min(beta, best)
            out.append(f"{pad}  child returned {v:.1f} -> node value {best:.1f}, alpha={fmt(alpha)} beta={fmt(beta)}")
            if alpha >= beta:
                left = len(shown) - i - 1
                if left:
                    out.append(f"{pad}  PRUNE (alpha >= beta): the remaining {left} move(s) are never evaluated")
                break
        return best

    v = rec(state, depth, -INF, INF, 0, "ROOT")
    out.append(f"\nValue of the root for team {'AB'[root]} = {v:.1f}")
    return "\n".join(out)


def sample_game(path):
    a = build_agent(ab(4, "field"), 1)
    b = build_agent(ab(4, "structure"), 2)
    r = play_game(GAME, a, b, seed=3, record=True)
    with open(path, "w") as f:
        f.write(f"Sample game: A = {a.name}   B = {b.name}   final score {r['score'][0]}-{r['score'][1]}\n"
                "(the first 4 plies are random openings)\n\n" + "\n".join(r["log"]))


def replot(out):
    """Rebuild the figures from the saved CSV files (no experiment is re-run)."""
    p = lambda name: os.path.join(out, name)
    if os.path.exists(p("exp1_pruning.csv")):
        plot_pruning(pd.read_csv(p("exp1_pruning.csv")), out)
    if os.path.exists(p("exp2_greedy_vs_search.csv")):
        plot_wdl(pd.read_csv(p("exp2_greedy_vs_search.csv")),
                 "Experiment 2: Greedy vs Random and Alpha-Beta vs Greedy (X's results)", p("exp2_greedy_vs_search.png"))
    if os.path.exists(p("exp3_heuristics.csv")):
        plot_wdl(pd.read_csv(p("exp3_heuristics.csv")), "Experiment 3: field vs structure heuristic (X's results)",
                 p("exp3_heuristics.png"))
    depth = pd.read_csv(p("exp4_depth.csv")) if os.path.exists(p("exp4_depth.csv")) else None
    if depth is not None:
        plot_depth(depth, out)
    if os.path.exists(p("exp5_mcts.csv")):
        plot_cost_benefit(pd.read_csv(p("exp5_mcts.csv")), depth, out)
