# Robo-Football (Assignment 2): game, players, experiments

Two-player, turn-based football on a 6x5 grid. Part 1 = the game, Part 2 = agents (Random, Greedy,
Minimax, Alpha-Beta, MCTS), Part 3 = experiments. Needs Python 3 with pandas and matplotlib.

```
pip install pandas matplotlib
python main.py test                 # 36 automated rule/agent checks (~1 s)
python main.py demo                 # watch a game, boards printed
python main.py trace                # alpha-beta trace you can follow by hand -> results/alphabeta_trace.txt
python main.py experiments --quick  # whole pipeline, small sample (~3 min)
python main.py experiments          # full run (~12 min on one core)
python main.py replot               # redraw figures from the saved CSVs
```

| file | what it holds |
|---|---|
| `game.py` | rules: state, legal moves, apply, end of match, rendering, invariants |
| `heuristics.py` | FIELD and STRUCTURE evaluation functions (+ a goals-only reference) |
| `agents.py` | Random, Greedy, Minimax, Alpha-Beta, MCTS, and the match runner |
| `tests.py` | automated checks |
| `experiments.py`, `main.py` | Part 3 + command line |
| `results/` | CSVs, PNGs, `report.txt` (full console output), `sample_game.txt`, `positions.txt`, `alphabeta_trace.txt` |

## 1. Rules and the decisions I had to make
Board 6 rows x 5 cols; A defends row 0 / attacks row 5, B the opposite. Turns strictly alternate A,B,A,B
whatever the possession. **Match = 80 turns (40 per team)**: enough for several attack attempts per team (rough estimate: an attack
takes ~4-5 own turns to advance, set up and shoot) while keeping depth-5 search cheap; 60 and 100 gave the
same qualitative picture in my tuning games. A goal on the last turn counts.

| open point in the spec | my decision |
|---|---|
| formation | A: A1(1,1) A2(1,3) A3(2,1) A4(2,3), keeper (0,2); B is the mirror (row r -> 5-r). Ball: A3 / B3. After a goal the **conceder** gets the ball and moves next. I first tried a "gap" formation (rows 2-3 empty): almost every game was 0-0, so I switched to a contact formation. |
| several opponents adjacent after a dribble | lowest cell index (lowest row, then lowest column) wins the ball. Keepers can tackle too. |
| defenders walking next to the carrier | **no tackle**: the rules only give a tackle on a *dribble*. |
| blocked passes | only opposing pieces block; teammates do not. Keeper is a legal receiver. |
| keeper holding the ball | may pass or dribble sideways on its line; cannot shoot (never in the attacking third). |
| no legal move | `SKIP` (the turn passes). Rare: 24 of 24,000 plies in random games (0.1%), none in 20 search-vs-greedy games. |

State = (10 piece cells, ball holder, side to move, turn, score), immutable; `apply` returns a new state.
Agents see the game only through `legal_moves / apply / is_terminal`; the runner re-checks every move for
legality, so a cheating agent is rejected (tested).

**Checks (`tests.py`)**: each rule on a hand-built position (tackle diagonal/orthogonal, tie-break, keeper
tackle, blocked/unblocked/diagonal passes, shooting zone for both teams, save at keeper distance 0/1 vs goal
at 2, reset + conceder restarts, last-turn goal, keeper sideways only); invariants over 150 random games
(no overlaps, keepers on their line, turns alternate, save <=> no goal, tackle <=> adjacent opponent);
mirror symmetry; heuristics zero-sum and mirror-symmetric; **minimax and alpha-beta give the same value and
move**; MCTS finds a free goal; games reproducible for a seed.

## 2. Players
* **Random**: uniform over legal moves. **Greedy**: best heuristic value after its own move (no look-ahead).
* **Minimax / Alpha-Beta**: same code path, depth in plies, same heuristic; root team = MAX. Alpha-beta
  prunes when `alpha >= beta`. Optional *move ordering* (shots, forward passes, forward dribbles first).
  Root moves are shuffled so ties are broken randomly; the *first* best move is kept (strict `>`), which is why
  alpha-beta returns exactly minimax's move.
* **MCTS (UCT)**: select by UCB1 (c=1), expand one child, roll out (random, or heuristic-guided with eps=0.3)
  for at most `rollout_depth` plies, score the end state by `tanh(h/60)`, back-propagate, play the most visited
  root child.
* **Heuristics** (all zero-sum: `h = 100*(goal diff) + side(me) - side(opp)`; one goal outweighs any shape):
  * **FIELD** - territory and shot threat: +6 possession, +4 per row the carrier has advanced, +0.5 per row for
    every outfield player, carrier in the shooting zone +15 (+4 per column between it and the keeper), and +25 if
    the keeper is not within 1 column (a shot would score).
  * **STRUCTURE** - passing network and defensive shape. With the ball: +3 per open passing lane, +1 per
    teammate within 2 cells, +1.5 per dribble square not next to an opponent, +8 per open pass to a teammate in
    the shooting zone (+20 more if the keeper cannot cover that column), shot terms as above. Without the ball:
    +2 per defender goal-side of the carrier, +2.5 per enemy passing lane we block, +2 per player marking the
    carrier (max 3), +15 if our keeper covers the carrier's column in the shooting zone.
  * *Why these ideas*: a shot is saved only if the keeper is within 1 column and a keeper steps 1 column per turn,
    so goals come from pulling the keeper sideways and then switching play. FIELD rewards the direct route,
    STRUCTURE rewards the passing/defensive structure that supports it. Both were adjusted once on separate
    tuning seeds (5000/7000) after early games were all 0-0 (wing-receiver term, keeper-gap term); final
    experiments use seeds 1000-4500.

## 3. Results (full run, `results/report.txt`)
Each pairing: 60 games (MCTS: 40) = 30 (20) seeds x both sides, same random 4-ply opening for both
orientations. Score% = (W + D/2)/N with a 95% Wilson interval; `p_sign` = exact sign test on decisive games
only. Wall-clock times vary between machines; nodes, scores and goals are exactly reproducible.

### Exp 1 - pruning (field heuristic; structure behaves the same, see `exp1_pruning_summary.csv`)
Nodes visited, 4 positions. All three searches returned the **same value in every case**.

| position | depth | minimax | alpha-beta | AB + ordering | saved | time (MM -> AB) |
|---|---|---|---|---|---|---|
| kickoff | 3 | 1,413 | 712 | 244 | 49.6% | 1.49 -> 0.83 ms |
| kickoff | 5 | 202,217 | 19,694 | 4,792 | 90.3% | 216.96 -> 23.39 ms |
| midfield duel | 5 | 108,735 | 14,607 | 6,265 | 86.6% | 118.29 -> 17.17 ms |
| attack in the zone | 5 | 63,505 | 15,618 | 4,077 | 75.4% | 69.36 -> 18.74 ms |
| defending a counter | 5 | 54,252 | 4,438 | 994 | 91.8% | 60.95 -> 5.53 ms |

Across depths 2-5 alpha-beta saves **85.5%** of nodes (94.5% with ordering). The saving *grows with depth*:
13-59% at depth 2, 27-62% at depth 3, 59-81% at depth 4, 71-92% at depth 5. Minimax grows about x10-15 per ply
at kickoff (1,413 -> ~20,600 -> 202,217 for depths 3-5); alpha-beta grows about x5 per ply (712 -> 19,694) and
with ordering about x4.4 per ply (244 -> 4,792), i.e. it moves towards the best-case sqrt(b^d) behaviour. Ordering is not always better (e.g. attack-in-the-zone, depth 2: 63 vs 57 nodes), because
a cheap static ordering can be wrong. Hypothesis (not tested) for the spread between positions: where one reply
is clearly best (defending a counter) cut-offs come early; where many moves have similar values (attack in the
zone) they come late.

### Exp 2 - Greedy vs search (X's results)
| X | Y | W-D-L | score |
|---|---|---|---|
| Greedy(field) | Random | 33-27-0 | 77.5% |
| Greedy(structure) | Random | 22-38-0 | 68.3% |
| AB(d=2, field) | Greedy(field) | 5-55-0 | 54.2% (p=.06) |
| AB(d=4, field) | Greedy(field) | 34-26-0 | 78.3% (p<.001) |
| AB(d=2, structure) | Greedy(structure) | 1-59-0 | 50.8% |
| AB(d=4, structure) | Greedy(structure) | 0-60-0 | 50.0% |

Look-ahead pays off with FIELD once it is deep enough (depth 4 here; depth 2 is barely better than greedy).
With STRUCTURE even depth 4 cannot beat its own greedy version (0-60-0). The style counts (Exp 3 profile) point
to why: Greedy(field) is tackled 5.0 times per game, Greedy(structure) 1.9 and it makes 21.7 defensive moves per
game against 13.9, i.e. it sits back. (This is an interpretation of those counts, not a separate experiment.)

### Exp 3 - heuristic vs heuristic (identical depth)
Head to head (field vs structure): depth 3: 0-58-2; depth 4: 0-60-0. **Evenly matched**: both defend almost
perfectly (goals conceded ~0). Gauntlet at depth 4 shows the difference in *attack*:

| agent | vs Random | vs Greedy(field) | vs Greedy(structure) | shots/g | goals/g |
|---|---|---|---|---|---|
| AB(d4, field) | 82.5% | 84.2% | 54.2% | 0.74 | 0.44 |
| AB(d4, structure) | 66.7% | 68.3% | 50.8% | 0.47 | 0.20 |

FIELD converts more chances; STRUCTURE is as hard to beat but creates fewer. STRUCTURE costs ~3-4x more time
per move (more features per evaluation).

### Exp 4 - depth vs performance (vs Greedy with the same heuristic)
| depth | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| FIELD score % | 52.5 | 52.5 | 60.0 | 86.7 | 86.7 |
| ms / move | 0.02 | 0.09 | 0.60 | 3.54 | 22.78 |
| nodes / move | 12 | 65 | 507 | 2,934 | 19,548 |

Nodes and time grow about x5-8 per ply. Against Greedy, depth 3 is modestly better (13-46-1, p=.0018), strength jumps at depth 4, and
the curve is then flat. Head-to-head the
extra depth still matters: d4 beats d3 13-47-0 (p=.0002), d5 beats d4 10-50-0 (p=.002), while d2 vs d1 and d3 vs d2
are not significant. STRUCTURE never beats Greedy(structure) at any depth (50-52.5%): its fortress is the limit,
not the depth.

### Exp 5 - MCTS (vs Greedy(field), 40 games each)
| MCTS | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|
| random roll-outs, depth 20 | - | 20.0 | 15.0 | 23.8 | 30.0 | 42.5 |
| guided roll-outs, depth 10 | 17.5 | - | 46.2 | - | 58.8 | - |
| time / move (random) | - | 2.01 ms | 4.10 | 8.29 | 16.57 | 35.36 |

MCTS's best result (58.8% at 38.86 ms/move) is far below depth-4 alpha-beta's 86.7% at 3.54 ms/move; against
depth-3 alpha-beta, MCTS(800 iterations) scores 36.2% (0-29-11) and guided MCTS(400) 41.2%. Roll-out depth 6 vs 40 (400 its) made no difference (36.2% both), so **roll-out quality, not length,
is the bottleneck**: heuristic-guided roll-outs at 400 iterations scored higher than random roll-outs at 800
(58.8% vs 42.5%) for about the same time per move, although the two intervals (43-73 vs 29-58) overlap. **Tactics probe** (20 seeds): finding "step the keeper over to stop a shot" takes MCTS 200 iterations
(80%) / 800 (100%); "pass to the wing, then shoot" needs 800 (100%; 30% at 200). Alpha-beta with a goals-only
evaluation needs exactly 1, 2, 3 plies for the three probes (6, 114 and 181 nodes, 0.03-0.2 ms; MCTS(800)
uses 16,800 nodes, 40-45 ms), and with the FIELD heuristic depth 1 finds all three: **a good heuristic substitutes for depth**. MCTS is the right tool when no
useful heuristic or branching is huge; here branching is small (<=~30) and the heuristic is cheap, so
alpha-beta wins on cost-benefit.

## 4. Limitations
* Equal-strength agents mostly draw, so many pairings have few decisive games; read score% with its interval.
* Heuristics and formation were tuned once (separate seeds); weights are hand-set, not optimised.
* MCTS parameters (c = 1, eps = 0.3, tanh scale 60) were not tuned; a stronger MCTS may exist.
* Match length (80) is a design choice; 60 and 100 were only checked informally.

## 5. Hand-tracing alpha-beta
`python main.py trace` prints, for a real position, every node with its alpha/beta, leaf values and the cut-off
line "PRUNE (alpha >= beta)". Procedure: MAX nodes raise alpha to the best child so far, MIN nodes lower beta;
the moment alpha >= beta the remaining siblings cannot change the root value and are skipped. The trace keeps
only the first 3 moves per node so the tree fits on a page (this changes which tree is searched, not the
algorithm).
