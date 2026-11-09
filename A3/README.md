# ARC-AGI-1: a classical-AI attempt (DSL + program search)

No machine learning. A hand-written DSL of ~100 grid primitives, and state-space search for a program
that maps every demonstration input to its output. Same toolbox as assignments 1 and 2 (BFS, greedy, A*).

```
pip install -r requirements.txt                  # numpy (solver) + pandas (only for summarising the CSVs)
python test_dsl.py                               # 14 unit tests (primitives + search), ~1 s
python run_arc.py --suite                        # the 9 tasks in ./tasks with BFS
python run_arc.py --suite --compare              # BFS vs greedy vs A*  -> results/suite_results.csv
python run_arc.py --task tasks/00d62c1b.json     # one task: demos, program found, prediction vs expected
python run_arc.py --dir data/training --limit 100 --time 8     # batch over the bundled public training set
python run_arc.py --dir data/evaluation --limit 100 --time 8   # ... and the public evaluation set
```

| file | contents |
|---|---|
| `dsl.py` | the primitives = **all the prior knowledge of the system** |
| `search.py` | state, goal test, heuristic, BFS / greedy / A*, up to 2 predictions |
| `run_arc.py` | command line, ARC scoring, batch statistics |
| `test_dsl.py` | unit tests |
| `tasks/` | the 9 tested tasks (ARC-AGI-1 training set) |
| `data/` | the full public ARC-AGI-1 training and evaluation sets (400 + 400 tasks, Apache-2.0, from github.com/fchollet/ARC-AGI) |
| `results/` | `suite_results.csv`, `batch_training_bfs.csv`, `batch_evaluation_bfs.csv`, console logs |

## 1. Problem formulation
* **State**: the tuple of grids obtained by running one candidate program on *every demo input and on the
  test input(s)*. Carrying the test input means two programs that agree on the demos but differ on the test are
  never merged by duplicate detection.
* **Initial state**: the untouched inputs (empty program).
* **Actions**: apply one DSL primitive to all grids (70-195 actions per task; colour-parameterised primitives
  such as `replace(a->b)` or `fill_holes(c)` only use colours that occur in that task, so the count grows with the
  number of colours).
* **Goal**: all demo grids equal their demo outputs, exactly or after ONE consistent colour map (learned from the
  demos; unseen colours map to themselves).
* **Path cost**: 1 per primitive, so the first solution BFS finds is the shortest = simplest program.
* **No stated goal to infer**: unlike ARC-AGI-3, the demos define the goal. There is no reward signal; the only
  "feedback" is how far the current grids are from the demo outputs (the heuristic).
* **Heuristic** `h` = average over demos of the fraction of wrong cells (same shape), or 1 + relative size
  difference (different shape). Not admissible, so greedy / A* give no optimality guarantee.
  A*-style priority: `f = depth + 3*h`.
* **Prediction**: ARC allows two attempts, so the solver returns up to two *different* test predictions from
  programs that fit all demos. A task counts as solved only if every test output is reproduced exactly.

## 2. Algorithm and why
* **Program search over a DSL** was chosen because it is exactly state-space search, which this course covers,
  and because every answer is an *explicit program* I can print and explain. Alternatives: DFS (no shortest-program
  guarantee, can dive into useless deep programs); iterative deepening (re-expands the tree, duplicate detection
  makes BFS cheaper here); minimax/MCTS (single agent, no opponent); RL (needs an environment and many episodes,
  far more experience than 2-4 demos).
* **BFS** is the default: complete up to the depth limit (3) and returns the shortest program.
  **Greedy / A\*** use the heuristic to reach a solution with far fewer expansions when `h` is informative.
* Speed-ups that do not change the result: goal test at generation, states at the maximum depth are tested but
  never stored, a successor is first checked on the first demo only (shape, then colour map), primitive calls are
  memoised. My first version ran ~114 expansions/s and spent the whole time budget even after solving.

## 3. Instances tested (public training set, depth <= 3, 8 s budget)
| task | what the rule is | BFS result | program found | expanded BFS / greedy / A* |
|---|---|---|---|---|
| 00d62c1b | fill enclosed holes with 4 | solved | `fill_holes(4)` | 21 / 74 / 20 |
| 3af2c5a8 | mirror the grid into 4 quadrants | solved | `mirror4` | 19 / 2 / 2 |
| 1e0a9b12 | cells fall down | solved | `gravity_down` | 18 / 2 / 2 |
| 22eb0ac0 | join same-colour cells in a row | solved | `connect_h` | 13 / 26 / 28 |
| 28bf18c6 | crop the shape, repeat it twice | solved | `crop_bbox -> tile_h2` | 206 / 5 / 4 |
| 1f85a75f | crop the object out of noise | **wrong on test** | `crop_largest` | 11 / 2 / 2 |
| 25d8a9c8 | uniform row -> 5, else 0 | **wrong on test** | `roll_right -> recolor_smallest(5)` + colour map | 317 / 312 / 267 |
| 007bbfb7 | every non-zero cell becomes a copy of the whole grid | nothing found (exhausted) | - | 3762 / 3762 / 3762 |
| 0ca9ddb6 | colour-dependent patterns around pixels | nothing found (timeout) | - | 3025 / 4692 / 4638 |

All three strategies solve the same 5 tasks. Greedy and A* need far fewer expansions on 4 of the 5
(e.g. 206 -> 5 and 4 on 28bf18c6), but not always (00d62c1b: greedy 74 vs BFS 21), because the pixel-mismatch
heuristic only helps when the right primitive moves the grid closer to the output.

**Batch (first 100 tasks of each public set, BFS, 8 s, sampling error about +-7 points):**
| set | solved | fit demos, wrong on test | nothing found (exhausted / timeout) | mean input cells |
|---|---|---|---|---|
| public training | 17% | 5% | 42 / 36 | 163 |
| public evaluation | 6% | 0% | 28 / 66 | 257 |

14 of 17 training solutions use one primitive. Solve rate on shape-changing tasks was higher than on same-shape tasks
(training 26% vs 14%; evaluation 14% vs 2%). The hidden set is not available, so the training-to-evaluation drop
(17% -> 6%) is only a hint of the public/private gap, not a measurement of it.

## 4. Failure analysis (concrete)
1. **007bbfb7** (nothing found). The rule needs "tile the grid, but only where the input cell is non-zero"
   (self-similarity). The search exhausted all 3,761 distinct depth-<=3 states in 1.8 s and none matches: the concept
   is simply not in the DSL. Fix: one new primitive (a Kronecker product of the mask with the grid). But that fixes one
   task: the DSL arms race.
2. **25d8a9c8** (overfit). The true rule is "a row is 5 if all three cells are equal, else 0". The solver instead found
   `roll_right -> recolor_smallest(5)` plus a colour map `{4:5, 2:0, 3:0, 5:0, 7:0, 6:5, 9:5, 1:5}`: the colour map memorised
   which demo colours end up as 5, so it fits the demos by accident. The test contains colour 8, which never occurs in the
   demos, so it fails. Fix: restrict colour maps (e.g. only permutations / few entries), or add a "uniform row" primitive.
   Shortest-program preference (Occam) does not protect against this because the colour map is free.
3. **1f85a75f** (wrong on test). `crop_largest` fits both demos, but my "object" is a multicolour 8-connected blob, so on
   the test image noise touching the object enlarges the box (predicted 5x6, correct 4x4). Fix: define objects by
   colour, or remove noise first; or have a second, different attempt. Shows how a prior (what counts as an object) is
   itself a modelling decision.
4. **0ca9ddb6** (timeout). Rule: each red pixel gets yellow on its 4 diagonals, each blue pixel gets orange on its 4
   neighbours, other colours stay. It needs colour-conditional local patterns applied to every pixel; no short program
   of my primitives does this and the search ran out of time (this is the kind of "apply a rule depending on context"
   that ARC-AGI-2 emphasises).
Also: the DSL assumes background 0, so tasks with another background are out of reach.

## 5. Scaling and limits
* Branching factor b = 70-195 (it grows with the number of colours because of `replace(a->b)`). Raw b^3 is 0.3-7 million
  sequences. Duplicate detection plus failed/no-op primitives cut this a lot: the exhausted run on 007bbfb7 generated
  3,761 states instead of 1,092,727 (0.34%). Depth 4 multiplies the raw number of sequences by b again (70-195x), so it is out of reach within seconds.
* In the 100-task batches, 36% of training tasks and 66% of evaluation tasks hit the time limit before depth 3 finished;
  evaluation grids are larger (257 vs 163 cells on average) and each primitive costs more.
* Hard tasks need programs of 6-10 steps, which is ~100^8 sequences: impossible by brute force. Remedies are better
  heuristics, a learned prior over programs, or an LLM proposing candidates (not used here).
* A bigger DSL covers more tasks but also raises b, so each added primitive makes every other task slower.

## 6. Connection to Chollet's thesis
The system shows **skill, not skill-acquisition efficiency**. All the intelligence is in the DSL I wrote: object
definition, background = 0, which geometry operations exist, what "fill" means. Given the priors it needs only 2-4
demos, which looks efficient, but the efficiency is bought with a large amount of human prior knowledge written
specifically with ARC-like puzzles in mind. Chollet's point applies directly: the 17% -> 6% drop between public
training and evaluation, and the arms race for each failed task, show that the skill does not transfer to genuinely
novel tasks. The overfit cases show the other side of few-shot learning: with 2-4 demos several programs fit and
the system has no principled way to choose except program length.

## 7. LLM use
This code was written with an LLM assistant (Claude). What it contributed: the DSL, the search, the runner and tests.
Verification: 14 unit tests on each primitive and on the search (shortest program, colour map, unsolvable task reports
failure), and running on real tasks against the ground-truth outputs. Where it was unreliable: the first search was
much too slow (the unit-test workload took 2.6 s, then 0.14 s after the rewrite) and kept searching after finding a solution; the first `gravity` was awkward and was rewritten before
testing. Every primitive and the search loop are short enough to read and check line by line.

## 8. What this taught me
* A tiny DSL plus search solves a visible minority of ARC-AGI-1 and almost nothing of what is truly novel.
* Overfitting is the main danger in few-shot program synthesis, not search speed.
* The public/private distinction matters: a solver tuned on public tasks looks better than it is.
* Priors and search are exchangeable: more priors (primitives) mean less search per task but more engineering.

## 9. Worked example: BFS on a toy task
Demo 1 alone: input `[[1,0],[0,0]]` -> output `[[0,0],[1,0]]`.
Depth 0: the grids are unchanged, no match. Depth 1: apply every primitive to the demo AND the test input, drop
invalid or already-seen states, test each new state against the output. `rot90` and `flipud` both give
`[[0,0],[1,0]]` (so do `roll_down` and `gravity_down`); BFS reports the first two that give *different* test
predictions, so one demo leaves the answer ambiguous. (I ran this: it returns `rot90` and `flipud`.)
Add demo 2: `[[0,0],[0,3]]` -> `[[0,0],[0,3]]` (a cell already at the bottom stays). Now `rot90`, `flipud` and
`roll_down` all fail on demo 2 and only `gravity_down` fits (also verified by running it). This is why several
demos matter, and why shortest-program preference alone cannot resolve ambiguity.
