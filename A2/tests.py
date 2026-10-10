"""
Robo-Football  -  FILE 4/6: automated checks.   Run:  python tests.py     (or: python main.py test)

Part A  rule checks     each rule of the specification is tested on a hand-built position
Part B  invariants      thousands of random moves: no overlaps, keepers on their line, etc.
Part C  agent checks    minimax == alpha-beta, heuristics are zero-sum and symmetric, agents
                        only return legal moves, a forced goal is found, runs are reproducible
"""
import random
import unittest

import game as gm
from game import (A, B, DRIBBLE, MOVE, PASS, SHOOT, SKIP, Game, cell, check_invariants, mirror)
from heuristics import HEURISTICS
from agents import (AlphaBetaAgent, GreedyAgent, MCTSAgent, MinimaxAgent, RandomAgent, build_agent,
                    play_game)

G = Game()

# Pieces a scenario does not mention are parked on free cells of their own back rows, so a
# scenario only contains what matters for the rule under test.
PARK_OUT = [(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 4), (1, 1), (1, 3), (1, 2)]
PARK_GK = [(0, 2), (0, 1), (0, 3), (0, 0), (0, 4)]


def scenario(a=None, b=None, ball=2, to_move=A, turn=0, score=(0, 0)):
    """a / b: {piece_number_within_team: (row, col)}; piece 4 is the keeper."""
    a, b = dict(a or {}), dict(b or {})
    taken = set(a.values()) | set(b.values())
    teams = []
    for team, spec in ((A, a), (B, b)):
        flip = (lambda rc: rc) if team == A else (lambda rc: (5 - rc[0], rc[1]))
        cells = [None] * 5
        for k, v in spec.items():
            cells[k] = v
        for k in range(5):
            if cells[k] is None:
                for cand in (PARK_GK if k == 4 else PARK_OUT):
                    cand = flip(cand)
                    if cand not in taken:
                        cells[k] = cand
                        taken.add(cand)
                        break
        teams.append(cells)
    return Game.make_state(teams[0], teams[1], ball, to_move, turn, score)


def kinds(s):
    return [m[0] for m in G.legal_moves(s)]


def moves_of(s, kind):
    return [m for m in G.legal_moves(s) if m[0] == kind]


class TestRules(unittest.TestCase):
    # ---------------------------------------------------------------- set-up
    def test_initial_state(self):
        s = G.initial_state()
        check_invariants(s)
        self.assertEqual((s.turn, s.to_move, s.score, s.ball), (0, A, (0, 0), 2))
        for i in range(5):                                   # B is the vertical mirror of A
            self.assertEqual(gm.ROW[s.pos[i + 5]], 5 - gm.ROW[s.pos[i]])
            self.assertEqual(gm.COL[s.pos[i + 5]], gm.COL[s.pos[i]])
        self.assertEqual(len(set(s.pos)), 10)

    def test_mirror_is_involution(self):
        rng = random.Random(1)
        for _ in range(50):
            s = random_state(rng)
            self.assertEqual(mirror(mirror(s)), s)

    # ---------------------------------------------------------------- dribble & tackle
    def test_dribble_into_empty_cell_only(self):
        s = scenario(a={2: (2, 2), 3: (2, 3)}, ball=2)
        dests = {m[2] for m in moves_of(s, DRIBBLE)}
        self.assertNotIn(cell(2, 3), dests)                  # occupied by a teammate
        self.assertEqual(dests, {cell(r, c) for r in (1, 2, 3) for c in (1, 2, 3)} - {cell(2, 2), cell(2, 3)})

    def test_safe_dribble_keeps_ball(self):
        s = scenario(a={2: (2, 2)}, b={0: (4, 3)}, ball=2)
        s2 = G.apply(s, (DRIBBLE, 2, cell(2, 1)))
        self.assertEqual(s2.ball, 2)
        self.assertEqual(s2.pos[2], cell(2, 1))

    def test_dribble_next_to_opponent_loses_ball_diagonal(self):
        s = scenario(a={2: (2, 2)}, b={0: (4, 3)}, ball=2)   # (3,2) is diagonal to (4,3)
        s2 = G.apply(s, (DRIBBLE, 2, cell(3, 2)))
        self.assertEqual(s2.ball, 5)                          # B's piece 0 wins the ball
        self.assertEqual(s2.pos[2], cell(3, 2))               # dribbler stays where it moved
        self.assertEqual(s2.to_move, B)

    def test_dribble_next_to_opponent_loses_ball_orthogonal(self):
        s = scenario(a={2: (2, 2)}, b={0: (4, 2)}, ball=2)
        self.assertEqual(G.apply(s, (DRIBBLE, 2, cell(3, 2))).ball, 5)

    def test_tackle_tie_break_lowest_cell(self):
        s = scenario(a={2: (2, 2)}, b={0: (4, 3), 1: (4, 1)}, ball=2)   # both touch (3,2)
        s2 = G.apply(s, (DRIBBLE, 2, cell(3, 2)))
        self.assertEqual(s2.ball, 6)                          # (4,1) has the lower cell index than (4,3)

    def test_keeper_can_tackle(self):
        s = scenario(a={2: (3, 2)}, b={0: (3, 0), 1: (3, 4), 2: (4, 4), 3: (4, 0)}, ball=2)   # only the keeper is near
        s2 = G.apply(s, (DRIBBLE, 2, cell(4, 2)))
        self.assertEqual(s2.ball, 9)

    def test_defensive_move_next_to_carrier_does_not_tackle(self):
        s = scenario(a={2: (2, 2)}, b={0: (4, 2)}, ball=2, to_move=B)
        s2 = G.apply(s, (MOVE, 5, cell(3, 2)))                # B walks next to the carrier
        self.assertEqual(s2.ball, 2)

    # ---------------------------------------------------------------- passing
    def test_pass_blocked_by_opponent_on_row(self):
        s = scenario(a={2: (2, 0), 3: (2, 4)}, b={0: (2, 2)}, ball=2)
        self.assertNotIn((PASS, 2, 3), G.legal_moves(s))

    def test_pass_clear_row(self):
        s = scenario(a={2: (2, 0), 3: (2, 4)}, b={0: (3, 2)}, ball=2)
        self.assertIn((PASS, 2, 3), G.legal_moves(s))

    def test_pass_not_blocked_by_teammate(self):
        s = scenario(a={2: (2, 0), 3: (2, 4), 1: (2, 2)}, ball=2)
        self.assertIn((PASS, 2, 3), G.legal_moves(s))

    def test_pass_diagonal_blocked_and_clear(self):
        base = dict(a={2: (1, 1), 3: (3, 3)})
        self.assertNotIn((PASS, 2, 3), G.legal_moves(scenario(b={0: (2, 2)}, ball=2, **base)))
        self.assertIn((PASS, 2, 3), G.legal_moves(scenario(b={0: (2, 3)}, ball=2, **base)))

    def test_pass_needs_a_straight_line(self):
        s = scenario(a={2: (1, 1), 3: (3, 2)}, ball=2)        # knight-move apart
        self.assertNotIn((PASS, 2, 3), G.legal_moves(s))

    def test_pass_adjacent_and_to_keeper(self):
        s = scenario(a={2: (1, 2), 3: (1, 3)}, b={0: (2, 2)}, ball=2)   # AG at (0,2) is straight below
        self.assertIn((PASS, 2, 3), G.legal_moves(s))
        self.assertIn((PASS, 2, 4), G.legal_moves(s))

    def test_pass_only_moves_the_ball(self):
        s = scenario(a={2: (2, 0), 3: (2, 4)}, ball=2)
        s2 = G.apply(s, (PASS, 2, 3))
        self.assertEqual((s2.ball, s2.pos, s2.turn, s2.to_move), (3, s.pos, 1, B))

    # ---------------------------------------------------------------- shooting
    def test_shoot_only_in_attacking_third(self):
        for r, expect_a in [(0, False), (1, False), (2, False), (3, False), (4, True), (5, True)]:
            s = scenario(a={2: (r, 4)}, b={4: (5, 0)}, ball=2)
            self.assertEqual(SHOOT in kinds(s), expect_a, f"A on row {r}")
        for r, expect_b in [(5, False), (4, False), (3, False), (2, False), (1, True), (0, True)]:
            s = scenario(b={2: (r, 4)}, ball=7, to_move=B, a={4: (0, 0)})
            self.assertEqual(SHOOT in kinds(s), expect_b, f"B on row {r}")

    def test_shot_saved_when_keeper_within_one_column(self):
        for kc in (1, 2, 3):                                  # shooter column 2: keeper col 1,2,3 -> saved
            s = scenario(a={2: (4, 2)}, b={4: (5, kc)}, ball=2)
            s2 = G.apply(s, (SHOOT, 2, 0))
            self.assertEqual(s2.ball, 9, f"keeper at col {kc}")
            self.assertEqual(s2.score, (0, 0))
            self.assertEqual(s2.pos[9], cell(5, kc))          # keeper stays put

    def test_shot_scores_when_keeper_two_columns_away(self):
        s = scenario(a={2: (4, 0)}, b={4: (5, 2)}, ball=2)
        s2 = G.apply(s, (SHOOT, 2, 0))
        self.assertEqual(s2.score, (1, 0))

    def test_goal_resets_and_conceder_restarts_with_ball(self):
        s = scenario(a={2: (5, 0)}, b={4: (5, 3)}, ball=2)
        s2 = G.apply(s, (SHOOT, 2, 0))
        self.assertEqual(s2.pos, gm.KICKOFF_POS)
        self.assertEqual(s2.ball, gm.KICKOFF_CARRIER[B])      # B conceded -> B has the ball
        self.assertEqual(s2.to_move, B)                       # ... and B moves next
        s = scenario(b={2: (0, 4)}, a={4: (0, 1)}, ball=7, to_move=B)
        s2 = G.apply(s, (SHOOT, 7, 0))
        self.assertEqual(s2.score, (0, 1))
        self.assertEqual((s2.ball, s2.to_move, s2.pos), (gm.KICKOFF_CARRIER[A], A, gm.KICKOFF_POS))

    # ---------------------------------------------------------------- defending / keeper
    def test_defence_moves_exactly_one_piece_one_step(self):
        s = scenario(a={2: (2, 2)}, ball=2, to_move=B)
        for m in G.legal_moves(s):
            self.assertEqual(m[0], MOVE)
            s2 = G.apply(s, m)
            changed = [i for i in range(10) if s.pos[i] != s2.pos[i]]
            self.assertEqual(len(changed), 1)
            i = changed[0]
            self.assertEqual(gm.CHEB[s.pos[i]][s2.pos[i]], 1)
            self.assertEqual(i // 5, B)
        check_invariants(G.apply(s, G.legal_moves(s)[0]))

    def test_keeper_moves_sideways_on_goal_line_only(self):
        s = scenario(a={0: (1, 0), 1: (1, 1), 2: (1, 3), 3: (1, 4)}, ball=7, to_move=A)   # A repositions
        gk_moves = [m for m in G.legal_moves(s) if m[1] == 4]
        self.assertTrue(gk_moves)
        for m in gk_moves:
            self.assertEqual(gm.ROW[m[2]], 0)
            self.assertEqual(abs(gm.COL[m[2]] - gm.COL[s.pos[4]]), 1)

    def test_no_move_into_occupied_cell(self):
        s = scenario(a={2: (2, 2), 3: (2, 3)}, ball=7, to_move=A)
        occ = set(s.pos)
        for m in G.legal_moves(s):
            self.assertNotIn(m[2], occ)

    def test_keeper_with_ball_dribbles_sideways_passes_but_never_shoots(self):
        s = scenario(a={4: (0, 2), 0: (1, 2), 1: (1, 0), 2: (1, 4), 3: (1, 1)}, ball=4)
        self.assertNotIn(SHOOT, kinds(s))
        drib = moves_of(s, DRIBBLE)
        self.assertEqual({m[2] for m in drib}, {cell(0, 1), cell(0, 3)})      # sideways only
        self.assertTrue(moves_of(s, PASS))

    # ---------------------------------------------------------------- turns and end of match
    def test_turns_alternate_regardless_of_possession(self):
        s = scenario(a={2: (2, 0), 3: (2, 4)}, ball=2)
        s2 = G.apply(s, (PASS, 2, 3))                         # A keeps the ball after its pass ...
        self.assertEqual(s2.ball // 5, A)
        self.assertEqual(s2.to_move, B)                       # ... but it is B's turn
        self.assertTrue(all(m[0] == MOVE for m in G.legal_moves(s2)))   # B must reposition

    def test_match_ends_after_budget_and_last_goal_counts(self):
        s = scenario(a={2: (4, 0)}, b={4: (5, 3)}, ball=2, turn=G.max_turns - 1)
        self.assertFalse(G.is_terminal(s))
        s2 = G.apply(s, (SHOOT, 2, 0))
        self.assertTrue(G.is_terminal(s2))
        self.assertEqual(G.legal_moves(s2), [])
        self.assertEqual(G.winner(s2), A)
        self.assertIsNone(G.winner(scenario(score=(1, 1))))

    def test_apply_does_not_modify_input_state(self):
        s = G.initial_state()
        key = s.key()
        for m in G.legal_moves(s):
            G.apply(s, m)
        self.assertEqual(s.key(), key)


def random_state(rng, plies=None):
    s = G.initial_state()
    for _ in range(plies if plies is not None else rng.randint(2, 40)):
        if G.is_terminal(s):
            break
        s = G.apply(s, rng.choice(G.legal_moves(s)))
    return s


class TestInvariants(unittest.TestCase):
    def test_random_playouts_keep_the_rules(self):
        rng = random.Random(0)
        for game_no in range(150):
            s = G.initial_state()
            goals = 0
            while not G.is_terminal(s):
                legal = G.legal_moves(s)
                self.assertTrue(legal)
                m = rng.choice(legal)
                n = G.apply(s, m)
                check_invariants(n)                           # no overlap, keepers on their line, ...
                self.assertEqual(n.turn, s.turn + 1)
                self.assertEqual(n.to_move, 1 - s.to_move)    # strict alternation
                self.assertGreaterEqual(n.score[0], s.score[0])
                self.assertGreaterEqual(n.score[1], s.score[1])
                self.assertLessEqual(n.score[0] + n.score[1] - s.score[0] - s.score[1], 1)
                if m[0] == SHOOT:
                    opp_gk = 5 * (1 - s.to_move) + 4
                    saved = abs(gm.COL[s.pos[opp_gk]] - gm.COL[s.pos[m[1]]]) <= 1
                    self.assertEqual(n.score == s.score, saved)       # save <=> no goal
                if m[0] == DRIBBLE:
                    near = [j for j in range(5 * (1 - s.to_move), 5 * (1 - s.to_move) + 5)
                            if n.pos[j] in gm.NEIGH_SET[m[2]]]
                    self.assertEqual(n.ball // 5 != s.ball // 5, bool(near))   # tackle <=> opponent adjacent
                goals += n.score != s.score
                s = n
            self.assertEqual(s.turn, G.max_turns)

    def test_every_legal_move_is_executable_and_distinct(self):
        rng = random.Random(5)
        for _ in range(300):
            s = random_state(rng)
            if G.is_terminal(s):
                continue
            legal = G.legal_moves(s)
            self.assertEqual(len(legal), len(set(legal)))
            for m in legal:
                check_invariants(G.apply(s, m))

    def test_move_generation_is_symmetric_between_teams(self):
        rng = random.Random(9)
        for _ in range(200):
            s = random_state(rng)
            if not G.is_terminal(s):
                self.assertEqual(len(G.legal_moves(s)), len(G.legal_moves(mirror(s))))


class TestAgents(unittest.TestCase):
    def test_heuristics_are_zero_sum_and_mirror_symmetric(self):
        rng = random.Random(3)
        for name, h in HEURISTICS.items():
            for _ in range(200):
                s = random_state(rng)
                self.assertAlmostEqual(h(s, A), -h(s, B), msg=name)
                self.assertAlmostEqual(h(s, A), h(mirror(s), B), msg=name)

    def test_alphabeta_equals_minimax_but_visits_fewer_nodes(self):
        rng = random.Random(11)
        for name in HEURISTICS:
            for _ in range(15):
                s = random_state(rng, rng.randint(2, 30))
                for depth in (1, 2, 3):
                    mm = MinimaxAgent(depth, name, shuffle=False)
                    ab = AlphaBetaAgent(depth, name, shuffle=False)
                    abo = AlphaBetaAgent(depth, name, shuffle=False, ordering=True)
                    m1, m2, m3 = mm.choose(G, s), ab.choose(G, s), abo.choose(G, s)
                    self.assertAlmostEqual(mm.last_value, ab.last_value)
                    self.assertAlmostEqual(mm.last_value, abo.last_value)
                    self.assertEqual(m1, m2)                  # same move (first best is kept)
                    self.assertLessEqual(ab.visited, mm.visited)

    def test_all_agents_return_legal_moves(self):
        rng = random.Random(4)
        specs = [{"type": "random"}, {"type": "greedy", "heuristic": "structure"},
                 {"type": "minimax", "depth": 2}, {"type": "alphabeta", "depth": 3, "heuristic": "structure"},
                 {"type": "mcts", "iterations": 60, "rollout_depth": 8}]
        for spec in specs:
            agent = build_agent(spec, 1)
            for _ in range(8):
                s = random_state(rng)
                if not G.is_terminal(s):
                    self.assertIn(agent.act(G, s), G.legal_moves(s), spec)

    def test_search_takes_a_free_goal_and_blocks_a_threat(self):
        s = scenario(a={2: (4, 0)}, b={4: (5, 3)}, ball=2)             # open shot for A
        for agent in (GreedyAgent("field"), AlphaBetaAgent(2, "field"), AlphaBetaAgent(3, "structure")):
            self.assertEqual(agent.choose(G, s)[0], SHOOT, agent.name)
        # A holds ball in the zone, keeper two columns away; B (to move) must step the keeper over.
        s = scenario(a={2: (4, 0)}, b={0: (3, 4), 1: (3, 3), 2: (4, 4), 3: (4, 3), 4: (5, 2)}, ball=2, to_move=B)
        m = AlphaBetaAgent(2, "field").choose(G, s)
        self.assertEqual((m[0], m[1]), (MOVE, 9))
        self.assertEqual(gm.COL[m[2]], 1)
        # MCTS also finds the free goal
        s = scenario(a={2: (4, 0)}, b={4: (5, 3)}, ball=2)
        self.assertEqual(MCTSAgent(300, 6, seed=1).choose(G, s)[0], SHOOT)

    def test_games_are_reproducible_for_a_seed(self):
        spec_a = {"type": "alphabeta", "depth": 2}
        spec_b = {"type": "mcts", "iterations": 40, "rollout_depth": 6}
        r1 = play_game(G, build_agent(spec_a, 1), build_agent(spec_b, 2), seed=3, record=True)
        r2 = play_game(G, build_agent(spec_a, 1), build_agent(spec_b, 2), seed=3, record=True)
        self.assertEqual(r1["log"], r2["log"])

    def test_illegal_move_from_an_agent_is_rejected(self):
        class Cheat(RandomAgent):
            def choose(self, game, state):
                self.last_nodes = 0
                return (DRIBBLE, 0, 0)
        with self.assertRaises(RuntimeError):
            play_game(G, Cheat(), RandomAgent(), opening_plies=0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
