"""
Robo-Football  -  FILE 3/6: the players (Part 2) and the match runner.

Every agent has the same interface and talks to the game ONLY through the public `Game`
API (legal_moves / apply / is_terminal), so agents are independent of each other and of
the rule implementation:

    agent.choose(game, state) -> move

Agents:  RandomAgent, GreedyAgent, MinimaxAgent (no pruning), AlphaBetaAgent (pruning),
         MCTSAgent.   Search agents take a heuristic by name ("field" / "structure").

Search value convention: every value is from the point of view of the ROOT team (the team
that is choosing now).  A node where the root team moves is a MAX node, the other a MIN node.
Depth is counted in PLIES (one team's turn = one ply).

Node accounting (what the experiments report):
    visited   = every state the search touches (a leaf evaluation also counts)
    expanded  = states whose children were generated (internal nodes)
"""
import math
import random
import time

from game import (ADV, DRIBBLE, MOVE, PASS, SHOOT, Game)
from heuristics import HEURISTICS, terminal_value

INF = float("inf")


class Agent:
    name = "agent"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.moves = 0          # cumulative statistics over the whole match
        self.time = 0.0
        self.nodes = 0          # "positions generated or visited" - comparable across agents
        self.last_nodes = 0

    def act(self, game, state):
        """Called by the match runner: times the decision and records the search effort."""
        t0 = time.perf_counter()
        m = self.choose(game, state)
        self.time += time.perf_counter() - t0
        self.moves += 1
        self.nodes += self.last_nodes
        return m

    def choose(self, game, state):
        raise NotImplementedError


# ----------------------------------------------------------------------------------------
# 1. Random baseline
# ----------------------------------------------------------------------------------------
class RandomAgent(Agent):
    name = "Random"

    def choose(self, game, state):
        self.last_nodes = 0
        return self.rng.choice(game.legal_moves(state))


# ----------------------------------------------------------------------------------------
# 2. Greedy (no look-ahead): best heuristic value right after MY move
# ----------------------------------------------------------------------------------------
class GreedyAgent(Agent):
    def __init__(self, heuristic="field", seed=0):
        super().__init__(seed)
        self.h = HEURISTICS[heuristic]
        self.name = f"Greedy({heuristic})"

    def choose(self, game, state):
        me = state.to_move
        moves = game.legal_moves(state)
        self.rng.shuffle(moves)                      # random tie-breaking
        best, best_v = moves[0], -INF
        for m in moves:
            child = game.apply(state, m)
            v = terminal_value(child, me) if game.is_terminal(child) else self.h(child, me)
            if v > best_v:
                best, best_v = m, v
        self.last_nodes = len(moves)
        return best


# ----------------------------------------------------------------------------------------
# 3 + 4. Depth-limited Minimax and Alpha-Beta (same code path, same heuristic)
# ----------------------------------------------------------------------------------------
def move_order_key(state, move):
    """Cheap static ordering used by 'alpha-beta + ordering': most promising moves first.
    shots, then forward passes, then forward dribbles, then forward repositioning."""
    kind, p, x = move
    t = state.to_move
    if kind == SHOOT:
        return -100
    if kind == PASS:
        return -50 - ADV[t][state.pos[x]]
    if kind == DRIBBLE:
        return -20 - ADV[t][x]
    if kind == MOVE:
        return -(ADV[t][x] - ADV[t][state.pos[p]])
    return 0


class MinimaxAgent(Agent):
    prune = False

    def __init__(self, depth=3, heuristic="field", seed=0, ordering=False, shuffle=True):
        super().__init__(seed)
        self.depth, self.ordering, self.shuffle = depth, ordering, shuffle
        self.h = HEURISTICS[heuristic]
        tag = "AB" if self.prune else "MM"
        self.name = f"{tag}(d={depth},{heuristic}{',ord' if ordering else ''})"
        self.visited = self.expanded = 0
        self.last_value = 0.0

    # -- evaluation at the horizon or at the end of the match
    def _leaf(self, s):
        if self.game.is_terminal(s):
            return terminal_value(s, self.root)
        return self.h(s, self.root)

    def _order(self, s, moves):
        return sorted(moves, key=lambda m: move_order_key(s, m))     # stable sort

    # -- plain minimax: looks at EVERY leaf of the depth-limited tree
    def _minimax(self, s, depth):
        self.visited += 1
        if depth == 0 or self.game.is_terminal(s):
            return self._leaf(s)
        self.expanded += 1
        game = self.game
        moves = game.legal_moves(s)
        if self.ordering:
            moves = self._order(s, moves)
        if s.to_move == self.root:                   # MAX node
            best = -INF
            for m in moves:
                v = self._minimax(game.apply(s, m), depth - 1)
                if v > best:
                    best = v
            return best
        best = INF                                   # MIN node
        for m in moves:
            v = self._minimax(game.apply(s, m), depth - 1)
            if v < best:
                best = v
        return best

    # -- alpha-beta: identical result, but skips branches that cannot change the answer
    def _alphabeta(self, s, depth, alpha, beta):
        self.visited += 1
        if depth == 0 or self.game.is_terminal(s):
            return self._leaf(s)
        self.expanded += 1
        game = self.game
        moves = game.legal_moves(s)
        if self.ordering:
            moves = self._order(s, moves)
        if s.to_move == self.root:                   # MAX node: raises alpha
            best = -INF
            for m in moves:
                v = self._alphabeta(game.apply(s, m), depth - 1, alpha, beta)
                if v > best:
                    best = v
                if best > alpha:
                    alpha = best
                if alpha >= beta:                    # MIN above will never allow this: prune
                    break
            return best
        best = INF                                   # MIN node: lowers beta
        for m in moves:
            v = self._alphabeta(game.apply(s, m), depth - 1, alpha, beta)
            if v < best:
                best = v
            if best < beta:
                beta = best
            if alpha >= beta:                        # MAX above already has something better: prune
                break
        return best

    def choose(self, game, state):
        self.game, self.root = game, state.to_move
        self.visited = self.expanded = 1             # the root itself
        moves = game.legal_moves(state)
        if self.shuffle:
            self.rng.shuffle(moves)                  # random tie-breaking between equal moves
        if self.ordering:
            moves = self._order(state, moves)
        best_m, best_v, alpha = moves[0], -INF, -INF
        for m in moves:
            child = game.apply(state, m)
            if self.prune:
                v = self._alphabeta(child, self.depth - 1, alpha, INF)
                if v > alpha:
                    alpha = v
            else:
                v = self._minimax(child, self.depth - 1)
            if v > best_v:                            # strict '>' : first best move is kept
                best_v, best_m = v, m
        self.last_value, self.last_nodes = best_v, self.visited
        return best_m


class AlphaBetaAgent(MinimaxAgent):
    prune = True


# ----------------------------------------------------------------------------------------
# 6. Monte-Carlo Tree Search (UCT) with depth-limited random roll-outs
# ----------------------------------------------------------------------------------------
class _Node:
    __slots__ = ("state", "parent", "move", "children", "untried", "visits", "reward")

    def __init__(self, state, parent, move, untried):
        self.state, self.parent, self.move = state, parent, move
        self.children, self.untried = [], untried
        self.visits, self.reward = 0, 0.0     # reward: sum of results for the team that MOVED INTO this node


class MCTSAgent(Agent):
    """UCT: repeat  select -> expand one child -> random roll-out -> back-propagate.

    * Selection uses UCB1:  mean_reward + c * sqrt(ln(N_parent) / N_child).
    * A roll-out plays for at most `rollout_depth` plies (or until the match ends).  Policy
      "random": uniformly random legal moves.  Policy "guided": with probability 1-epsilon the
      move with the best heuristic value one ply ahead (a greedy move), else random.
      If it stops early, the state is scored by the heuristic squashed into [-1, 1] with tanh,
      so goals still dominate.
    * The move with the MOST VISITS at the root is played.
    """

    def __init__(self, iterations=400, rollout_depth=20, c=1.0, heuristic="field", seed=0,
                 scale=60.0, rollout_policy="random", epsilon=0.3):
        super().__init__(seed)
        self.iterations, self.rollout_depth, self.c, self.scale = iterations, rollout_depth, c, scale
        self.policy, self.epsilon = rollout_policy, epsilon
        self.h = HEURISTICS[heuristic]
        self.name = f"MCTS(it={iterations},rd={rollout_depth}{',guided' if rollout_policy == 'guided' else ''})"

    def _rollout(self, game, s, root_team):
        d = 0
        guided = self.policy == "guided"
        while d < self.rollout_depth and not game.is_terminal(s):
            moves = game.legal_moves(s)
            if guided and self.rng.random() >= self.epsilon:
                me, best, best_v = s.to_move, moves[0], -INF
                for m in moves:
                    c = game.apply(s, m)
                    v = terminal_value(c, me) if game.is_terminal(c) else self.h(c, me)
                    if v > best_v:
                        best, best_v = m, v
                self.generated += len(moves)
                s = game.apply(s, best)
            else:
                s = game.apply(s, self.rng.choice(moves))
            d += 1
        self.generated += d
        v = terminal_value(s, root_team) if game.is_terminal(s) else self.h(s, root_team)
        return math.tanh(v / self.scale)                      # in [-1, 1], root-team point of view

    def choose(self, game, state):
        root_team = state.to_move
        root_moves = game.legal_moves(state)
        self.generated = 0
        if len(root_moves) == 1:
            self.last_nodes = 0
            return root_moves[0]
        root = _Node(state, None, None, list(root_moves))
        rng = self.rng
        for _ in range(self.iterations):
            node = root
            while not node.untried and node.children:         # 1. selection (UCB1)
                ln = math.log(node.visits)
                node = max(node.children, key=lambda ch: ch.reward / ch.visits +
                           self.c * math.sqrt(ln / ch.visits))
            if node.untried:                                  # 2. expansion
                m = node.untried.pop(rng.randrange(len(node.untried)))
                cs = game.apply(node.state, m)
                self.generated += 1
                child = _Node(cs, node, m, [] if game.is_terminal(cs) else game.legal_moves(cs))
                node.children.append(child)
                node = child
            v = self._rollout(game, node.state, root_team)    # 3. roll-out
            while node is not None:                           # 4. back-propagation
                node.visits += 1
                mover = 1 - node.state.to_move                # team that moved into this node
                node.reward += ((v if mover == root_team else -v) + 1.0) / 2.0   # map [-1,1] -> [0,1]
                node = node.parent
        self.last_nodes = self.generated
        best = max(root.children, key=lambda ch: ch.visits)
        return best.move


# ----------------------------------------------------------------------------------------
# Agent factory (an agent is described by a small dict so it can be logged / re-created)
# ----------------------------------------------------------------------------------------
def build_agent(spec: dict, seed: int = 0) -> Agent:
    t = spec["type"]
    h = spec.get("heuristic", "field")
    if t == "random":
        return RandomAgent(seed)
    if t == "greedy":
        return GreedyAgent(h, seed)
    if t == "minimax":
        return MinimaxAgent(spec["depth"], h, seed, spec.get("ordering", False), spec.get("shuffle", True))
    if t == "alphabeta":
        return AlphaBetaAgent(spec["depth"], h, seed, spec.get("ordering", False), spec.get("shuffle", True))
    if t == "mcts":
        return MCTSAgent(spec.get("iterations", 400), spec.get("rollout_depth", 20), spec.get("c", 1.0), h, seed,
                         rollout_policy=spec.get("policy", "random"))
    raise ValueError(t)


def spec_label(spec: dict) -> str:
    t, h = spec["type"], spec.get("heuristic", "field")
    if t == "random":
        return "Random"
    if t == "greedy":
        return f"Greedy({h})"
    if t in ("minimax", "alphabeta"):
        return f"{'AB' if t == 'alphabeta' else 'MM'}(d={spec['depth']},{h}{',ord' if spec.get('ordering') else ''})"
    return (f"MCTS(it={spec.get('iterations', 400)},rd={spec.get('rollout_depth', 20)}"
            f"{',guided' if spec.get('policy') == 'guided' else ''})")


# ----------------------------------------------------------------------------------------
# Match runner
# ----------------------------------------------------------------------------------------
def new_stats():
    return {"possession": 0, "passes": 0, "dribbles": 0, "shots": 0, "goals": 0,
            "shots_saved": 0, "tackled": 0, "defensive": 0}


def play_game(game: Game, agent_a: Agent, agent_b: Agent, seed: int = 0, opening_plies: int = 4,
              record: bool = False):
    """Play one match.  The first `opening_plies` plies are uniformly random (same for a given
    seed) so that repeated games between deterministic agents are not identical.
    Every move is checked for legality by the environment, not trusted from the agent.
    Returns the score plus per-team event counts (used to describe each agent's playing style)."""
    rng = random.Random(f"opening-{seed}")
    agents = (agent_a, agent_b)
    s = game.initial_state()
    log = []
    stats = (new_stats(), new_stats())
    ply = 0
    while not game.is_terminal(s):
        legal = game.legal_moves(s)
        if ply < opening_plies:
            m = rng.choice(legal)
        else:
            m = agents[s.to_move].act(game, s)
            if m not in legal:
                raise RuntimeError(f"illegal move {m} from {agents[s.to_move].name}")
        nxt = game.apply(s, m)
        me, kind = stats[s.to_move], m[0]
        stats[s.ball // 5]["possession"] += 1                 # turns that start with the ball
        event = ""
        if kind == PASS:
            me["passes"] += 1
        elif kind == DRIBBLE:
            me["dribbles"] += 1
            if nxt.ball // 5 != s.ball // 5:
                me["tackled"] += 1
                event = "  (tackled)"
        elif kind == SHOOT:
            me["shots"] += 1
            if nxt.score != s.score:
                me["goals"] += 1
                event = "  *** GOAL ***"
            else:
                me["shots_saved"] += 1
                event = "  (saved)"
        elif kind == MOVE:
            me["defensive"] += 1
        if record:
            log.append(f"{'AB'[s.to_move]} {game.describe(s, m)}{event}")
            log.append(game.render(nxt))
        s = nxt
        ply += 1
    return {"score": s.score, "turns": s.turn, "winner": game.winner(s), "log": log,
            "final": s, "stats": stats}
