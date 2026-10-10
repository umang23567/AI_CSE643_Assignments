"""
Robo-Football  -  FILE 1/6: the game environment (Part 1).

Everything about the RULES lives here. Agents never look inside the rules; they only use
the small public API of `Game`:

    game.initial_state()          -> State
    game.legal_moves(state)       -> list of moves
    game.apply(state, move)       -> NEW State (states are never mutated)
    game.is_terminal(state)       -> bool
    game.winner(state)            -> 0 (team A), 1 (team B) or None (draw)
    game.render(state)            -> printable board
    game.describe(state, move)    -> human readable move text

DESIGN DECISIONS (the spec leaves these open, so they are documented here)
--------------------------------------------------------------------------
Board      6 rows (0-5) x 5 columns (0-4). Cell index = row*5 + col.
           Team A defends row 0 and attacks row 5; Team B defends row 5, attacks row 0.
Pieces     10 pieces. Index 0-3 = A outfield, 4 = A goalkeeper, 5-8 = B outfield, 9 = B GK.
Formation  Team A: GK (0,2); outfield A1 (1,1)  A2 (1,3)  A3 (2,1)  A4 (2,3).  Team B is the
           mirror image (row r -> 5-r): B1 (4,1)  B2 (4,3)  B3 (3,1)  B4 (3,3), keeper (5,2).
           Two defenders + two midfielders each; the midfield lines touch at kickoff so
           tackles, passing lanes and blocking matter from the very first move (I tried a
           'gap' formation with empty rows 2-3: it gave almost only 0-0 draws).
           A kicks off: the ball is with A3 at (2,1) and A moves first.  After a goal
           everything resets to this formation and the team that CONCEDED gets the ball
           (A3 or B3) and moves next.
Turns      Teams strictly alternate (A, B, A, B ...) whatever the possession.
           A match lasts MAX_TURNS = 80 total turns (40 per team).  A goal scored on the
           last turn still counts.
Tackle     Only a DRIBBLE can lose the ball.  After the carrier moves, every opposing piece
           (outfield or keeper) in the 8 cells around the new cell is a tackler.  If there are
           several, the one standing in the LOWEST cell index (lowest row, then lowest
           column) wins the ball.  The dribbler stays where it moved.
Defending  A team without the ball moves exactly one piece one step (8 directions) into an
           empty cell.  Walking next to the carrier does NOT win the ball (the rules only
           give a tackle on a dribble).  The keeper moves sideways along its goal line only.
Pass       To any teammate (keeper included) on the same row / column / exact diagonal,
           unless an OPPOSING piece stands strictly between them.  Teammates do not block.
Shoot      Outfield carrier standing in the two rows nearest the opponent goal.  Goal unless
           the opposing keeper is within 1 column (|col diff| <= 1): then it is saved and the
           keeper takes the ball (keeper stays where it is).
Keeper     When the keeper HOLDS the ball it may pass, or dribble sideways along its goal
           line (it never leaves the line).  It cannot shoot.
No moves   If a team has no legal move at all (practically never happens) it must SKIP.
"""
from typing import List, Optional, Tuple

# ----------------------------------------------------------------------------------------
# Board geometry (pre-computed once; all agents and heuristics share these tables)
# ----------------------------------------------------------------------------------------
ROWS, COLS = 6, 5
NCELLS = ROWS * COLS
A, B = 0, 1
ATTACK_ADV = ROWS - 2          # advancement >= 4  <=>  inside the attacking third (2 rows)

ROW = [i // COLS for i in range(NCELLS)]
COL = [i % COLS for i in range(NCELLS)]
DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def cell(r: int, c: int) -> int:
    return r * COLS + c


def _in(r, c):
    return 0 <= r < ROWS and 0 <= c < COLS


NEIGH = [tuple(cell(ROW[i] + dr, COL[i] + dc) for dr, dc in DIRS if _in(ROW[i] + dr, COL[i] + dc))
         for i in range(NCELLS)]
NEIGH_SET = [frozenset(n) for n in NEIGH]
SIDE = [tuple(cell(ROW[i], COL[i] + dc) for dc in (-1, 1) if _in(ROW[i], COL[i] + dc))
        for i in range(NCELLS)]                      # sideways neighbours (keeper moves)
CHEB = [[max(abs(ROW[a] - ROW[b]), abs(COL[a] - COL[b])) for b in range(NCELLS)] for a in range(NCELLS)]
# ADV[team][cell] = how far the cell is from the team's OWN goal line = distance advanced
ADV = [[ROW[i] for i in range(NCELLS)], [ROWS - 1 - ROW[i] for i in range(NCELLS)]]


def _between(a: int, b: int):
    """Cells strictly between a and b if they share a row/column/diagonal, else None."""
    if a == b:
        return None
    dr, dc = ROW[b] - ROW[a], COL[b] - COL[a]
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        return None
    steps = max(abs(dr), abs(dc))
    sr, sc = (dr > 0) - (dr < 0), (dc > 0) - (dc < 0)
    return tuple(cell(ROW[a] + k * sr, COL[a] + k * sc) for k in range(1, steps))


BETWEEN = [[_between(a, b) for b in range(NCELLS)] for a in range(NCELLS)]

# ----------------------------------------------------------------------------------------
# Kickoff formation
# ----------------------------------------------------------------------------------------
KICKOFF_A = (cell(1, 1), cell(1, 3), cell(2, 1), cell(2, 3), cell(0, 2))
KICKOFF_B = tuple(cell(ROWS - 1 - ROW[c], COL[c]) for c in KICKOFF_A)
KICKOFF_POS = KICKOFF_A + KICKOFF_B
KICKOFF_CARRIER = (2, 7)             # piece that holds the ball at a kickoff, per team (A3 / B3)

# Move kinds.  A move is a tuple (kind, piece, target).
MOVE, DRIBBLE, PASS, SHOOT, SKIP = range(5)
MAX_TURNS = 80


def team_of(piece: int) -> int:
    return piece // 5


def piece_name(piece: int) -> str:
    return "AB"[piece // 5] + ("G" if piece % 5 == 4 else str(piece % 5 + 1))


def cell_name(c: int) -> str:
    return f"({ROW[c]},{COL[c]})"


class State:
    """Immutable game state.

    pos      tuple of 10 cell indices (see piece numbering above)
    ball     index (0-9) of the piece holding the ball
    to_move  0 = team A, 1 = team B
    turn     number of turns already played
    score    (goals A, goals B)
    """
    __slots__ = ("pos", "ball", "to_move", "turn", "score")

    def __init__(self, pos, ball, to_move, turn, score):
        self.pos, self.ball, self.to_move, self.turn, self.score = pos, ball, to_move, turn, score

    def key(self):
        return (self.pos, self.ball, self.to_move, self.turn, self.score)

    def __eq__(self, other):
        return isinstance(other, State) and self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def has_ball(self, team: int) -> bool:
        return self.ball // 5 == team


# ----------------------------------------------------------------------------------------
# The game
# ----------------------------------------------------------------------------------------
class Game:
    def __init__(self, max_turns: int = MAX_TURNS):
        self.max_turns = max_turns

    # ---- construction -------------------------------------------------------------
    def initial_state(self, kickoff_team: int = A) -> State:
        return State(KICKOFF_POS, KICKOFF_CARRIER[kickoff_team], kickoff_team, 0, (0, 0))

    @staticmethod
    def make_state(a_cells, b_cells, ball: int, to_move: int = A, turn: int = 0,
                   score=(0, 0)) -> State:
        """Build any position, e.g. for tests or experiments.
        a_cells / b_cells: five (row, col) pairs each (4 outfield players, then the keeper)."""
        pos = tuple(cell(r, c) for r, c in list(a_cells) + list(b_cells))
        s = State(pos, ball, to_move, turn, tuple(score))
        check_invariants(s)
        return s

    # ---- queries --------------------------------------------------------------------
    def is_terminal(self, s: State) -> bool:
        return s.turn >= self.max_turns

    @staticmethod
    def winner(s: State) -> Optional[int]:
        if s.score[0] == s.score[1]:
            return None
        return A if s.score[0] > s.score[1] else B

    def legal_moves(self, s: State) -> List[Tuple[int, int, int]]:
        if s.turn >= self.max_turns:
            return []
        t = s.to_move
        pos = s.pos
        base = 5 * t
        occ = set(pos)
        moves = []
        if s.ball // 5 == t:                                     # our team holds the ball
            c = s.ball
            pc = pos[c]
            for d in (SIDE[pc] if c == base + 4 else NEIGH[pc]):  # keeper dribbles sideways only
                if d not in occ:
                    moves.append((DRIBBLE, c, d))
            ob = 5 * (1 - t)
            opp = set(pos[ob:ob + 5])
            for j in range(base, base + 5):
                if j == c:
                    continue
                btw = BETWEEN[pc][pos[j]]
                if btw is not None and opp.isdisjoint(btw):
                    moves.append((PASS, c, j))
            if c != base + 4 and ADV[t][pc] >= ATTACK_ADV:
                moves.append((SHOOT, c, 0))
        else:                                                    # defensive repositioning
            for p in range(base, base + 4):
                for d in NEIGH[pos[p]]:
                    if d not in occ:
                        moves.append((MOVE, p, d))
            for d in SIDE[pos[base + 4]]:
                if d not in occ:
                    moves.append((MOVE, base + 4, d))
        if not moves:
            moves.append((SKIP, 0, 0))
        return moves

    def apply(self, s: State, move) -> State:
        """Return the state after `move`. The input state is not modified."""
        kind, p, x = move
        t = s.to_move
        pos, ball, score = s.pos, s.ball, s.score
        if kind == MOVE:
            pos = pos[:p] + (x,) + pos[p + 1:]
        elif kind == DRIBBLE:
            pos = pos[:p] + (x,) + pos[p + 1:]
            adj = NEIGH_SET[x]
            ob = 5 * (1 - t)
            tackler, best_cell = -1, NCELLS
            for j in range(ob, ob + 5):
                pj = pos[j]
                if pj in adj and pj < best_cell:                  # tie-break: lowest cell index
                    tackler, best_cell = j, pj
            if tackler >= 0:
                ball = tackler
        elif kind == PASS:
            ball = x
        elif kind == SHOOT:
            opp_gk = 5 * (1 - t) + 4
            if abs(COL[pos[opp_gk]] - COL[pos[p]]) <= 1:          # saved: keeper takes the ball
                ball = opp_gk
            else:                                                 # goal: reset, conceder restarts
                score = (score[0] + 1, score[1]) if t == A else (score[0], score[1] + 1)
                pos = KICKOFF_POS
                ball = KICKOFF_CARRIER[1 - t]
        # kind == SKIP: nothing changes except the turn
        return State(pos, ball, 1 - t, s.turn + 1, score)

    # ---- presentation ---------------------------------------------------------------
    def describe(self, s: State, move) -> str:
        kind, p, x = move
        if kind == MOVE:
            return f"{piece_name(p)} moves {cell_name(s.pos[p])}->{cell_name(x)}"
        if kind == DRIBBLE:
            return f"{piece_name(p)} dribbles {cell_name(s.pos[p])}->{cell_name(x)}"
        if kind == PASS:
            return f"{piece_name(p)} passes to {piece_name(x)}"
        if kind == SHOOT:
            return f"{piece_name(p)} shoots from {cell_name(s.pos[p])}"
        return "skips"

    def render(self, s: State) -> str:
        labels = {}
        for i, c in enumerate(s.pos):
            labels[c] = piece_name(i) + ("*" if i == s.ball else "")
        head = (f"Turn {s.turn}/{self.max_turns} | {'AB'[s.to_move]} to move | "
                f"Score A {s.score[0]} - {s.score[1]} B | ball: {piece_name(s.ball)}")
        lines = [head, "       " + "".join(f"c{c}".center(5) for c in range(COLS))]
        for r in range(ROWS - 1, -1, -1):
            row = "".join(labels.get(cell(r, c), ".").center(5) for c in range(COLS))
            note = "  <- B's goal line (A attacks up)" if r == ROWS - 1 else (
                "  <- A's goal line" if r == 0 else "")
            lines.append(f"  r{r}   {row}{note}")
        return "\n".join(lines)


def mirror(s: State) -> State:
    """Swap the two teams and flip the board vertically (used to test heuristic fairness)."""
    flip = lambda c: cell(ROWS - 1 - ROW[c], COL[c])
    pos = tuple(flip(s.pos[(j + 5) % 10]) for j in range(10))
    return State(pos, (s.ball + 5) % 10, 1 - s.to_move, s.turn, (s.score[1], s.score[0]))


def check_invariants(s: State) -> None:
    """Raises AssertionError if the state breaks a basic rule of the game."""
    assert len(s.pos) == 10, "need 10 pieces"
    assert len(set(s.pos)) == 10, f"two pieces share a cell: {s.pos}"
    assert all(0 <= c < NCELLS for c in s.pos), "piece off the board"
    assert 0 <= s.ball < 10, "ball holder must be a piece"
    assert ROW[s.pos[4]] == 0, "team A keeper must stay on row 0"
    assert ROW[s.pos[9]] == ROWS - 1, "team B keeper must stay on row 5"
    assert s.to_move in (A, B)
    assert 0 <= s.turn
