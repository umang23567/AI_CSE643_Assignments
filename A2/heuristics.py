"""
Robo-Football  -  FILE 2/6: evaluation heuristics.

A heuristic is a function  h(state, team) -> float  = how good `state` is for `team`.
All heuristics here are ZERO-SUM by construction:  h(s, A) = - h(s, B).

    h(s, team) = 100 * (goals_team - goals_opp)  +  side(s, team) - side(s, opp)

Goals dominate everything (one goal = 100, positional terms are worth a few points each),
so a heuristic can never prefer "nice shape" over actually scoring.

Two different ideas about what matters:

  FIELD      "territory and shot threat"
             Reward advancing the ball and the team up the pitch, and reward a carrier who
             stands in the attacking third with an open shot.  It knows nothing about passing
             lanes or defending, so it plays direct, forward football.

  STRUCTURE  "passing network and defensive shape"
             With the ball: count open passing lanes, support players nearby, and dribble
             squares that are safe from a tackle.  Without the ball: count defenders that are
             goal-side of the carrier, enemy passing lanes we block, players marking the
             carrier, and whether our keeper covers the carrier's shooting column.
"""
from game import ADV, ATTACK_ADV, BETWEEN, CHEB, COL, NEIGH, NEIGH_SET

GOAL_VALUE = 100.0
WIN_BONUS = 1000.0


def terminal_value(s, team: int) -> float:
    """Value of a finished match for `team`: a win/loss dwarfs any heuristic score."""
    d = s.score[team] - s.score[1 - team]
    return WIN_BONUS * ((d > 0) - (d < 0)) + GOAL_VALUE * d


# ----------------------------------------------------------------------------------------
# Heuristic 1: FIELD POSITION
# ----------------------------------------------------------------------------------------
W_POSSESSION = 6.0     # having the ball at all
W_CARRIER_ADV = 4.0    # per row the carrier has advanced (0..5)
W_TEAM_ADV = 0.5       # per row each outfield player has advanced
W_SHOT_ZONE = 15.0     # carrier in the attacking third (a shot is possible)
W_OPEN_SHOT = 25.0     # ... and the opposing keeper is NOT within 1 column (shot = goal)
W_GK_DIST = 4.0        # ... plus per column of distance to the keeper (it can only step 1 column per turn)


def _field_side(s, t: int) -> float:
    pos, base, adv = s.pos, 5 * t, ADV[t]
    v = W_TEAM_ADV * (adv[pos[base]] + adv[pos[base + 1]] + adv[pos[base + 2]] + adv[pos[base + 3]])
    if s.ball // 5 == t:
        c = s.ball
        pc = pos[c]
        v += W_POSSESSION + W_CARRIER_ADV * adv[pc]
        if c != base + 4 and adv[pc] >= ATTACK_ADV:
            gap = abs(COL[pos[5 * (1 - t) + 4]] - COL[pc])
            v += W_SHOT_ZONE + W_GK_DIST * gap
            if gap > 1:
                v += W_OPEN_SHOT
    return v


def h_field(s, team: int) -> float:
    d = s.score[team] - s.score[1 - team]
    return GOAL_VALUE * d + _field_side(s, team) - _field_side(s, 1 - team)


# ----------------------------------------------------------------------------------------
# Heuristic 2: PASSING NETWORK & DEFENSIVE STRUCTURE
# ----------------------------------------------------------------------------------------
S_POSSESSION = 6.0
S_CARRIER_ADV = 2.0    # smaller than FIELD: progress matters less than shape here
S_PASS_LANE = 3.0      # per open pass to a teammate (max 4 exist)
S_SUPPORT = 1.0        # per teammate within 2 cells of the carrier
S_SAFE_DRIBBLE = 1.5   # per empty neighbouring cell where a dribble would NOT be tackled
S_SHOT_ZONE = 12.0
S_OPEN_SHOT = 30.0
S_RECEIVER = 8.0       # teammate in the shooting zone that we can pass to
S_RECEIVER_OPEN = 20.0  # ... and the keeper is NOT within 1 column of that teammate (pass-then-shoot)
D_GOAL_SIDE = 2.0      # per defender between the carrier and our goal
D_LANE_BLOCK = 2.5     # per enemy passing lane that one of our players blocks
D_MARK = 2.0           # per player adjacent to the carrier (max 3): any dribble next to us is a tackle
D_GK_COVER = 15.0      # carrier is in its shooting zone and our keeper is within 1 column


def _struct_side(s, t: int) -> float:
    pos, base, ob = s.pos, 5 * t, 5 * (1 - t)
    carrier = s.ball
    pc = pos[carrier]
    if carrier // 5 == t:                                   # ---- we attack
        v = S_POSSESSION + S_CARRIER_ADV * ADV[t][pc]
        opp = set(pos[ob:ob + 5])
        occ = set(pos)
        lanes = support = 0
        for j in range(base, base + 5):
            if j == carrier:
                continue
            pj = pos[j]
            btw = BETWEEN[pc][pj]
            if btw is not None and opp.isdisjoint(btw):
                lanes += 1
                if j != base + 4 and ADV[t][pj] >= ATTACK_ADV:           # receiver in the shooting zone
                    v += S_RECEIVER
                    if abs(COL[pos[ob + 4]] - COL[pj]) > 1:
                        v += S_RECEIVER_OPEN
            if CHEB[pc][pj] <= 2:
                support += 1
        safe = 0
        for n in NEIGH[pc]:
            if n not in occ and NEIGH_SET[n].isdisjoint(opp):
                safe += 1
        v += S_PASS_LANE * lanes + S_SUPPORT * support + S_SAFE_DRIBBLE * safe
        if carrier != base + 4 and ADV[t][pc] >= ATTACK_ADV:
            v += S_SHOT_ZONE
            if abs(COL[pos[ob + 4]] - COL[pc]) > 1:
                v += S_OPEN_SHOT
        return v
    # ---- we defend (the opponent holds the ball at pc)
    v = 0.0
    own = set(pos[base:base + 5])
    carrier_dist = ADV[t][pc]                               # carrier's distance from OUR goal
    for j in range(base, base + 4):
        if ADV[t][pos[j]] < carrier_dist:
            v += D_GOAL_SIDE
    blocked = 0
    for j in range(ob, ob + 5):
        if j == carrier:
            continue
        btw = BETWEEN[pc][pos[j]]
        if btw is not None and not own.isdisjoint(btw):
            blocked += 1
    v += D_LANE_BLOCK * blocked
    v += D_MARK * min(3, sum(1 for q in own if q in NEIGH_SET[pc]))
    if carrier != ob + 4 and ADV[1 - t][pc] >= ATTACK_ADV and abs(COL[pos[base + 4]] - COL[pc]) <= 1:
        v += D_GK_COVER
    return v


def h_structure(s, team: int) -> float:
    d = s.score[team] - s.score[1 - team]
    return GOAL_VALUE * d + _struct_side(s, team) - _struct_side(s, 1 - team)


# ----------------------------------------------------------------------------------------
# Reference evaluation with NO positional knowledge (used only to measure how many plies of
# look-ahead a position really needs, see the tactics probe in experiment 5)
# ----------------------------------------------------------------------------------------
def h_goals(s, team: int) -> float:
    return GOAL_VALUE * (s.score[team] - s.score[1 - team])


HEURISTICS = {"field": h_field, "structure": h_structure, "goals": h_goals}
MAIN_HEURISTICS = ("field", "structure")     # the two heuristics compared in the experiments
