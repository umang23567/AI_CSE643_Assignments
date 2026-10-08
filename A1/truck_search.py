"""
CSE643 Assignment 1 - Truck Loading and Delivery Planning
FILE 1/3: problem model, heuristics and search algorithms.

MODEL (assumptions)
-------------------
* Packages arrive one at a time (sorted by arrival time); each is loaded into a
  truck immediately, so within a truck the load order = arrival order and the
  LAST loaded package is at the FRONT (unloaded first, LIFO stack).
* Trucks all have the same capacity. Any number of empty trucks is available
  (optionally limited by `max_trucks`). Every truck used costs w1.
* A truck is dispatched when its last package has been loaded, i.e. at the
  arrival time of its latest package. Earlier packages in it WAIT.
* Destinations lie on one highway, stop k is k time units from the centre.
  Packages can only be unloaded from the FRONT, so the unloading order is forced:
  front package first, then the next one, and so on. The truck starts at the
  centre (position 0) and drives to each package's stop in that order, so
        travel time of the i-th unloaded package
            = |d_1 - 0| + |d_2 - d_1| + ... + |d_i - d_(i-1)|
  A package with a nearer stop sitting in front of a farther one therefore costs
  real extra driving (the truck has to come back), so no arbitrary penalty
  weight is needed.
* Delivery time of a package = (dispatch time - arrival time) + travel time.
* Objective:  cost = w1 * (#trucks) + w2 * (average delivery time)

Search formulation
------------------
State  = (index of next package, tuple of trucks), a truck = tuple of package
         indices in load order.  Action = put next package into a non-full
         truck or open a new truck.  Goal = all packages assigned.
g(n)   = exact cost accumulated so far (every step adds a non-negative
         increment, so g is a true path cost and g(goal) = final objective).
h(n)   = admissible lower bounds on the remaining cost (see section 2).
"""
import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple


# =====================================================================
# 1. DATA MODEL
# =====================================================================

class Package(NamedTuple):
    id: int            # 1..n in arrival order
    destination: int   # highway stop 1, 2, 3, ...
    arrival_time: int


class State:
    __slots__ = ("next_idx", "trucks", "g", "parent", "action")

    def __init__(self, next_idx, trucks, g, parent=None, action=""):
        self.next_idx = next_idx
        self.trucks = trucks      # Tuple[Tuple[int, ...], ...]
        self.g = g
        self.parent = parent
        self.action = action

    def key(self):
        # Trucks are created lazily (always with the lowest unassigned package),
        # so the tuple order is already canonical: no mirror-image states.
        return (self.next_idx, self.trucks)


@dataclass
class SearchResult:
    goal: Optional[State]
    expanded: int
    generated: int


class Problem:
    def __init__(self, packages: List[Package], capacity: int, w1: float, w2: float,
                 max_trucks: Optional[int] = None, reorder: bool = False):
        # Sort by arrival time and re-number so ids follow arrival order.
        srt = sorted(packages, key=lambda p: (p.arrival_time, p.id))
        self.pkgs = [Package(i + 1, p.destination, p.arrival_time) for i, p in enumerate(srt)]
        self.n = len(self.pkgs)
        self.cap = capacity
        self.w1, self.w2 = w1, w2
        self.w_t = w2 / self.n                  # weight per unit of total delivery time
        self.max_trucks = max_trucks
        self.reorder = reorder                  # variation: buffer lets centre load far->near
        if max_trucks is not None and max_trucks * capacity < self.n:
            raise ValueError("Fleet too small for the number of packages")
        self._cache: Dict[Tuple[int, ...], float] = {}
        self.suffix_dest = [0] * (self.n + 1)   # sum of destinations of pkgs[i:]
        for i in range(self.n - 1, -1, -1):
            self.suffix_dest[i] = self.suffix_dest[i + 1] + self.pkgs[i].destination

    # ---------- truck evaluation ----------
    def truck_loaded(self, truck: Tuple[int, ...]) -> List[Package]:
        pk = [self.pkgs[i] for i in truck]
        if self.reorder:
            pk.sort(key=lambda p: -p.destination)   # farthest at the bottom
        return pk

    def truck_details(self, truck: Tuple[int, ...]):
        """Returns (dispatch_time, {pkg_id: delivery_time}).
        Last of the loaded list = front of truck = unloaded first."""
        pk = self.truck_loaded(truck)
        dispatch = max(p.arrival_time for p in pk)
        times, pos, travel = {}, 0, 0
        for p in reversed(pk):                      # front -> back (forced unload order)
            travel += abs(p.destination - pos)      # drive to this package's stop
            pos = p.destination
            times[p.id] = (dispatch - p.arrival_time) + travel
        return dispatch, times

    def truck_sum(self, truck: Tuple[int, ...]) -> float:
        """Total delivery time of all packages in the truck (cached)."""
        if not truck:
            return 0.0
        v = self._cache.get(truck)
        if v is None:
            v = float(sum(self.truck_details(truck)[1].values()))
            self._cache[truck] = v
        return v

    def full_cost(self, trucks) -> float:
        return self.w1 * len(trucks) + self.w_t * sum(self.truck_sum(t) for t in trucks)

    # ---------- search interface ----------
    def initial(self) -> State:
        return State(0, (), 0.0)

    def is_goal(self, s: State) -> bool:
        return s.next_idx == self.n

    def successors(self, s: State) -> List[State]:
        """Existing non-full trucks first, then (if allowed) a new truck."""
        i = s.next_idx
        pid = self.pkgs[i].id
        out = []
        for t_i, t in enumerate(s.trucks):
            if len(t) < self.cap:
                nt = t + (i,)
                delta = self.truck_sum(nt) - self.truck_sum(t)      # always >= 0
                trucks = s.trucks[:t_i] + (nt,) + s.trucks[t_i + 1:]
                out.append(State(i + 1, trucks, s.g + self.w_t * delta, s, f"P{pid} -> Truck {t_i + 1}"))
        if self.max_trucks is None or len(s.trucks) < self.max_trucks:
            nt = (i,)
            g = s.g + self.w1 + self.w_t * self.truck_sum(nt)
            out.append(State(i + 1, s.trucks + (nt,), g, s, f"P{pid} -> New Truck {len(s.trucks) + 1}"))
        return out


# =====================================================================
# 2. HEURISTICS (all admissible, h(goal)=0)
# =====================================================================
# Cost increments are >= 0, and loading package j adds at least
# w2/n * destination_j (its own travel time), so:

def h_zero(p: Problem, s: State) -> float:
    """h=0  -> A* degenerates to Uniform-Cost Search."""
    return 0.0


def h_time(p: Problem, s: State) -> float:
    """Every remaining package needs >= its destination in travel time."""
    return p.w_t * p.suffix_dest[s.next_idx]


def h_trucks(p: Problem, s: State) -> float:
    """Remaining packages that do not fit in free slots of open trucks need new trucks."""
    rem = p.n - s.next_idx
    free = sum(p.cap - len(t) for t in s.trucks)
    extra = max(0, rem - free)
    return p.w1 * math.ceil(extra / p.cap)


def h_combined(p: Problem, s: State) -> float:
    """Sum of the two (they bound different cost components, so the sum is admissible)."""
    return h_time(p, s) + h_trucks(p, s)


# =====================================================================
# 3. SEARCH ALGORITHMS
# =====================================================================

def solve_bfs(p: Problem, max_nodes: int) -> SearchResult:
    """BFS (goal test at generation). Finds a feasible, NOT necessarily optimal, solution."""
    init = p.initial()
    q, seen = deque([init]), {init.key()}
    exp = gen = 0
    while q and exp < max_nodes:
        cur = q.popleft()
        exp += 1
        for succ in p.successors(cur):
            gen += 1
            if p.is_goal(succ):
                return SearchResult(succ, exp, gen)
            k = succ.key()
            if k not in seen:
                seen.add(k)
                q.append(succ)
    return SearchResult(None, exp, gen)


def solve_dfs(p: Problem, max_nodes: int) -> SearchResult:
    """DFS; tries existing trucks before a new truck (so it behaves like first-fit)."""
    init = p.initial()
    stack, seen = [init], {init.key()}
    exp = gen = 0
    while stack and exp < max_nodes:
        cur = stack.pop()
        exp += 1
        if p.is_goal(cur):
            return SearchResult(cur, exp, gen)
        for succ in reversed(p.successors(cur)):      # first successor is popped first
            gen += 1
            k = succ.key()
            if k not in seen:
                seen.add(k)
                stack.append(succ)
    return SearchResult(None, exp, gen)


def solve_best_first(p: Problem, h: Callable, max_nodes: int,
                     weight: float = 1.0, greedy: bool = False) -> SearchResult:
    """
    One routine for the best-first family:
      greedy=False, weight=1, h=h_zero      -> Uniform-Cost Search (optimal)
      greedy=False, weight=1, admissible h  -> A*  (optimal)
      greedy=False, weight=W>1              -> Weighted A* (<= W * optimal)
      greedy=True                           -> Greedy best-first, priority (h, g)
    Re-opens a state if it is reached again with a lower g.
    """
    def prio(s, hv):
        return (hv, s.g) if greedy else (s.g + weight * hv, hv)

    init = p.initial()
    counter = 0
    h0 = h(p, init)
    pq = [(*prio(init, h0), counter, init)]
    best_g = {init.key(): init.g}
    exp = gen = 0

    while pq and exp < max_nodes:
        *_, cur = heapq.heappop(pq)
        if cur.g > best_g[cur.key()] + 1e-12:
            continue                                   # stale queue entry
        exp += 1
        if p.is_goal(cur):
            return SearchResult(cur, exp, gen)
        for succ in p.successors(cur):
            gen += 1
            k = succ.key()
            if k not in best_g or succ.g < best_g[k] - 1e-12:
                best_g[k] = succ.g
                counter += 1
                heapq.heappush(pq, (*prio(succ, h(p, succ)), counter, succ))
    return SearchResult(None, exp, gen)


def get_algorithms(max_nodes: int) -> Dict[str, Callable[[Problem], SearchResult]]:
    return {
        "BFS": lambda p: solve_bfs(p, max_nodes),
        "DFS": lambda p: solve_dfs(p, max_nodes),
        "UCS (h=0)": lambda p: solve_best_first(p, h_zero, max_nodes),
        "A* (h_time)": lambda p: solve_best_first(p, h_time, max_nodes),
        "A* (h_trucks)": lambda p: solve_best_first(p, h_trucks, max_nodes),
        "A* (h_combined)": lambda p: solve_best_first(p, h_combined, max_nodes),
        "Weighted A* (W=2)": lambda p: solve_best_first(p, h_combined, max_nodes, weight=2.0),
        "Greedy (h_combined)": lambda p: solve_best_first(p, h_combined, max_nodes, greedy=True),
    }


OPTIMAL_ALGOS = ["UCS (h=0)", "A* (h_time)", "A* (h_trucks)", "A* (h_combined)"]


# =====================================================================
# 4. SOLUTION HELPERS
# =====================================================================

def solution_path(goal: State):
    """Yields states from the goal back to (excluding) the initial state."""
    s = goal
    while s.parent is not None:
        yield s
        s = s.parent


def describe_solution(p: Problem, goal: State) -> str:
    lines, all_times = [], []
    for t_i, t in enumerate(goal.trucks, start=1):
        dispatch, times = p.truck_details(t)
        order = [f"P{q.id}(d={q.destination})" for q in p.truck_loaded(t)]
        lines.append(f"  Truck {t_i}: bottom -> front: {' , '.join(order)} | dispatched at t={dispatch}")
        all_times += list(times.values())
    avg = sum(all_times) / len(all_times)
    lines.append(f"  #trucks = {len(goal.trucks)}, avg delivery time = {avg:.2f}, "
                 f"cost = {goal.g:.2f} (= {p.w1} * {len(goal.trucks)} + {p.w2} * avg time)")
    return "\n".join(lines)


def solution_stats(p: Problem, goal: State) -> Tuple[int, float]:
    """(#trucks, average delivery time) of a solution."""
    tot = sum(p.truck_sum(t) for t in goal.trucks)
    return len(goal.trucks), tot / p.n
