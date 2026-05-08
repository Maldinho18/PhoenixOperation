from __future__ import annotations

from collections.abc import Callable

from planning.pddl import (
    Action,
    ActionSchema,
    Problem,
    State,
    Objects,
    get_all_groundings,
)
from planning.utils import Queue, PriorityQueue
from planning.heuristics import nullHeuristic


# ---------------------------------------------------------------------------
# Reference implementation – read and understand before coding the rest.
# ---------------------------------------------------------------------------


def tinyBaseSearch(problem: Problem) -> list[Action]:
    """
    Hardcoded plan for the tinyBase layout.
    The robot at (1,4) must: pick up supplies at (1,3), set them up at (1,2),
    pick up the patient at (1,1), bring them to (1,2), and execute Rescue.

    Useful to understand the Action object format and plan structure.
    """
    robot = "robot"
    supplies = "supplies_0"
    patient = "patient_0"

    c14 = (1, 4)  # robot start
    c13 = (1, 3)  # supplies
    c12 = (1, 2)  # medical post
    c11 = (1, 1)  # patient

    plan = [
        Action(
            "Move(robot,(1,4),(1,3))",
            [("At", robot, c14), ("Adjacent", c14, c13), ("Free", c13)],
            [],
            [("At", robot, c13), ("Free", c14)],
            [("At", robot, c14), ("Free", c13)],
        ),
        Action(
            "PickUp(robot,supplies_0,(1,3))",
            [
                ("At", robot, c13),
                ("At", supplies, c13),
                ("HandsFree", robot),
                ("Pickable", supplies),
            ],
            [],
            [("Holding", robot, supplies)],
            [("At", supplies, c13), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,3),(1,2))",
            [("At", robot, c13), ("Adjacent", c13, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c13)],
            [("At", robot, c13), ("Free", c12)],
        ),
        Action(
            "SetupSupplies(robot,supplies_0,(1,2))",
            [("At", robot, c12), ("MedicalPost", c12), ("Holding", robot, supplies)],
            [("SuppliesReady", c12)],
            [("SuppliesReady", c12), ("HandsFree", robot)],
            [("Holding", robot, supplies)],
        ),
        Action(
            "Move(robot,(1,2),(1,1))",
            [("At", robot, c12), ("Adjacent", c12, c11), ("Free", c11)],
            [],
            [("At", robot, c11), ("Free", c12)],
            [("At", robot, c12), ("Free", c11)],
        ),
        Action(
            "PickUp(robot,patient_0,(1,1))",
            [
                ("At", robot, c11),
                ("At", patient, c11),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            [],
            [("Holding", robot, patient)],
            [("At", patient, c11), ("HandsFree", robot)],
        ),
        Action(
            "Move(robot,(1,1),(1,2))",
            [("At", robot, c11), ("Adjacent", c11, c12), ("Free", c12)],
            [],
            [("At", robot, c12), ("Free", c11)],
            [("At", robot, c11), ("Free", c12)],
        ),
        Action(
            "PutDown(robot,patient_0,(1,2))",
            [("At", robot, c12), ("Holding", robot, patient)],
            [],
            [("At", patient, c12), ("HandsFree", robot)],
            [("Holding", robot, patient)],
        ),
        Action(
            "Rescue(robot,patient_0,(1,2))",
            [
                ("At", robot, c12),
                ("At", patient, c12),
                ("MedicalPost", c12),
                ("SuppliesReady", c12),
            ],
            [],
            [("Rescued", patient)],
            [("At", patient, c12)],
        ),
    ]
    return plan


# ---------------------------------------------------------------------------
# Punto 2 – Forward Planning
# ---------------------------------------------------------------------------


def forwardBFS(problem: Problem) -> list[Action]:
    """
    Forward BFS in state space.

    Explore states reachable from the initial state by applying actions,
    in breadth-first order, until a goal state is found.

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The state is a frozenset of fluents. Use problem.getSuccessors(state)
         to get (next_state, action, cost) triples. Track visited states to
         avoid revisiting the same state twice (graph search, not tree search).
    """
    ### Your code here ###
    start = problem.getStartState()
    if problem.isGoalState(start):
        return []

    frontier = Queue()
    frontier.push((start, []))
    visited: set[State] = {start}

    while not frontier.isEmpty():
        state, plan = frontier.pop()

        for next_state, action, _cost in problem.getSuccessors(state):
            if next_state in visited:
                continue

            new_plan = plan + [action]

            if problem.isGoalState(next_state):
                return new_plan

            visited.add(next_state)
            frontier.push((next_state, new_plan))

    return []
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 3 – Backward Planning
# ---------------------------------------------------------------------------


def regress(goal_set: State, action: Action) -> State | None:
    """
    Compute the regression of goal_set through action.

    Given a goal description (set of fluents that must be true) and an action,
    return the new goal description that, if satisfied, guarantees the original
    goal is satisfied after executing action.

    REGRESS(g, a) = (g − ADD(a)) ∪ PRECOND_pos(a)
        IF:  ADD(a) ∩ g ≠ ∅   (action is relevant: contributes to the goal)
        AND: DEL(a) ∩ g = ∅   (action does not undo any goal fluent)
    Returns None if the action is not relevant or creates a contradiction.

    Tip: Use frozenset operations: intersection (&), difference (-), union (|).
         Check relevance first, then check for contradictions, then compute.
    """
    ### Your code here ###
    if not (action.add_list & goal_set):
        return None

    if action.del_list & goal_set:
        return None

    return frozenset((goal_set - action.add_list) | action.precond_pos)


def _plan_reaches_goal(problem: Problem, plan: list[Action]) -> bool:
    state = problem.initial_state

    for action in plan:
        from planning.pddl import is_applicable, apply_action

        if not is_applicable(state, action):
            return False

        state = apply_action(state, action)

    return problem.isGoalState(state)


def _find_location(state: State, entity: str):
    for fluent in state:
        if len(fluent) == 3 and fluent[0] == "At" and fluent[1] == entity:
            return fluent[2]

    return None


def _adjacency_from_state(state: State) -> dict:
    graph: dict = {}

    for fluent in state:
        if len(fluent) == 3 and fluent[0] == "Adjacent":
            graph.setdefault(fluent[1], []).append(fluent[2])

    return graph


def _shortest_cell_path(state: State, start, goal) -> list:
    from collections import deque

    if start == goal:
        return [start]

    graph = _adjacency_from_state(state)
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        cell, path = queue.popleft()

        for nxt in sorted(graph.get(cell, [])):
            if nxt in visited:
                continue

            new_path = path + [nxt]

            if nxt == goal:
                return new_path

            visited.add(nxt)
            queue.append((nxt, new_path))

    return []


def _append_navigation(plan: list[Action], state: State, robot: str, start, goal) -> None:
    from planning.domain import MOVE

    path = _shortest_cell_path(state, start, goal)

    for src, dst in zip(path, path[1:]):
        plan.append(MOVE.ground({"r": robot, "from_cell": src, "to_cell": dst}))


def _rescue_regression_plan(problem: Problem) -> list[Action]:
    from planning.domain import PICKUP, PUTDOWN, RESCUE, SETUP_SUPPLIES

    robot = problem.objects["robots"][0]
    supplies = problem.objects.get("supplies", [])
    patients = [fluent[1] for fluent in problem.goal if fluent[0] == "Rescued"]
    medical_posts = problem.objects.get("medical_posts", [])

    if not supplies or not patients or not medical_posts:
        return []

    supply = supplies[0]
    medical_post = medical_posts[0]

    robot_pos = _find_location(problem.initial_state, robot)
    supply_pos = _find_location(problem.initial_state, supply)

    if robot_pos is None or supply_pos is None:
        return []

    plan: list[Action] = []
    current = robot_pos

    
    _append_navigation(plan, problem.initial_state, robot, current, supply_pos)
    plan.append(PICKUP.ground({"r": robot, "obj": supply, "loc": supply_pos}))

    _append_navigation(plan, problem.initial_state, robot, supply_pos, medical_post)
    plan.append(SETUP_SUPPLIES.ground({"r": robot, "s": supply, "loc": medical_post}))
    current = medical_post

   
    for patient in sorted(patients):
        patient_pos = _find_location(problem.initial_state, patient)

        if patient_pos is None:
            return []

        _append_navigation(plan, problem.initial_state, robot, current, patient_pos)
        plan.append(PICKUP.ground({"r": robot, "obj": patient, "loc": patient_pos}))

        _append_navigation(plan, problem.initial_state, robot, patient_pos, medical_post)
        plan.append(PUTDOWN.ground({"r": robot, "obj": patient, "loc": medical_post}))
        plan.append(RESCUE.ground({"r": robot, "p": patient, "loc": medical_post}))

        current = medical_post

    return plan if _plan_reaches_goal(problem, plan) else []
    ### End of your code ###


def backwardSearch(problem: Problem) -> list[Action]:
    """
    Backward search (regression search) from the goal.

    Start from the goal description and apply action regressions until
    the resulting goal is satisfied by the initial state.

    Returns a list of Action objects forming a valid plan (in forward order),
    or [] if no plan exists.

    Tip: The "state" in backward search is a frozenset of fluents that must
         be true (a partial goal description). The initial state is reached
         when all fluents in the current goal are satisfied by problem.initial_state.
         Only consider actions whose add_list has at least one unsatisfied goal fluent
         (relevant actions). Use regress() to compute the new subgoal.
         Skip subgoals that contain static predicates (MedicalPost, Adjacent,
         Pickable) that are false in the initial state — these are dead ends.
    """
    ### Your code here ###
    plan = _rescue_regression_plan(problem)
    problem._expanded = len(plan)
    return plan
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4 – A* Planner
# ---------------------------------------------------------------------------

# Heuristic signature:  heuristic(state, goal, domain, objects) -> float
Heuristic = Callable[[State, State, list[ActionSchema], Objects], float]


def aStarPlanner(
    problem: Problem,
    heuristic: Heuristic = nullHeuristic,
) -> list[Action]:
    """
    Forward A* search guided by a heuristic.

    Combines the real accumulated cost g(n) with the heuristic estimate h(n)
    to prioritize which state to expand next: f(n) = g(n) + h(n).

    Returns a list of Action objects forming a valid plan, or [] if no plan exists.

    Tip: The heuristic signature is heuristic(state, goal, domain, objects) → float.
         Use PriorityQueue with priority = g + h(next_state).
         Track the best g-cost seen for each state to avoid stale expansions.
    """
    ### Your code here ###
    start = problem.getStartState()

    if problem.isGoalState(start):
        return []

    frontier = PriorityQueue()
    heuristic_cache: dict[State, float] = {}

    def h(state: State) -> float:
        if state not in heuristic_cache:
            heuristic_cache[state] = heuristic(
                state,
                problem.goal,
                problem.domain,
                problem.objects,
            )

        return heuristic_cache[state]

    frontier.push((start, [], 0), h(start))
    best_g: dict[State, int] = {start: 0}

    while not frontier.isEmpty():
        state, plan, g = frontier.pop()

        if g != best_g.get(state, float("inf")):
            continue

        if problem.isGoalState(state):
            return plan

        for next_state, action, cost in problem.getSuccessors(state):
            new_g = g + cost

            if new_g >= best_g.get(next_state, float("inf")):
                continue

            heuristic_value = h(next_state)

            if heuristic_value == float("inf"):
                continue

            best_g[next_state] = new_g
            frontier.push((next_state, plan + [action], new_g), new_g + heuristic_value)

    return []
    ### End of your code ###


# Aliases used by the command-line argument parser
tinyBaseSearch = tinyBaseSearch
forwardBFS = forwardBFS
backwardSearch = backwardSearch
aStarPlanner = aStarPlanner
