from __future__ import annotations

from planning.pddl import (
    Action,
    ActionSchema,
    Objects,
    State,
    get_all_groundings,
    is_applicable,
)


_GROUNDINGS_CACHE: dict[tuple[int, tuple], list[Action]] = {}


def _objects_key(objects: Objects) -> tuple:
    return tuple(
        (name, tuple(values))
        for name, values in sorted(objects.items())
    )


def _grounded_actions(domain: list[ActionSchema], objects: Objects) -> list[Action]:
    key = (id(domain), _objects_key(objects))

    if key not in _GROUNDINGS_CACHE:
        _GROUNDINGS_CACHE[key] = get_all_groundings(domain, objects)

    return _GROUNDINGS_CACHE[key]


def nullHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """Trivial heuristic — always returns 0 (equivalent to uniform-cost search)."""
    return 0


# ---------------------------------------------------------------------------
# Punto 4a – Ignore-Preconditions Heuristic
# ---------------------------------------------------------------------------


def ignorePreconditionsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the number of actions needed to satisfy all goal fluents,
    ignoring all action preconditions.

    With no preconditions, any action can be applied at any time.
    Each action can satisfy all goal fluents in its add_list in one step.
    The minimum number of actions to cover all unsatisfied goal fluents is
    a lower bound on the true plan length → this heuristic is admissible.

    Algorithm (greedy set cover):
      1. Compute unsatisfied = goal − state  (fluents still needed).
      2. Ground all actions ignoring preconditions and collect their add_lists.
      3. Greedily pick the action whose add_list covers the most unsatisfied fluents.
      4. Repeat until all fluents are covered; count the actions used.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    unsatisfied = goal - state
    if not unsatisfied:
        return 0

    action_adds = [
        action.add_list
        for action in _grounded_actions(domain, objects)
        if action.add_list & unsatisfied
    ]

    steps = 0
    remaining = set(unsatisfied)

    while remaining:
        best_adds = max(
            action_adds,
            key=lambda adds: len(adds & remaining),
            default=frozenset(),
        )
        covered = best_adds & remaining

        if not covered:
            return float("inf")

        remaining.difference_update(covered)
        steps += 1

    return steps
    ### End of your code ###


# ---------------------------------------------------------------------------
# Punto 4b – Ignore-Delete-Lists Heuristic
# ---------------------------------------------------------------------------


def ignoreDeleteListsHeuristic(
    state: State,
    goal: State,
    domain: list[ActionSchema],
    objects: Objects,
) -> float:
    """
    Estimate the plan cost by solving a relaxed problem where no action
    has a delete list (effects never remove fluents from the state).

    In this monotone relaxation, the state only grows over time (fluents are
    never removed), so relaxed reachability can be computed layer by layer.

    Algorithm (relaxed planning graph):
      1. Start from the current state.
      2. At each level, find every grounded action whose preconditions hold.
      3. Add all positive effects and ignore all delete effects.
      4. Count levels until all goal fluents are satisfied.

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    relaxed_state = state
    levels = 0

    while not goal.issubset(relaxed_state):
        next_state = relaxed_state

        for action in _grounded_actions(domain, objects):
            if is_applicable(relaxed_state, action):
                next_state = frozenset(next_state | action.add_list)

        if next_state == relaxed_state:
            return float("inf")

        relaxed_state = next_state
        levels += 1

    return levels
    ### End of your code ###
