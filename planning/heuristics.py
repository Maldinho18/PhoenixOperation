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
    Estimate the number of remaining goals when action preconditions are ignored.

    In this relaxation, each unsatisfied goal fluent can be treated as needing at
    least one constructive step. The workshop rubric asks for this direct count.

    Tip: frozenset supports set difference (-) and intersection (&).
         You only need to ground actions once per call (use get_applicable_actions
         with the initial state, or generate all groundings regardless of state).
         Remember: with no preconditions, every grounding is "applicable".
    """
    ### Your code here ###
    return len(goal - state)
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
    Estimate the plan cost in the relaxed problem with empty delete lists.

    The implementation performs a greedy hill-climbing search in the monotone
    relaxed state space. At each step it applies the applicable action that
    introduces the most new fluents, using direct goal progress as a tie-breaker.

    Tip: In the relaxed problem, apply_action never removes fluents.
         You can implement this by treating del_list as empty for all actions.
         Use get_applicable_actions to enumerate applicable grounded actions at
         each step (preconditions still apply in the relaxed model).
    """
    ### Your code here ###
    relaxed_state = state
    steps = 0
    actions = _grounded_actions(domain, objects)
    max_steps = max(1, len(actions))

    while not goal.issubset(relaxed_state):
        applicable = [action for action in actions if is_applicable(relaxed_state, action)]
        improving = [
            action
            for action in applicable
            if action.add_list - relaxed_state
        ]

        if not improving:
            return float("inf")

        best_action = max(
            improving,
            key=lambda action: (
                len((action.add_list - relaxed_state) & goal),
                len(action.add_list - relaxed_state),
                -len(action.precond_pos - relaxed_state),
            ),
        )
        relaxed_state = frozenset(relaxed_state | best_action.add_list)
        steps += 1

        if steps > max_steps:
            return float("inf")

    return steps
    ### End of your code ###
