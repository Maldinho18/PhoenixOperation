from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import world.rescue_layout as rescue_layout
from planning.heuristics import (
    ignoreDeleteListsHeuristic,
    ignorePreconditionsHeuristic,
)
from planning.htn import build_htn_hierarchy, hierarchicalSearch
from planning.pddl import Action, apply_action, is_applicable
from planning.planner import aStarPlanner, backwardSearch, forwardBFS
from planning.problems import MultiRescueProblem, SimpleRescueProblem


@dataclass
class Result:
    layout: str
    problem: str
    planner: str
    heuristic: str
    length: int
    expanded: int
    seconds: float
    valid: bool


def validate(problem, plan: list[Action]) -> bool:
    state = problem.initial_state
    for action in plan:
        if not is_applicable(state, action):
            return False
        state = apply_action(state, action)
    return problem.isGoalState(state)


def run_classical(
    layout_name: str,
    problem_cls,
    planner_name: str,
    planner: Callable,
    heuristic_name: str = "",
    heuristic: Callable | None = None,
) -> Result:
    layout = rescue_layout.get_layout(layout_name)
    problem = problem_cls(layout)
    start = time.perf_counter()
    if heuristic is None:
        plan = planner(problem)
    else:
        plan = planner(problem, heuristic)
    seconds = time.perf_counter() - start
    return Result(
        layout_name,
        problem_cls.__name__,
        planner_name,
        heuristic_name,
        len(plan),
        problem._expanded,
        seconds,
        validate(problem, plan),
    )


def run_htn(layout_name: str, order_name: str = "layout") -> Result:
    layout = rescue_layout.get_layout(layout_name)
    problem = MultiRescueProblem(layout)
    patient_order = problem.objects["patients"]
    if order_name == "reverse":
        patient_order = list(reversed(patient_order))
    start = time.perf_counter()
    hlas = build_htn_hierarchy(problem, patient_order)
    plan = hierarchicalSearch(problem, hlas)
    seconds = time.perf_counter() - start
    return Result(
        layout_name,
        "MultiRescueProblem",
        f"HTN-{order_name}",
        "",
        len(plan),
        problem._expanded,
        seconds,
        validate(problem, plan),
    )


def print_table(results: list[Result]) -> None:
    print("| layout | problem | planner | heuristic | length | expanded | seconds | valid |")
    print("|---|---|---|---|---:|---:|---:|---|")
    for result in results:
        print(
            f"| {result.layout} | {result.problem} | {result.planner} | "
            f"{result.heuristic or '-'} | {result.length} | {result.expanded} | "
            f"{result.seconds:.4f} | {result.valid} |"
        )


def main() -> None:
    results: list[Result] = []

    for layout_name in ["tinyBase", "smallRescue", "htnBase"]:
        results.append(
            run_classical(layout_name, SimpleRescueProblem, "forwardBFS", forwardBFS)
        )
        results.append(
            run_classical(layout_name, SimpleRescueProblem, "backwardSearch", backwardSearch)
        )
        results.append(
            run_classical(
                layout_name,
                SimpleRescueProblem,
                "aStarPlanner",
                aStarPlanner,
                "ignorePreconditions",
                ignorePreconditionsHeuristic,
            )
        )
        results.append(
            run_classical(
                layout_name,
                SimpleRescueProblem,
                "aStarPlanner",
                aStarPlanner,
                "ignoreDeleteLists",
                ignoreDeleteListsHeuristic,
            )
        )

    for layout_name in ["tinyMulti", "duoRescue", "smallMulti"]:
        results.append(run_htn(layout_name, "layout"))
        results.append(run_htn(layout_name, "reverse"))

    print_table(results)


if __name__ == "__main__":
    main()
