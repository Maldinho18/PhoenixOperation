from __future__ import annotations
from collections import deque
from planning.pddl import Action, Problem, apply_action, is_applicable


# ---------------------------------------------------------------------------
# HTN Infrastructure
# ---------------------------------------------------------------------------


class HLA:
    """
    A High-Level Action (HLA) in HTN planning.

    An HLA is an abstract task that can be refined into sequences of
    more primitive actions (or other HLAs). Each refinement is a list
    of HLA or Action objects.

    name:        Human-readable name for display
    refinements: List of possible refinements, each a list of HLA/Action objects
    """

    def __init__(self, name: str, refinements: list[list] | None = None) -> None:
        self.name = name
        self.refinements = refinements or []

    def __repr__(self) -> str:
        return f"HLA({self.name})"


def is_primitive(action: Action | HLA) -> bool:
    """Return True if action is a primitive (grounded Action), False if it is an HLA."""
    return isinstance(action, Action)


def is_plan_primitive(plan: list[Action | HLA]) -> bool:
    """Return True if every step in the plan is a primitive action."""
    return all(is_primitive(step) for step in plan)


# ---------------------------------------------------------------------------
# Punto 5a – hierarchicalSearch
# ---------------------------------------------------------------------------


def hierarchicalSearch(problem: Problem, hlas: list[HLA]) -> list[Action]:

    # Each queue element is a plan (list of Action | HLA)
    queue: deque[list] = deque()
    queue.append(list(hlas))
 
  
    visited: set[tuple] = set()
 
    while queue:
        plan = queue.popleft()
        problem._expanded += 1
 
        
        sig = tuple(
            step.name if hasattr(step, "name") else str(step) for step in plan
        )
        if sig in visited:
            continue
        visited.add(sig)
 
        # If plan is fully primitive, check if it solves the problem
        if is_plan_primitive(plan):
            state = problem.initial_state
            for action in plan:
                if not is_applicable(state, action):
                    break
                state = apply_action(state, action)
            else:
                # All actions applied successfully
                if problem.isGoalState(state):
                    return plan  
            continue
 
        # Find the first HLA in the plan
        first_hla_idx = next(i for i, step in enumerate(plan) if not is_primitive(step))
        hla = plan[first_hla_idx]
 
        # Expand each refinement of that HLA
        for refinement in hla.refinements:
            new_plan = plan[:first_hla_idx] + list(refinement) + plan[first_hla_idx + 1:]
            queue.append(new_plan)
 
    return []  # No plan found


# ---------------------------------------------------------------------------
# Punto 5b – HLA Definitions
# ---------------------------------------------------------------------------


def build_htn_hierarchy(problem: Problem, patient_order: list[str] | None = None) -> list[HLA]:

    state = problem.initial_state
    objects = problem.objects

    robots = objects["robots"]
    cells = objects["cells"]
    supplies_list = objects["supplies"]
    patients = patient_order or objects["patients"]
    medical_posts = objects["medical_posts"]

    robot = robots[0]
    if not supplies_list or not patients or not medical_posts:
        return []
 
    # Build adjacency map from fluents in the initial state
    adjacency: dict[tuple, list[tuple]] = {c: [] for c in cells}
    for fluent in state:
        if fluent[0] == "Adjacent":
            _, a, b = fluent
            if b not in adjacency[a]:
                adjacency[a].append(b)
    for neighbors in adjacency.values():
        neighbors.sort()
 
    # ------------------------------------------------------------------
    # Helper: Navigate HLA
    # A Navigate(from_cell, to_cell) has one refinement per adjacent cell:
    # a single Move primitive.  For non-adjacent pairs we chain Navigate HLAs.
    # ------------------------------------------------------------------
 
    def make_navigate(from_cell, to_cell) -> HLA:
        """
        Build a Navigate HLA from from_cell to to_cell.
        Refinements: one refinement per shortest path (BFS), each being a
        sequence of primitive Move actions.
        """
        # BFS to find all shortest paths
        if from_cell == to_cell:
            # Null move – empty refinement
            hla = HLA(f"Navigate({from_cell}->{to_cell})", [[]])
            return hla
 
        # BFS for shortest path(s)
        queue: deque[list] = deque([[from_cell]])
        shortest: list[list] = []
        found_len = None
 
        while queue:
            path = queue.popleft()
            if found_len and len(path) > found_len:
                break
            current = path[-1]
            for neighbor in adjacency.get(current, []):
                if neighbor in path:
                    continue
                new_path = path + [neighbor]
                if neighbor == to_cell:
                    if found_len is None:
                        found_len = len(new_path)
                    shortest.append(new_path)
                elif found_len is None or len(new_path) < found_len:
                    queue.append(new_path)
 
        # Build refinements: each path becomes a sequence of Move actions
        refinements = []
        for path in shortest:
            moves = []
            for i in range(len(path) - 1):
                fc = path[i]
                tc = path[i + 1]
                move = Action(
                    f"Move({robot}, {fc}, {tc})",
                    precond_pos=[
                        ("At", robot, fc),
                        ("Adjacent", fc, tc),
                        ("Free", tc),
                    ],
                    precond_neg=[],
                    add_list=[
                        ("At", robot, tc),
                        ("Free", fc),
                    ],
                    del_list=[
                        ("At", robot, fc),
                        ("Free", tc),
                    ],
                )
                moves.append(move)
            refinements.append(moves)
 
        return HLA(f"Navigate({from_cell}->{to_cell})", refinements)
 
    # ------------------------------------------------------------------
    # Helper: get current location of an object from initial state
    # ------------------------------------------------------------------
    def get_location(obj):
        for fluent in state:
            if fluent[0] == "At" and fluent[1] == obj:
                return fluent[2]
        return None
 
    robot_loc = get_location(robot)
 
    # ------------------------------------------------------------------
    # Build top-level HLAs – one FullRescueMission per (supply, patient) pair
    # ------------------------------------------------------------------
    # Pair each patient with a supply (round-robin if counts differ)
    top_level: list[HLA] = []
 
    prepared_supplies = False

    for i, patient in enumerate(patients):
        supply = supplies_list[0]
        medical_post = medical_posts[0]  # Use first medical post
 
        supply_loc = get_location(supply)
        patient_loc = get_location(patient)
 
        # --- Primitive actions ---
 
        pickup_supply = Action(
            f"PickUp({robot}, {supply}, {supply_loc})",
            precond_pos=[
                ("At", robot, supply_loc),
                ("At", supply, supply_loc),
                ("HandsFree", robot),
                ("Pickable", supply),
            ],
            precond_neg=[],
            add_list=[("Holding", robot, supply)],
            del_list=[("At", supply, supply_loc), ("HandsFree", robot)],
        )
 
        setup_supplies = Action(
            f"SetupSupplies({robot}, {supply}, {medical_post})",
            precond_pos=[
                ("At", robot, medical_post),
                ("MedicalPost", medical_post),
                ("Holding", robot, supply),
            ],
            precond_neg=[("SuppliesReady", medical_post)],
            add_list=[("SuppliesReady", medical_post), ("HandsFree", robot)],
            del_list=[("Holding", robot, supply)],
        )
 
        pickup_patient = Action(
            f"PickUp({robot}, {patient}, {patient_loc})",
            precond_pos=[
                ("At", robot, patient_loc),
                ("At", patient, patient_loc),
                ("HandsFree", robot),
                ("Pickable", patient),
            ],
            precond_neg=[],
            add_list=[("Holding", robot, patient)],
            del_list=[("At", patient, patient_loc), ("HandsFree", robot)],
        )
 
        putdown_patient = Action(
            f"PutDown({robot}, {patient}, {medical_post})",
            precond_pos=[
                ("At", robot, medical_post),
                ("Holding", robot, patient),
            ],
            precond_neg=[],
            add_list=[("At", patient, medical_post), ("HandsFree", robot)],
            del_list=[("Holding", robot, patient)],
        )
 
        rescue_patient = Action(
            f"Rescue({robot}, {patient}, {medical_post})",
            precond_pos=[
                ("At", robot, medical_post),
                ("At", patient, medical_post),
                ("MedicalPost", medical_post),
                ("SuppliesReady", medical_post),
            ],
            precond_neg=[],
            add_list=[("Rescued", patient)],
            del_list=[("At", patient, medical_post)],
        )
 
        # --- PrepareSupplies HLA ---
        # Refinement: Navigate(robot_loc -> supply_loc) + PickUp + Navigate(supply_loc -> medical_post) + SetupSupplies
        nav_to_supply = make_navigate(robot_loc, supply_loc)
        nav_supply_to_post = make_navigate(supply_loc, medical_post)
 
        prepare_supplies = HLA(
            f"PrepareSupplies({supply}, {medical_post})",
            refinements=[
                [nav_to_supply, pickup_supply, nav_supply_to_post, setup_supplies]
            ],
        )
 
        # --- ExtractPatient HLA ---
        # Refinement: Navigate(medical_post -> patient_loc) + PickUp + Navigate(patient_loc -> medical_post) + PutDown
        nav_post_to_patient = make_navigate(medical_post, patient_loc)
        nav_patient_to_post = make_navigate(patient_loc, medical_post)
 
        extract_patient = HLA(
            f"ExtractPatient({patient}, {medical_post})",
            refinements=[
                [nav_post_to_patient, pickup_patient, nav_patient_to_post, putdown_patient]
            ],
        )
 
        mission_refinement = [prepare_supplies, extract_patient, rescue_patient]
        if prepared_supplies:
            mission_refinement = [extract_patient, rescue_patient]

        # --- FullRescueMission HLA ---
        full_mission = HLA(
            f"FullRescueMission({supply}, {patient}, {medical_post})",
            refinements=[mission_refinement],
        )
 
        top_level.append(full_mission)
 
        # After this mission, the robot is at medical_post for the next iteration
        robot_loc = medical_post
        prepared_supplies = True
 
    return top_level
