"""
csp_solver.py — Constraint Satisfaction Problem Solver for Pipeline Scheduling.

Formulates the task scheduling problem as a CSP:
- Variables: Each task gets assigned a (resource_id, start_time) pair.
- Domains: All valid (resource, time_slot) combinations for each task.
- Constraints: Dependencies, resource capacity, SLA deadlines, resource fitness.

Solving strategies:
- Backtracking search with:
  - MRV (Minimum Remaining Values) for variable ordering
  - LCV (Least Constraining Value) for value ordering
  - Forward checking to prune inconsistent domains early
- AC-3 (Arc Consistency) as optional preprocessing
"""

from dataclasses import dataclass, field
from typing import Optional
from src.task_dag import Pipeline, Task, Resource, Schedule, ScheduleEntry


# ================================================================
# CSP Data Structures
# ================================================================

@dataclass
class CSPVariable:
    """
    A CSP variable representing a task to be scheduled.

    Attributes:
        task_id: The task this variable represents.
        domain: List of possible (resource_id, start_time) assignments.
    """
    task_id: str
    domain: list[tuple[str, float]] = field(default_factory=list)

    def domain_size(self) -> int:
        return len(self.domain)


@dataclass
class CSPAssignment:
    """A single variable assignment: task → (resource, start_time)."""
    task_id: str
    resource_id: str
    start_time: float
    end_time: float


@dataclass
class CSPResult:
    """
    Result of the CSP solver.

    Attributes:
        success: Whether a valid assignment was found.
        assignments: List of CSPAssignment objects (empty if failed).
        schedule: A Schedule object built from the assignments.
        nodes_explored: Number of backtracking nodes visited.
        backtracks: Number of times the solver had to backtrack.
        message: Human-readable status message.
    """
    success: bool
    assignments: list[CSPAssignment] = field(default_factory=list)
    schedule: Optional[Schedule] = None
    nodes_explored: int = 0
    backtracks: int = 0
    message: str = ""


class CSPSolver:
    """
    Solves the pipeline scheduling problem as a CSP.

    Usage:
        solver = CSPSolver(pipeline, time_horizon=480, time_step=15)
        result = solver.solve()
        if result.success:
            print(result.schedule)
    """

    def __init__(self, pipeline: Pipeline,
                 time_horizon: float = 480.0,
                 time_step: float = 15.0,
                 start_time: float = 0.0):
        """
        Args:
            pipeline: The Pipeline to schedule.
            time_horizon: Total scheduling window in minutes (default: 480 = 8 hours).
            time_step: Time slot granularity in minutes (default: 15).
            start_time: Earliest possible start time in minutes from midnight.
        """
        self.pipeline = pipeline
        self.time_horizon = time_horizon
        self.time_step = time_step
        self.start_time = start_time
        self.time_slots = self._generate_time_slots()

        # Solver statistics
        self.nodes_explored = 0
        self.backtracks = 0

    def _generate_time_slots(self) -> list[float]:
        """Generate all possible time slots within the horizon."""
        slots = []
        t = self.start_time
        while t <= self.start_time + self.time_horizon:
            slots.append(t)
            t += self.time_step
        return slots

    # ================================================================
    # Domain Generation
    # ================================================================

    def _build_variables(self) -> dict[str, CSPVariable]:
        """
        Create CSP variables with initial domains for each task.

        A task's domain = all (resource, time_slot) pairs where:
        1. The resource has enough CPU and memory for the task.
        2. The time slot allows the task to finish within the horizon.
        """
        variables = {}

        for task_id, task in self.pipeline.tasks.items():
            domain = []
            for resource in self.pipeline.resources:
                # Check resource fitness
                if (task.cpu_required <= resource.cpu_capacity and
                        task.memory_required <= resource.memory_capacity):
                    for t in self.time_slots:
                        end_time = t + task.duration_estimate
                        # Must finish within horizon
                        if end_time <= self.start_time + self.time_horizon:
                            # If task has SLA, must finish before deadline
                            if task.sla_deadline is None or end_time <= task.sla_deadline:
                                domain.append((resource.resource_id, t))

            variables[task_id] = CSPVariable(task_id=task_id, domain=domain)

        return variables

    # ================================================================
    # Constraint Checking
    # ================================================================

    def _check_dependency_constraint(self, task_id: str, start_time: float,
                                      assignments: dict[str, CSPAssignment]
                                      ) -> bool:
        """
        Dependency constraint: task can't start until ALL upstream tasks finish.

        Returns True if the constraint is satisfied.
        """
        for upstream_id in self.pipeline.get_dependencies(task_id):
            if upstream_id in assignments:
                upstream_end = assignments[upstream_id].end_time
                if start_time < upstream_end:
                    return False
            # If upstream not yet assigned, we can't check — will be caught later
        return True

    def _check_resource_capacity_constraint(self, task_id: str,
                                             resource_id: str,
                                             start_time: float,
                                             end_time: float,
                                             assignments: dict[str, CSPAssignment]
                                             ) -> bool:
        """
        Resource capacity constraint: at any point in time, the total CPU usage
        on a resource must not exceed its capacity.

        Returns True if the constraint is satisfied.
        """
        task = self.pipeline.tasks[task_id]
        resource = None
        for r in self.pipeline.resources:
            if r.resource_id == resource_id:
                resource = r
                break
        if resource is None:
            return False

        # Check every time point where this task would be running
        for other_id, other_assign in assignments.items():
            if other_assign.resource_id != resource_id:
                continue
            # Check if they overlap in time
            if start_time < other_assign.end_time and other_assign.start_time < end_time:
                # They overlap — check combined CPU
                other_task = self.pipeline.tasks[other_id]
                combined_cpu = task.cpu_required + other_task.cpu_required
                if combined_cpu > resource.cpu_capacity:
                    return False

        return True

    def _check_memory_capacity_constraint(self, task_id: str,
                                           resource_id: str,
                                           start_time: float,
                                           end_time: float,
                                           assignments: dict[str, CSPAssignment]
                                           ) -> bool:
        """
        Memory capacity constraint: similar to CPU but for memory.
        """
        task = self.pipeline.tasks[task_id]
        resource = None
        for r in self.pipeline.resources:
            if r.resource_id == resource_id:
                resource = r
                break
        if resource is None:
            return False

        for other_id, other_assign in assignments.items():
            if other_assign.resource_id != resource_id:
                continue
            if start_time < other_assign.end_time and other_assign.start_time < end_time:
                other_task = self.pipeline.tasks[other_id]
                combined_mem = task.memory_required + other_task.memory_required
                if combined_mem > resource.memory_capacity:
                    return False

        return True

    def _is_consistent(self, task_id: str, resource_id: str,
                        start_time: float,
                        assignments: dict[str, CSPAssignment]) -> bool:
        """
        Check if assigning (resource_id, start_time) to task_id is consistent
        with ALL constraints given current assignments.
        """
        task = self.pipeline.tasks[task_id]
        end_time = start_time + task.duration_estimate

        # Constraint 1: Dependencies
        if not self._check_dependency_constraint(task_id, start_time, assignments):
            return False

        # Constraint 2: Resource CPU capacity
        if not self._check_resource_capacity_constraint(
                task_id, resource_id, start_time, end_time, assignments):
            return False

        # Constraint 3: Resource memory capacity
        if not self._check_memory_capacity_constraint(
                task_id, resource_id, start_time, end_time, assignments):
            return False

        # Constraint 4: SLA deadline (already filtered in domain, but double check)
        if task.sla_deadline is not None and end_time > task.sla_deadline:
            return False

        return True

    # ================================================================
    # Variable Ordering: MRV (Minimum Remaining Values)
    # ================================================================

    def _select_unassigned_variable(self, variables: dict[str, CSPVariable],
                                     assignments: dict[str, CSPAssignment]
                                     ) -> Optional[str]:
        """
        MRV heuristic: choose the unassigned variable with the smallest domain.
        Ties broken by highest priority (schedule important tasks first).
        """
        unassigned = [
            tid for tid in variables
            if tid not in assignments
        ]
        if not unassigned:
            return None

        return min(
            unassigned,
            key=lambda tid: (
                variables[tid].domain_size(),
                -self.pipeline.tasks[tid].priority  # higher priority = break ties
            )
        )

    # ================================================================
    # Value Ordering: LCV (Least Constraining Value)
    # ================================================================

    def _order_domain_values(self, task_id: str,
                              variables: dict[str, CSPVariable],
                              assignments: dict[str, CSPAssignment]
                              ) -> list[tuple[str, float]]:
        """
        LCV heuristic: try values that rule out the fewest choices
        for unassigned neighbors first.

        For efficiency, we approximate: prefer earlier start times
        (leaves more room for dependent tasks) and resources with
        more remaining capacity.
        """
        domain = variables[task_id].domain

        def lcv_score(value: tuple[str, float]) -> float:
            resource_id, start_time = value
            task = self.pipeline.tasks[task_id]
            end_time = start_time + task.duration_estimate

            # Score: how many domain values does this eliminate from neighbors?
            eliminated = 0
            dependents = self.pipeline.get_dependents(task_id)

            for dep_id in dependents:
                if dep_id in assignments:
                    continue
                for res_id, t in variables[dep_id].domain:
                    # Would this force the dependent to start after our end?
                    if t < end_time:
                        eliminated += 1

            return eliminated

        return sorted(domain, key=lcv_score)

    # ================================================================
    # Forward Checking
    # ================================================================

    def _forward_check(self, task_id: str, resource_id: str,
                        start_time: float,
                        variables: dict[str, CSPVariable],
                        assignments: dict[str, CSPAssignment]
                        ) -> Optional[dict[str, list[tuple[str, float]]]]:
        """
        After assigning (resource_id, start_time) to task_id,
        prune inconsistent values from unassigned neighbors' domains.

        Returns:
            Dict of {task_id: pruned_values} for undo on backtrack.
            None if any domain becomes empty (= dead end).
        """
        task = self.pipeline.tasks[task_id]
        end_time = start_time + task.duration_estimate
        pruned = {}

        # Check downstream dependents — they can't start before we finish
        dependents = self.pipeline.get_dependents(task_id)
        for dep_id in dependents:
            if dep_id in assignments:
                continue
            var = variables[dep_id]
            removed = []
            for val in var.domain:
                res_id, t = val
                if t < end_time:
                    removed.append(val)
            if removed:
                pruned[dep_id] = removed
                var.domain = [v for v in var.domain if v not in removed]
                if var.domain_size() == 0:
                    # Dead end — restore and return None
                    self._restore_domains(pruned, variables)
                    return None

        # Check same-resource tasks — prune overlapping time slots
        # that would violate capacity constraints
        for other_id in variables:
            if other_id in assignments or other_id == task_id:
                continue
            var = variables[other_id]
            removed = []
            other_task = self.pipeline.tasks[other_id]
            for val in var.domain:
                res_id, t = val
                if res_id != resource_id:
                    continue
                other_end = t + other_task.duration_estimate
                # Check if they would overlap
                if t < end_time and start_time < other_end:
                    # Check capacity
                    resource = next(
                        r for r in self.pipeline.resources
                        if r.resource_id == resource_id
                    )
                    if (task.cpu_required + other_task.cpu_required >
                            resource.cpu_capacity):
                        removed.append(val)
                    if (task.memory_required + other_task.memory_required >
                            resource.memory_capacity):
                        if val not in removed:
                            removed.append(val)

            if removed:
                if other_id not in pruned:
                    pruned[other_id] = []
                pruned[other_id].extend(removed)
                var.domain = [v for v in var.domain if v not in removed]
                if var.domain_size() == 0:
                    self._restore_domains(pruned, variables)
                    return None

        return pruned

    def _restore_domains(self, pruned: dict[str, list[tuple[str, float]]],
                          variables: dict[str, CSPVariable]):
        """Restore pruned values to variable domains (undo forward checking)."""
        for task_id, values in pruned.items():
            variables[task_id].domain.extend(values)

    # ================================================================
    # AC-3 (Arc Consistency)
    # ================================================================

    def _ac3(self, variables: dict[str, CSPVariable]) -> bool:
        """
        AC-3 algorithm: enforce arc consistency as preprocessing.

        For each pair of constrained variables, remove values from domains
        that have no consistent counterpart.

        Returns:
            True if all domains still have values, False if any domain is emptied.
        """
        # Build queue of arcs (directed constraint pairs)
        queue = []
        for task_id in variables:
            # Dependency arcs
            for dep_id in self.pipeline.get_dependencies(task_id):
                if dep_id in variables:
                    queue.append((task_id, dep_id))  # task depends on dep
            for dep_id in self.pipeline.get_dependents(task_id):
                if dep_id in variables:
                    queue.append((task_id, dep_id))

        while queue:
            xi, xj = queue.pop(0)
            if self._revise(variables, xi, xj):
                if variables[xi].domain_size() == 0:
                    return False
                # Add all neighbors of xi (except xj) back to queue
                neighbors = (
                    set(self.pipeline.get_dependencies(xi)) |
                    set(self.pipeline.get_dependents(xi))
                )
                for xk in neighbors:
                    if xk != xj and xk in variables:
                        queue.append((xk, xi))

        return True

    def _revise(self, variables: dict[str, CSPVariable],
                xi: str, xj: str) -> bool:
        """
        Revise the domain of xi given constraint with xj.
        Remove any value from xi's domain that has no consistent
        value in xj's domain.

        Returns True if any value was removed.
        """
        revised = False
        to_remove = []

        for val_i in variables[xi].domain:
            res_i, t_i = val_i
            task_i = self.pipeline.tasks[xi]
            end_i = t_i + task_i.duration_estimate

            # Check if there's at least one consistent value in xj's domain
            has_consistent = False
            for val_j in variables[xj].domain:
                res_j, t_j = val_j
                task_j = self.pipeline.tasks[xj]
                end_j = t_j + task_j.duration_estimate

                consistent = True

                # Dependency constraint: if xi depends on xj, xi must start after xj ends
                if xj in self.pipeline.get_dependencies(xi):
                    if t_i < end_j:
                        consistent = False

                # If xj depends on xi, xj must start after xi ends
                if xi in self.pipeline.get_dependencies(xj):
                    if t_j < end_i:
                        consistent = False

                # Resource capacity: if same resource and overlapping
                if consistent and res_i == res_j:
                    if t_i < end_j and t_j < end_i:
                        resource = next(
                            (r for r in self.pipeline.resources
                             if r.resource_id == res_i), None
                        )
                        if resource:
                            if (task_i.cpu_required + task_j.cpu_required >
                                    resource.cpu_capacity):
                                consistent = False

                if consistent:
                    has_consistent = True
                    break

            if not has_consistent:
                to_remove.append(val_i)
                revised = True

        for val in to_remove:
            variables[xi].domain.remove(val)

        return revised

    # ================================================================
    # Backtracking Search
    # ================================================================

    def _backtrack(self, variables: dict[str, CSPVariable],
                    assignments: dict[str, CSPAssignment]) -> bool:
        """
        Recursive backtracking search with MRV, LCV, and forward checking.

        Returns True if a complete consistent assignment is found.
        """
        self.nodes_explored += 1

        # Check if all variables are assigned
        if len(assignments) == len(variables):
            return True

        # Select next variable (MRV)
        task_id = self._select_unassigned_variable(variables, assignments)
        if task_id is None:
            return True

        task = self.pipeline.tasks[task_id]

        # Order domain values (LCV)
        ordered_values = self._order_domain_values(task_id, variables, assignments)

        for resource_id, start_time in ordered_values:
            # Check consistency
            if self._is_consistent(task_id, resource_id, start_time, assignments):
                # Make assignment
                end_time = start_time + task.duration_estimate
                assignments[task_id] = CSPAssignment(
                    task_id=task_id,
                    resource_id=resource_id,
                    start_time=start_time,
                    end_time=end_time
                )

                # Forward check
                pruned = self._forward_check(
                    task_id, resource_id, start_time, variables, assignments
                )

                if pruned is not None:
                    # Recurse
                    if self._backtrack(variables, assignments):
                        return True
                    # Undo forward checking
                    self._restore_domains(pruned, variables)

                # Undo assignment
                del assignments[task_id]
                self.backtracks += 1

        return False

    # ================================================================
    # Public Interface
    # ================================================================

    def solve(self, use_ac3: bool = True) -> CSPResult:
        """
        Solve the scheduling CSP.

        Args:
            use_ac3: Whether to run AC-3 preprocessing (default: True).

        Returns:
            CSPResult with success status, assignments, and schedule.
        """
        self.nodes_explored = 0
        self.backtracks = 0

        # Build variables with domains
        variables = self._build_variables()

        # Check for empty domains
        for task_id, var in variables.items():
            if var.domain_size() == 0:
                return CSPResult(
                    success=False,
                    message=f"Task '{task_id}' has no valid (resource, time) assignments. "
                            f"Check resource capacity and SLA deadlines."
                )

        # AC-3 preprocessing
        if use_ac3:
            if not self._ac3(variables):
                return CSPResult(
                    success=False,
                    nodes_explored=0,
                    message="AC-3 detected infeasibility: no consistent assignment exists."
                )

        # Backtracking search
        assignments: dict[str, CSPAssignment] = {}
        success = self._backtrack(variables, assignments)

        if success:
            # Build schedule from assignments
            schedule = Schedule()
            for assign in assignments.values():
                schedule.add_entry(ScheduleEntry(
                    task_id=assign.task_id,
                    resource_id=assign.resource_id,
                    start_time=assign.start_time,
                    end_time=assign.end_time
                ))
            return CSPResult(
                success=True,
                assignments=list(assignments.values()),
                schedule=schedule,
                nodes_explored=self.nodes_explored,
                backtracks=self.backtracks,
                message=f"Solution found. Explored {self.nodes_explored} nodes, "
                        f"{self.backtracks} backtracks."
            )
        else:
            return CSPResult(
                success=False,
                nodes_explored=self.nodes_explored,
                backtracks=self.backtracks,
                message=f"No solution found after exploring {self.nodes_explored} nodes "
                        f"with {self.backtracks} backtracks."
            )

    def solve_without_fc(self, use_ac3: bool = False) -> CSPResult:
        """
        Solve WITHOUT forward checking — for comparison in experiments.
        Uses plain backtracking with MRV + LCV only.
        """
        self.nodes_explored = 0
        self.backtracks = 0

        variables = self._build_variables()

        for task_id, var in variables.items():
            if var.domain_size() == 0:
                return CSPResult(
                    success=False,
                    message=f"Task '{task_id}' has no valid assignments."
                )

        if use_ac3:
            if not self._ac3(variables):
                return CSPResult(success=False, message="AC-3 detected infeasibility.")

        assignments: dict[str, CSPAssignment] = {}
        success = self._backtrack_plain(variables, assignments)

        if success:
            schedule = Schedule()
            for assign in assignments.values():
                schedule.add_entry(ScheduleEntry(
                    task_id=assign.task_id,
                    resource_id=assign.resource_id,
                    start_time=assign.start_time,
                    end_time=assign.end_time
                ))
            return CSPResult(
                success=True,
                assignments=list(assignments.values()),
                schedule=schedule,
                nodes_explored=self.nodes_explored,
                backtracks=self.backtracks,
                message=f"Solution found (no FC). Explored {self.nodes_explored} nodes."
            )
        else:
            return CSPResult(
                success=False,
                nodes_explored=self.nodes_explored,
                backtracks=self.backtracks,
                message=f"No solution found (no FC) after {self.nodes_explored} nodes."
            )

    def _backtrack_plain(self, variables: dict[str, CSPVariable],
                          assignments: dict[str, CSPAssignment]) -> bool:
        """Plain backtracking without forward checking (for experiments)."""
        self.nodes_explored += 1

        if len(assignments) == len(variables):
            return True

        task_id = self._select_unassigned_variable(variables, assignments)
        if task_id is None:
            return True

        task = self.pipeline.tasks[task_id]

        for resource_id, start_time in variables[task_id].domain:
            if self._is_consistent(task_id, resource_id, start_time, assignments):
                end_time = start_time + task.duration_estimate
                assignments[task_id] = CSPAssignment(
                    task_id=task_id,
                    resource_id=resource_id,
                    start_time=start_time,
                    end_time=end_time
                )
                if self._backtrack_plain(variables, assignments):
                    return True
                del assignments[task_id]
                self.backtracks += 1

        return False

    # ================================================================
    # Schedule Validation
    # ================================================================

    @staticmethod
    def validate_schedule(schedule: Schedule, pipeline: Pipeline) -> list[str]:
        """
        Validate a schedule against all constraints.

        Returns list of violation messages. Empty list = valid schedule.
        """
        violations = []

        # Check dependencies
        for task_id in pipeline.tasks:
            entry = schedule.get_entry_for_task(task_id)
            if entry is None:
                violations.append(f"Task '{task_id}' is not scheduled.")
                continue
            for upstream_id in pipeline.get_dependencies(task_id):
                upstream_entry = schedule.get_entry_for_task(upstream_id)
                if upstream_entry is None:
                    violations.append(
                        f"Dependency violation: '{task_id}' depends on "
                        f"'{upstream_id}' which is not scheduled."
                    )
                elif entry.start_time < upstream_entry.end_time:
                    violations.append(
                        f"Dependency violation: '{task_id}' starts at "
                        f"{entry.start_time:.0f} but '{upstream_id}' "
                        f"ends at {upstream_entry.end_time:.0f}."
                    )

        # Check SLA deadlines
        for task_id, task in pipeline.tasks.items():
            if task.sla_deadline is not None:
                entry = schedule.get_entry_for_task(task_id)
                if entry and entry.end_time > task.sla_deadline:
                    violations.append(
                        f"SLA violation: '{task_id}' ends at "
                        f"{entry.end_time:.0f} but deadline is "
                        f"{task.sla_deadline:.0f}."
                    )

        # Check resource capacity (CPU)
        for resource in pipeline.resources:
            entries = schedule.get_entries_for_resource(resource.resource_id)
            for i, e1 in enumerate(entries):
                for e2 in entries[i + 1:]:
                    if e1.overlaps_with(e2):
                        t1 = pipeline.tasks[e1.task_id]
                        t2 = pipeline.tasks[e2.task_id]
                        if t1.cpu_required + t2.cpu_required > resource.cpu_capacity:
                            violations.append(
                                f"CPU capacity violation on '{resource.resource_id}': "
                                f"'{e1.task_id}' ({t1.cpu_required} CPU) and "
                                f"'{e2.task_id}' ({t2.cpu_required} CPU) overlap "
                                f"but resource only has {resource.cpu_capacity} CPUs."
                            )

        return violations
