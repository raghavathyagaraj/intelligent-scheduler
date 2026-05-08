"""
search_planner.py — Search & Planning for Failure Recovery.

When tasks fail, take longer than expected, or new high-priority tasks arrive,
the planner finds the best recovery plan using search:

- A* Search: Optimal recovery plan using heuristic = estimated remaining time + SLA penalties.
- Greedy Best-First: Faster but suboptimal — uses heuristic only.

State space:
- State = snapshot of all task statuses + resource usage + current time.
- Actions = retry_task, skip_task, reschedule_task, reassign_resource.
- Goal = all required tasks completed (or best-effort with SLA maximization).
"""

import heapq
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional
from src.task_dag import Pipeline, Task, TaskStatus, Resource, Schedule, ScheduleEntry


# ================================================================
# State Representation
# ================================================================

@dataclass
class PlannerState:
    """
    A snapshot of the pipeline execution state.

    Attributes:
        task_statuses: Mapping of task_id → status string.
        current_time: Current simulation time (minutes from midnight).
        schedule: Current partial schedule.
        resource_usage: {resource_id: current_cpu_in_use}.
        actions_taken: List of actions taken to reach this state.
        cost: Total cost (g-value in A*) — sum of action costs.
    """
    task_statuses: dict[str, str]
    current_time: float
    schedule: list[tuple[str, str, float, float]]  # (task, resource, start, end)
    resource_usage: dict[str, float]
    actions_taken: list[str] = field(default_factory=list)
    cost: float = 0.0

    def to_hashable(self) -> tuple:
        """Create a hashable representation for visited-state tracking."""
        status_tuple = tuple(sorted(self.task_statuses.items()))
        return (status_tuple, round(self.current_time, 1))

    def __lt__(self, other):
        """For heapq comparison."""
        return self.cost < other.cost


@dataclass
class PlannerAction:
    """An action the planner can take."""
    action_type: str       # "retry", "skip", "reschedule", "wait"
    task_id: str
    resource_id: Optional[str] = None
    start_time: Optional[float] = None
    cost: float = 0.0
    description: str = ""


@dataclass
class SearchResult:
    """
    Result of the search planner.

    Attributes:
        success: Whether a plan was found that reaches the goal.
        actions: Ordered list of actions in the recovery plan.
        final_state: The state at the end of the plan.
        schedule: The resulting schedule after executing the plan.
        nodes_explored: Number of states explored during search.
        sla_adherence: Percentage of SLA-bound tasks meeting deadlines.
        total_cost: Total cost of the plan.
    """
    success: bool
    actions: list[PlannerAction] = field(default_factory=list)
    final_state: Optional[PlannerState] = None
    schedule: Optional[Schedule] = None
    nodes_explored: int = 0
    sla_adherence: float = 0.0
    total_cost: float = 0.0
    message: str = ""


class SearchPlanner:
    """
    A* and Greedy Best-First search for pipeline re-planning.

    Given a pipeline with some tasks completed, some failed, and a current time,
    finds the best sequence of actions to complete remaining work while
    maximizing SLA adherence and minimizing total makespan.
    """

    # Action cost weights
    RETRY_COST = 5.0
    SKIP_COST = 50.0      # skipping is expensive — means lost work
    SCHEDULE_COST = 1.0   # scheduling a ready task is cheap
    WAIT_COST = 0.5       # waiting per time step
    SLA_VIOLATION_PENALTY = 100.0

    def __init__(self, pipeline: Pipeline, current_time: float = 0.0,
                 max_nodes: int = 5000):
        """
        Args:
            pipeline: The pipeline being scheduled.
            current_time: Current time in minutes from midnight.
            max_nodes: Maximum nodes to explore before giving up.
        """
        self.pipeline = pipeline
        self.current_time = current_time
        self.max_nodes = max_nodes

    # ================================================================
    # State Initialization
    # ================================================================

    def create_initial_state(self,
                              task_statuses: Optional[dict[str, str]] = None,
                              current_schedule: Optional[list[tuple]] = None
                              ) -> PlannerState:
        """
        Create the initial state from current pipeline status.

        Args:
            task_statuses: Override statuses. If None, reads from pipeline.
            current_schedule: Existing schedule entries as
                              (task_id, resource_id, start, end) tuples.
        """
        if task_statuses is None:
            task_statuses = {
                tid: t.status.value
                for tid, t in self.pipeline.tasks.items()
            }

        resource_usage = {r.resource_id: 0.0 for r in self.pipeline.resources}

        return PlannerState(
            task_statuses=dict(task_statuses),
            current_time=self.current_time,
            schedule=list(current_schedule or []),
            resource_usage=resource_usage,
            actions_taken=[],
            cost=0.0
        )

    # ================================================================
    # Goal Test
    # ================================================================

    def _is_goal(self, state: PlannerState) -> bool:
        """
        Goal: all tasks are either completed or skipped.
        (Failed tasks that can't be retried are treated as skipped.)
        """
        for task_id, status in state.task_statuses.items():
            if status not in ("completed", "skipped"):
                return False
        return True

    # ================================================================
    # Action Generation
    # ================================================================

    def _get_actions(self, state: PlannerState) -> list[PlannerAction]:
        """Generate all valid actions from the current state."""
        actions = []

        for task_id, status in state.task_statuses.items():
            task = self.pipeline.tasks[task_id]

            if status == "failed":
                # Action: retry the failed task
                for resource in self.pipeline.resources:
                    if (task.cpu_required <= resource.cpu_capacity and
                            task.memory_required <= resource.memory_capacity):
                        actions.append(PlannerAction(
                            action_type="retry",
                            task_id=task_id,
                            resource_id=resource.resource_id,
                            start_time=state.current_time,
                            cost=self.RETRY_COST,
                            description=f"Retry '{task_id}' on {resource.resource_id} "
                                        f"at t={state.current_time:.0f}"
                        ))

                # Action: skip the failed task
                actions.append(PlannerAction(
                    action_type="skip",
                    task_id=task_id,
                    cost=self.SKIP_COST,
                    description=f"Skip '{task_id}' (accept failure)"
                ))

            elif status == "pending":
                # Check if dependencies are met
                deps = self.pipeline.get_dependencies(task_id)
                deps_met = all(
                    state.task_statuses.get(d) == "completed"
                    for d in deps
                )
                deps_blocked = any(
                    state.task_statuses.get(d) in ("failed", "skipped")
                    for d in deps
                )

                if deps_met:
                    # Action: schedule this ready task
                    for resource in self.pipeline.resources:
                        if (task.cpu_required <= resource.cpu_capacity and
                                task.memory_required <= resource.memory_capacity):
                            actions.append(PlannerAction(
                                action_type="schedule",
                                task_id=task_id,
                                resource_id=resource.resource_id,
                                start_time=state.current_time,
                                cost=self.SCHEDULE_COST,
                                description=f"Schedule '{task_id}' on "
                                            f"{resource.resource_id} at "
                                            f"t={state.current_time:.0f}"
                            ))
                elif deps_blocked:
                    # Upstream failed/skipped — this task must be skipped too
                    actions.append(PlannerAction(
                        action_type="skip",
                        task_id=task_id,
                        cost=self.SKIP_COST * 0.5,  # cheaper — not our fault
                        description=f"Skip '{task_id}' (upstream blocked)"
                    ))

        # If no scheduling actions available but tasks remain, add a wait action
        scheduling_actions = [a for a in actions if a.action_type == "schedule"]
        if not scheduling_actions and not self._is_goal(state):
            has_pending = any(s == "pending" for s in state.task_statuses.values())
            if has_pending:
                actions.append(PlannerAction(
                    action_type="wait",
                    task_id="__system__",
                    cost=self.WAIT_COST,
                    description=f"Wait at t={state.current_time:.0f}"
                ))

        return actions

    # ================================================================
    # State Transition
    # ================================================================

    def _apply_action(self, state: PlannerState,
                       action: PlannerAction) -> PlannerState:
        """Apply an action to a state and return the new state."""
        new_state = PlannerState(
            task_statuses=dict(state.task_statuses),
            current_time=state.current_time,
            schedule=list(state.schedule),
            resource_usage=dict(state.resource_usage),
            actions_taken=list(state.actions_taken),
            cost=state.cost + action.cost
        )
        new_state.actions_taken.append(action.description)

        if action.action_type == "schedule":
            task = self.pipeline.tasks[action.task_id]
            end_time = action.start_time + task.duration_estimate
            new_state.task_statuses[action.task_id] = "completed"
            new_state.schedule.append((
                action.task_id, action.resource_id,
                action.start_time, end_time
            ))
            new_state.current_time = max(new_state.current_time, end_time)

        elif action.action_type == "retry":
            task = self.pipeline.tasks[action.task_id]
            end_time = action.start_time + task.duration_estimate
            new_state.task_statuses[action.task_id] = "completed"
            new_state.schedule.append((
                action.task_id, action.resource_id,
                action.start_time, end_time
            ))
            new_state.current_time = max(new_state.current_time, end_time)

        elif action.action_type == "skip":
            new_state.task_statuses[action.task_id] = "skipped"
            # Also skip all downstream tasks that depend on this one
            self._cascade_skip(new_state, action.task_id)

        elif action.action_type == "wait":
            new_state.current_time += 15  # advance by one time step

        return new_state

    def _cascade_skip(self, state: PlannerState, skipped_task_id: str):
        """Recursively skip all downstream tasks that depend on a skipped task."""
        for dependent_id in self.pipeline.get_dependents(skipped_task_id):
            if state.task_statuses.get(dependent_id) == "pending":
                # Check if ALL upstream paths are blocked
                deps = self.pipeline.get_dependencies(dependent_id)
                all_blocked = all(
                    state.task_statuses.get(d) in ("skipped", "failed")
                    for d in deps
                )
                if all_blocked:
                    state.task_statuses[dependent_id] = "skipped"
                    self._cascade_skip(state, dependent_id)

    # ================================================================
    # Heuristic
    # ================================================================

    def _heuristic(self, state: PlannerState) -> float:
        """
        Admissible heuristic for A*:
        h(state) = estimated time to complete remaining tasks
                  + SLA violation penalties for at-risk tasks.

        This is admissible because:
        - We use the sum of remaining task durations (optimistic — assumes
          perfect parallelization).
        - SLA penalties are only added when the deadline is already exceeded.
        """
        h = 0.0

        # Sum of remaining task durations (lower bound on remaining time)
        remaining_durations = []
        for task_id, status in state.task_statuses.items():
            if status in ("pending", "failed"):
                task = self.pipeline.tasks[task_id]
                remaining_durations.append(task.duration_estimate)

        if remaining_durations:
            # Optimistic: assume perfect parallelization across all resources
            num_resources = max(len(self.pipeline.resources), 1)
            total_work = sum(remaining_durations)
            h += total_work / num_resources

        # SLA violation penalties
        for task_id, status in state.task_statuses.items():
            task = self.pipeline.tasks[task_id]
            if task.sla_deadline is None:
                continue

            if status in ("pending", "failed"):
                # Estimate earliest possible completion
                earliest = state.current_time + task.duration_estimate
                if earliest > task.sla_deadline:
                    h += self.SLA_VIOLATION_PENALTY
            elif status == "skipped":
                # Skipped SLA task — that's a violation
                h += self.SLA_VIOLATION_PENALTY

        return h

    # ================================================================
    # A* Search
    # ================================================================

    def search_astar(self, initial_state: Optional[PlannerState] = None
                      ) -> SearchResult:
        """
        A* search for optimal recovery plan.

        Returns the plan with minimum total cost (g + h).
        """
        if initial_state is None:
            initial_state = self.create_initial_state()

        nodes_explored = 0
        visited = set()

        # Priority queue: (f_score, tie_breaker, state)
        counter = 0
        h0 = self._heuristic(initial_state)
        frontier = [(initial_state.cost + h0, counter, initial_state)]

        while frontier and nodes_explored < self.max_nodes:
            f_score, _, current = heapq.heappop(frontier)
            nodes_explored += 1

            state_hash = current.to_hashable()
            if state_hash in visited:
                continue
            visited.add(state_hash)

            # Goal check
            if self._is_goal(current):
                schedule = self._build_schedule(current)
                sla_rate = schedule.get_sla_adherence_rate(self.pipeline.tasks)
                return SearchResult(
                    success=True,
                    actions=self._extract_actions(current),
                    final_state=current,
                    schedule=schedule,
                    nodes_explored=nodes_explored,
                    sla_adherence=sla_rate,
                    total_cost=current.cost,
                    message=f"A* found plan in {nodes_explored} nodes, "
                            f"cost={current.cost:.1f}, SLA={sla_rate:.0f}%"
                )

            # Expand
            for action in self._get_actions(current):
                new_state = self._apply_action(current, action)
                new_hash = new_state.to_hashable()
                if new_hash not in visited:
                    h = self._heuristic(new_state)
                    f = new_state.cost + h
                    counter += 1
                    heapq.heappush(frontier, (f, counter, new_state))

        return SearchResult(
            success=False,
            nodes_explored=nodes_explored,
            message=f"A* exhausted search after {nodes_explored} nodes."
        )

    # ================================================================
    # Greedy Best-First Search
    # ================================================================

    def search_greedy(self, initial_state: Optional[PlannerState] = None
                       ) -> SearchResult:
        """
        Greedy best-first search: expands the node with lowest heuristic only.
        Faster than A* but not guaranteed optimal.
        """
        if initial_state is None:
            initial_state = self.create_initial_state()

        nodes_explored = 0
        visited = set()

        counter = 0
        h0 = self._heuristic(initial_state)
        frontier = [(h0, counter, initial_state)]

        while frontier and nodes_explored < self.max_nodes:
            h_score, _, current = heapq.heappop(frontier)
            nodes_explored += 1

            state_hash = current.to_hashable()
            if state_hash in visited:
                continue
            visited.add(state_hash)

            if self._is_goal(current):
                schedule = self._build_schedule(current)
                sla_rate = schedule.get_sla_adherence_rate(self.pipeline.tasks)
                return SearchResult(
                    success=True,
                    actions=self._extract_actions(current),
                    final_state=current,
                    schedule=schedule,
                    nodes_explored=nodes_explored,
                    sla_adherence=sla_rate,
                    total_cost=current.cost,
                    message=f"Greedy found plan in {nodes_explored} nodes, "
                            f"cost={current.cost:.1f}, SLA={sla_rate:.0f}%"
                )

            for action in self._get_actions(current):
                new_state = self._apply_action(current, action)
                new_hash = new_state.to_hashable()
                if new_hash not in visited:
                    h = self._heuristic(new_state)
                    counter += 1
                    heapq.heappush(frontier, (h, counter, new_state))

        return SearchResult(
            success=False,
            nodes_explored=nodes_explored,
            message=f"Greedy exhausted search after {nodes_explored} nodes."
        )

    # ================================================================
    # Helpers
    # ================================================================

    def _build_schedule(self, state: PlannerState) -> Schedule:
        """Build a Schedule object from state's schedule entries."""
        schedule = Schedule()
        for task_id, resource_id, start, end in state.schedule:
            schedule.add_entry(ScheduleEntry(task_id, resource_id, start, end))
        return schedule

    def _extract_actions(self, state: PlannerState) -> list[PlannerAction]:
        """Extract the action sequence from a final state."""
        actions = []
        for desc in state.actions_taken:
            actions.append(PlannerAction(
                action_type="recorded",
                task_id="",
                description=desc
            ))
        return actions

    # ================================================================
    # Convenience: Re-plan from failure
    # ================================================================

    def replan_from_failure(self, failed_task_ids: list[str],
                             completed_task_ids: list[str],
                             current_time: float) -> SearchResult:
        """
        High-level method: given which tasks failed and which completed,
        find a recovery plan.

        Args:
            failed_task_ids: Tasks that have failed.
            completed_task_ids: Tasks that already completed successfully.
            current_time: Current simulation time.

        Returns:
            SearchResult with the recovery plan.
        """
        self.current_time = current_time

        statuses = {}
        for task_id in self.pipeline.tasks:
            if task_id in completed_task_ids:
                statuses[task_id] = "completed"
            elif task_id in failed_task_ids:
                statuses[task_id] = "failed"
            else:
                statuses[task_id] = "pending"

        # Build current schedule from completed tasks
        current_schedule = []
        for task_id in completed_task_ids:
            task = self.pipeline.tasks[task_id]
            # Approximate: we don't know exact times, but it doesn't matter
            # for forward planning
            current_schedule.append(
                (task_id, self.pipeline.resources[0].resource_id,
                 0, task.duration_estimate)
            )

        initial = self.create_initial_state(
            task_statuses=statuses,
            current_schedule=current_schedule
        )

        return self.search_astar(initial)
