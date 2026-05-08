"""
simulator.py — Pipeline Execution Simulator.

Simulates the execution of a scheduled pipeline with realistic behaviors:
- Normal: tasks run at exactly their estimated duration.
- Stochastic: runtimes vary ±30% randomly.
- Failure: random task failures with configurable probability.
- Spike: one random task takes 3-5x its estimated duration.

Feeds events to the scheduler agent for re-planning decisions.
"""

import random
from dataclasses import dataclass, field
from typing import Optional
from src.task_dag import Pipeline, Task, TaskStatus, Schedule, ScheduleEntry


@dataclass
class SimEvent:
    """An event that occurred during simulation."""
    time: float
    task_id: str
    event_type: str   # "started", "completed", "failed", "sla_violated"
    details: str = ""
    actual_duration: Optional[float] = None


@dataclass
class SimResult:
    """Complete result of a simulation run."""
    events: list[SimEvent] = field(default_factory=list)
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)
    actual_durations: dict[str, float] = field(default_factory=dict)
    sla_violations: list[str] = field(default_factory=list)
    total_time: float = 0.0
    mode: str = "normal"

    def summary(self) -> str:
        lines = [
            f"Simulation Result (mode={self.mode})",
            f"  Completed: {len(self.completed_tasks)} tasks",
            f"  Failed: {len(self.failed_tasks)} tasks",
            f"  SLA violations: {len(self.sla_violations)}",
            f"  Total time: {self.total_time:.0f} minutes",
        ]
        if self.failed_tasks:
            lines.append(f"  Failed tasks: {self.failed_tasks}")
        if self.sla_violations:
            lines.append(f"  SLA violated: {self.sla_violations}")
        return "\n".join(lines)


class Simulator:
    """
    Simulates pipeline execution on a given schedule.

    Modes:
        normal: Tasks run at exact estimated duration.
        stochastic: Runtimes vary uniformly by ±variance_pct.
        failure: Random task failures with given probability.
        spike: One random task takes spike_multiplier × its estimate.
    """

    def __init__(self, pipeline: Pipeline,
                 mode: str = "normal",
                 failure_rate: float = 0.1,
                 variance_pct: float = 0.3,
                 spike_multiplier: float = 4.0,
                 seed: Optional[int] = None):
        """
        Args:
            pipeline: The pipeline to simulate.
            mode: "normal", "stochastic", "failure", or "spike".
            failure_rate: Probability of task failure (for failure mode).
            variance_pct: Runtime variance as fraction (for stochastic mode).
            spike_multiplier: How much slower the spike task runs.
            seed: Random seed for reproducibility.
        """
        if mode not in ("normal", "stochastic", "failure", "spike"):
            raise ValueError(f"Invalid mode: {mode}")

        self.pipeline = pipeline
        self.mode = mode
        self.failure_rate = failure_rate
        self.variance_pct = variance_pct
        self.spike_multiplier = spike_multiplier
        self.rng = random.Random(seed)

        # Pick spike task once (for spike mode)
        task_ids = list(pipeline.tasks.keys())
        self._spike_task = self.rng.choice(task_ids) if task_ids else None

    def _get_actual_duration(self, task: Task) -> float:
        """Determine actual runtime based on simulation mode."""
        estimate = task.duration_estimate

        if self.mode == "normal":
            return estimate

        elif self.mode == "stochastic":
            low = estimate * (1 - self.variance_pct)
            high = estimate * (1 + self.variance_pct)
            return self.rng.uniform(low, high)

        elif self.mode == "failure":
            # Duration is stochastic even for non-failed tasks
            low = estimate * (1 - self.variance_pct * 0.5)
            high = estimate * (1 + self.variance_pct * 0.5)
            return self.rng.uniform(low, high)

        elif self.mode == "spike":
            if task.task_id == self._spike_task:
                return estimate * self.spike_multiplier
            else:
                low = estimate * 0.9
                high = estimate * 1.1
                return self.rng.uniform(low, high)

        return estimate

    def _should_fail(self, task: Task) -> bool:
        """Determine if a task fails during this simulation."""
        if self.mode == "failure":
            return self.rng.random() < self.failure_rate
        return False

    def run(self, schedule: Schedule) -> SimResult:
        """
        Execute the simulation on a given schedule.

        Processes tasks in schedule order, applies mode-specific behaviors,
        and records all events.

        Args:
            schedule: The schedule to simulate.

        Returns:
            SimResult with all events and outcomes.
        """
        result = SimResult(mode=self.mode)

        # Sort entries by start time
        sorted_entries = sorted(schedule.entries, key=lambda e: e.start_time)

        for entry in sorted_entries:
            task_id = entry.task_id
            task = self.pipeline.tasks.get(task_id)
            if task is None:
                continue

            # Check if dependencies are met
            deps = self.pipeline.get_dependencies(task_id)
            deps_failed = any(d in result.failed_tasks for d in deps)
            if deps_failed:
                # Skip this task — upstream failed
                result.failed_tasks.append(task_id)
                result.events.append(SimEvent(
                    time=entry.start_time,
                    task_id=task_id,
                    event_type="failed",
                    details="Upstream dependency failed"
                ))
                continue

            # Task starts
            result.events.append(SimEvent(
                time=entry.start_time,
                task_id=task_id,
                event_type="started",
                details=f"Started on {entry.resource_id}"
            ))

            # Determine if task fails
            if self._should_fail(task):
                # Task fails partway through
                fail_time = entry.start_time + self._get_actual_duration(task) * 0.5
                result.failed_tasks.append(task_id)
                result.events.append(SimEvent(
                    time=fail_time,
                    task_id=task_id,
                    event_type="failed",
                    details="Task execution failed"
                ))
                continue

            # Task completes
            actual_duration = self._get_actual_duration(task)
            actual_end = entry.start_time + actual_duration
            result.actual_durations[task_id] = actual_duration
            result.completed_tasks.append(task_id)

            result.events.append(SimEvent(
                time=actual_end,
                task_id=task_id,
                event_type="completed",
                actual_duration=actual_duration,
                details=f"Completed in {actual_duration:.1f}m "
                        f"(estimated {task.duration_estimate:.0f}m)"
            ))

            # Check SLA
            if task.sla_deadline is not None and actual_end > task.sla_deadline:
                result.sla_violations.append(task_id)
                result.events.append(SimEvent(
                    time=actual_end,
                    task_id=task_id,
                    event_type="sla_violated",
                    details=f"Finished at {actual_end:.0f}m, "
                            f"deadline was {task.sla_deadline:.0f}m"
                ))

        # Total time
        if result.events:
            result.total_time = max(e.time for e in result.events)

        return result

    def get_spike_task(self) -> Optional[str]:
        """Return which task is the spike task (for spike mode)."""
        return self._spike_task
