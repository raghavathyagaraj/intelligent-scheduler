"""
task_dag.py — Core data model for the Intelligent Pipeline Scheduler.

Defines the fundamental building blocks:
- Task: A unit of work with duration, resource needs, priority, and SLA.
- Resource: A compute resource (server/cluster) with CPU and memory capacity.
- ScheduleEntry: A single assignment of a task to a resource at a specific time.
- Schedule: A complete mapping of all tasks to resources and time slots.
- Pipeline: A DAG (Directed Acyclic Graph) of tasks with dependency edges.
"""

import json
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


class TaskStatus(Enum):
    """Possible states of a task during execution."""
    PENDING = "pending"
    READY = "ready"          # all dependencies met, waiting to be scheduled
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Task:
    """
    A single unit of work in a data pipeline.

    Attributes:
        task_id: Unique identifier (e.g., "extract_orders").
        name: Human-readable name.
        duration_estimate: Estimated runtime in minutes.
        cpu_required: Number of CPU cores needed.
        memory_required: Memory needed in GB.
        priority: Higher number = higher priority (1-10).
        sla_deadline: Must complete by this time (minutes from midnight).
                      None means no hard deadline.
        status: Current execution status.
    """
    task_id: str
    name: str
    duration_estimate: float          # minutes
    cpu_required: int = 2             # CPU cores
    memory_required: float = 4.0      # GB
    priority: int = 5                 # 1 (lowest) to 10 (highest)
    sla_deadline: Optional[float] = None  # minutes from midnight (e.g., 360 = 6:00 AM)
    status: TaskStatus = TaskStatus.PENDING

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False

    def __repr__(self):
        return f"Task({self.task_id}, est={self.duration_estimate}m, cpu={self.cpu_required}, status={self.status.value})"

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "duration_estimate": self.duration_estimate,
            "cpu_required": self.cpu_required,
            "memory_required": self.memory_required,
            "priority": self.priority,
            "sla_deadline": self.sla_deadline,
            "status": self.status.value,
        }


@dataclass
class Resource:
    """
    A compute resource (server, cluster node, etc.)

    Attributes:
        resource_id: Unique identifier (e.g., "server_1").
        name: Human-readable name.
        cpu_capacity: Total CPU cores available.
        memory_capacity: Total memory in GB.
    """
    resource_id: str
    name: str
    cpu_capacity: int = 8          # total CPU cores
    memory_capacity: float = 32.0  # total GB

    def __hash__(self):
        return hash(self.resource_id)

    def __eq__(self, other):
        if isinstance(other, Resource):
            return self.resource_id == other.resource_id
        return False

    def __repr__(self):
        return f"Resource({self.resource_id}, cpu={self.cpu_capacity}, mem={self.memory_capacity}GB)"

    def to_dict(self) -> dict:
        return {
            "resource_id": self.resource_id,
            "name": self.name,
            "cpu_capacity": self.cpu_capacity,
            "memory_capacity": self.memory_capacity,
        }


@dataclass
class ScheduleEntry:
    """
    A single scheduling decision: which task runs on which resource, when.

    Attributes:
        task_id: The task being scheduled.
        resource_id: The resource it's assigned to.
        start_time: Start time in minutes from midnight.
        end_time: Expected end time (start_time + duration).
    """
    task_id: str
    resource_id: str
    start_time: float   # minutes from midnight
    end_time: float     # minutes from midnight

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def overlaps_with(self, other: "ScheduleEntry") -> bool:
        """Check if two schedule entries overlap in time."""
        return (self.start_time < other.end_time and
                other.start_time < self.end_time)

    def __repr__(self):
        return (f"ScheduleEntry({self.task_id} → {self.resource_id}, "
                f"{self.start_time:.0f}-{self.end_time:.0f}m)")


class Schedule:
    """
    A complete schedule: collection of ScheduleEntry objects.

    Provides methods to query and validate the schedule.
    """

    def __init__(self):
        self.entries: list[ScheduleEntry] = []

    def add_entry(self, entry: ScheduleEntry):
        """Add a scheduling decision."""
        self.entries.append(entry)

    def get_entry_for_task(self, task_id: str) -> Optional[ScheduleEntry]:
        """Get the schedule entry for a specific task."""
        for entry in self.entries:
            if entry.task_id == task_id:
                return entry
        return None

    def get_entries_for_resource(self, resource_id: str) -> list[ScheduleEntry]:
        """Get all entries assigned to a specific resource, sorted by start time."""
        return sorted(
            [e for e in self.entries if e.resource_id == resource_id],
            key=lambda e: e.start_time
        )

    def get_makespan(self) -> float:
        """Total time from first start to last end."""
        if not self.entries:
            return 0.0
        earliest = min(e.start_time for e in self.entries)
        latest = max(e.end_time for e in self.entries)
        return latest - earliest

    def get_resource_utilization(self, resource: Resource,
                                 time_window: Optional[tuple[float, float]] = None
                                 ) -> float:
        """
        Calculate resource CPU utilization as a percentage.

        Args:
            resource: The resource to check.
            time_window: (start, end) in minutes. Defaults to full schedule span.

        Returns:
            Utilization percentage (0.0 to 100.0).
        """
        entries = self.get_entries_for_resource(resource.resource_id)
        if not entries:
            return 0.0

        if time_window is None:
            window_start = min(e.start_time for e in entries)
            window_end = max(e.end_time for e in entries)
        else:
            window_start, window_end = time_window

        window_length = window_end - window_start
        if window_length <= 0:
            return 0.0

        # Sum of task durations on this resource / window length
        total_busy = sum(e.duration for e in entries)
        return min((total_busy / window_length) * 100.0, 100.0)

    def check_sla_compliance(self, tasks: dict[str, Task]) -> dict[str, bool]:
        """
        Check which tasks meet their SLA deadlines.

        Returns:
            Dict of task_id → True (met SLA) or False (violated).
            Tasks without SLAs are not included.
        """
        compliance = {}
        for entry in self.entries:
            task = tasks.get(entry.task_id)
            if task and task.sla_deadline is not None:
                compliance[entry.task_id] = entry.end_time <= task.sla_deadline
        return compliance

    def get_sla_adherence_rate(self, tasks: dict[str, Task]) -> float:
        """Percentage of SLA-bound tasks that meet their deadline."""
        compliance = self.check_sla_compliance(tasks)
        if not compliance:
            return 100.0
        met = sum(1 for v in compliance.values() if v)
        return (met / len(compliance)) * 100.0

    def __repr__(self):
        return f"Schedule({len(self.entries)} entries, makespan={self.get_makespan():.0f}m)"

    def to_list(self) -> list[dict]:
        return [
            {
                "task_id": e.task_id,
                "resource_id": e.resource_id,
                "start_time": e.start_time,
                "end_time": e.end_time,
            }
            for e in self.entries
        ]


class Pipeline:
    """
    A DAG of tasks representing a data pipeline.

    Uses networkx.DiGraph under the hood. Edges represent dependencies:
    an edge from A → B means 'A must complete before B can start'.

    Provides methods for topological sort, critical path, ready-task detection,
    and loading/saving from JSON.
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.dag = nx.DiGraph()
        self.tasks: dict[str, Task] = {}
        self.resources: list[Resource] = []

    def add_task(self, task: Task):
        """Add a task to the pipeline."""
        self.tasks[task.task_id] = task
        self.dag.add_node(task.task_id)

    def add_dependency(self, upstream_id: str, downstream_id: str):
        """
        Add a dependency: upstream must finish before downstream starts.
        Edge direction: upstream → downstream.
        """
        if upstream_id not in self.tasks:
            raise ValueError(f"Upstream task '{upstream_id}' not found in pipeline.")
        if downstream_id not in self.tasks:
            raise ValueError(f"Downstream task '{downstream_id}' not found in pipeline.")
        if upstream_id == downstream_id:
            raise ValueError("A task cannot depend on itself.")

        self.dag.add_edge(upstream_id, downstream_id)

        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.dag):
            self.dag.remove_edge(upstream_id, downstream_id)
            raise ValueError(
                f"Adding dependency {upstream_id} → {downstream_id} would create a cycle."
            )

    def add_resource(self, resource: Resource):
        """Add a compute resource to the pipeline environment."""
        self.resources.append(resource)

    def get_dependencies(self, task_id: str) -> list[str]:
        """Get all upstream tasks that must complete before this task."""
        return list(self.dag.predecessors(task_id))

    def get_dependents(self, task_id: str) -> list[str]:
        """Get all downstream tasks that depend on this task."""
        return list(self.dag.successors(task_id))

    def get_topological_order(self) -> list[str]:
        """Return tasks in a valid execution order (topological sort)."""
        return list(nx.topological_sort(self.dag))

    def get_ready_tasks(self) -> list[str]:
        """
        Get tasks that are ready to run:
        - Status is PENDING or READY
        - All upstream dependencies are COMPLETED
        """
        ready = []
        for task_id, task in self.tasks.items():
            if task.status not in (TaskStatus.PENDING, TaskStatus.READY):
                continue
            deps = self.get_dependencies(task_id)
            if all(self.tasks[d].status == TaskStatus.COMPLETED for d in deps):
                ready.append(task_id)
        return ready

    def get_critical_path(self) -> tuple[list[str], float]:
        """
        Find the critical path — the longest path through the DAG by duration.

        Returns:
            (path, total_duration) where path is a list of task_ids.
        """
        # Use negative weights to find longest path via shortest path algorithm
        weighted_dag = nx.DiGraph()
        for u, v in self.dag.edges():
            weighted_dag.add_edge(u, v, weight=-self.tasks[u].duration_estimate)

        # Add virtual source connected to all roots (no predecessors)
        roots = [n for n in self.dag.nodes() if self.dag.in_degree(n) == 0]
        leaves = [n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]

        weighted_dag.add_node("__source__")
        weighted_dag.add_node("__sink__")

        for root in roots:
            weighted_dag.add_edge("__source__", root, weight=0)
        for leaf in leaves:
            weighted_dag.add_edge(leaf, "__sink__", weight=-self.tasks[leaf].duration_estimate)

        # Shortest path with negative weights = longest path
        path = nx.shortest_path(weighted_dag, "__source__", "__sink__", weight="weight")

        # Remove virtual nodes
        path = [p for p in path if p not in ("__source__", "__sink__")]
        total_duration = sum(self.tasks[t].duration_estimate for t in path)

        return path, total_duration

    def get_all_roots(self) -> list[str]:
        """Tasks with no dependencies (entry points)."""
        return [n for n in self.dag.nodes() if self.dag.in_degree(n) == 0]

    def get_all_leaves(self) -> list[str]:
        """Tasks with no dependents (exit points)."""
        return [n for n in self.dag.nodes() if self.dag.out_degree(n) == 0]

    def reset_all_statuses(self):
        """Reset all task statuses to PENDING."""
        for task in self.tasks.values():
            task.status = TaskStatus.PENDING

    def validate(self) -> list[str]:
        """
        Validate the pipeline for common issues.

        Returns:
            List of warning/error messages. Empty list = valid.
        """
        issues = []

        if not nx.is_directed_acyclic_graph(self.dag):
            issues.append("ERROR: Pipeline contains cycles.")

        if not self.resources:
            issues.append("WARNING: No resources defined.")

        # Check if any task needs more resources than any single resource can provide
        for task_id, task in self.tasks.items():
            can_fit = False
            for res in self.resources:
                if (task.cpu_required <= res.cpu_capacity and
                        task.memory_required <= res.memory_capacity):
                    can_fit = True
                    break
            if not can_fit and self.resources:
                issues.append(
                    f"ERROR: Task '{task_id}' requires {task.cpu_required} CPUs / "
                    f"{task.memory_required}GB but no resource can fit it."
                )

        # Check for orphan nodes (no edges at all in a multi-task pipeline)
        if len(self.tasks) > 1:
            isolated = list(nx.isolates(self.dag))
            for node in isolated:
                issues.append(
                    f"WARNING: Task '{node}' has no dependencies or dependents."
                )

        # Check SLA feasibility on critical path
        cp_path, cp_duration = self.get_critical_path()
        for task_id in cp_path:
            task = self.tasks[task_id]
            if task.sla_deadline is not None and cp_duration > task.sla_deadline:
                issues.append(
                    f"WARNING: Critical path duration ({cp_duration:.0f}m) exceeds "
                    f"SLA deadline for '{task_id}' ({task.sla_deadline:.0f}m). "
                    f"SLA may be infeasible."
                )

        return issues

    def summary(self) -> str:
        """Return a human-readable summary of the pipeline."""
        cp_path, cp_duration = self.get_critical_path()
        lines = [
            f"Pipeline: {self.name}",
            f"  Tasks: {len(self.tasks)}",
            f"  Dependencies: {self.dag.number_of_edges()}",
            f"  Resources: {len(self.resources)}",
            f"  Root tasks: {self.get_all_roots()}",
            f"  Leaf tasks: {self.get_all_leaves()}",
            f"  Critical path: {' → '.join(cp_path)} ({cp_duration:.0f}m)",
            f"  Topological order: {self.get_topological_order()}",
        ]
        sla_tasks = [t for t in self.tasks.values() if t.sla_deadline is not None]
        if sla_tasks:
            lines.append(f"  SLA-bound tasks: {[t.task_id for t in sla_tasks]}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize pipeline to dictionary (for JSON export)."""
        return {
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks.values()],
            "dependencies": [
                {"upstream": u, "downstream": v}
                for u, v in self.dag.edges()
            ],
            "resources": [r.to_dict() for r in self.resources],
        }

    def save_json(self, filepath: str):
        """Save pipeline definition to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> "Pipeline":
        """Load a pipeline definition from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        pipeline = cls(name=data.get("name", "pipeline"))

        # Add tasks
        for t in data["tasks"]:
            task = Task(
                task_id=t["task_id"],
                name=t["name"],
                duration_estimate=t["duration_estimate"],
                cpu_required=t.get("cpu_required", 2),
                memory_required=t.get("memory_required", 4.0),
                priority=t.get("priority", 5),
                sla_deadline=t.get("sla_deadline"),
            )
            pipeline.add_task(task)

        # Add dependencies
        for dep in data["dependencies"]:
            pipeline.add_dependency(dep["upstream"], dep["downstream"])

        # Add resources
        for r in data.get("resources", []):
            resource = Resource(
                resource_id=r["resource_id"],
                name=r["name"],
                cpu_capacity=r.get("cpu_capacity", 8),
                memory_capacity=r.get("memory_capacity", 32.0),
            )
            pipeline.add_resource(resource)

        return pipeline

    def __repr__(self):
        return f"Pipeline({self.name}, {len(self.tasks)} tasks, {self.dag.number_of_edges()} deps)"
