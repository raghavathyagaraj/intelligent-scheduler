"""
knowledge_base.py — Knowledge Representation & Reasoning Engine.

Implements a First-Order Logic (FOL) knowledge base for the pipeline scheduler.
Stores facts as (predicate, subject, object) tuples and applies rules via
forward chaining (derive all consequences) and backward chaining (prove a goal).

Key capabilities:
- Store and query facts about tasks, resources, dependencies, and statuses.
- Forward chaining: apply rules repeatedly until no new facts are derived.
- Backward chaining: given a goal, trace back through rules to verify it.
- SLA risk detection: proactively identify tasks likely to miss deadlines.
- Tool selection reasoning: determine scheduling actions based on current state.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
from src.task_dag import Pipeline, Task, TaskStatus, Resource


@dataclass
class Fact:
    """
    A single fact in the knowledge base.

    Represented as a (predicate, subject, value) triple.
    Examples:
        Fact("task_status", "extract", "completed")
        Fact("depends_on", "transform", "extract")
        Fact("requires_cpu", "transform", 4)
        Fact("has_sla", "report", 360)
        Fact("ready_to_run", "transform", True)
        Fact("sla_at_risk", "report", True)
    """
    predicate: str
    subject: str
    value: Any

    def __hash__(self):
        return hash((self.predicate, self.subject, str(self.value)))

    def __eq__(self, other):
        if isinstance(other, Fact):
            return (self.predicate == other.predicate and
                    self.subject == other.subject and
                    self.value == other.value)
        return False

    def __repr__(self):
        return f"{self.predicate}({self.subject}, {self.value})"


@dataclass
class Rule:
    """
    An inference rule: IF conditions THEN conclusion.

    The condition_fn takes (kb: KnowledgeBase, subject: str) and returns True/False.
    The conclusion_fn takes (kb: KnowledgeBase, subject: str) and returns a list
    of Fact objects to add.

    Attributes:
        name: Human-readable rule name (for logging/debugging).
        description: What this rule does in plain English.
        condition_fn: Function that checks if the rule should fire.
        conclusion_fn: Function that produces new facts when the rule fires.
        applies_to: Which subjects this rule applies to ("tasks", "resources", or "all").
    """
    name: str
    description: str
    condition_fn: callable
    conclusion_fn: callable
    applies_to: str = "tasks"  # "tasks", "resources", or "all"


class KnowledgeBase:
    """
    First-Order Logic Knowledge Base with forward and backward chaining.

    Stores facts about the pipeline (tasks, resources, dependencies, statuses)
    and applies rules to derive new facts (readiness, SLA risks, scheduling
    recommendations).
    """

    def __init__(self):
        self.facts: set[Fact] = set()
        self.rules: list[Rule] = []
        self._derived_facts: set[Fact] = set()  # facts added by inference
        self._inference_log: list[str] = []       # log of rule firings

    # ================================================================
    # Fact Management
    # ================================================================

    def add_fact(self, predicate: str, subject: str, value: Any):
        """Add a single fact to the knowledge base."""
        fact = Fact(predicate, subject, value)
        self.facts.add(fact)

    def remove_fact(self, predicate: str, subject: str, value: Any):
        """Remove a specific fact."""
        fact = Fact(predicate, subject, value)
        self.facts.discard(fact)
        self._derived_facts.discard(fact)

    def remove_facts_by_predicate(self, predicate: str, subject: str):
        """Remove all facts with a given predicate and subject."""
        to_remove = {f for f in self.facts
                     if f.predicate == predicate and f.subject == subject}
        self.facts -= to_remove
        self._derived_facts -= to_remove

    def query(self, predicate: str, subject: str) -> list[Any]:
        """
        Query the KB: return all values for a given predicate and subject.

        Example: kb.query("task_status", "extract") → ["completed"]
        """
        return [f.value for f in self.facts
                if f.predicate == predicate and f.subject == subject]

    def query_single(self, predicate: str, subject: str) -> Optional[Any]:
        """Query and return the first matching value, or None."""
        results = self.query(predicate, subject)
        return results[0] if results else None

    def has_fact(self, predicate: str, subject: str, value: Any = None) -> bool:
        """
        Check if a fact exists.
        If value is None, checks if any fact with (predicate, subject) exists.
        """
        if value is not None:
            return Fact(predicate, subject, value) in self.facts
        return any(f.predicate == predicate and f.subject == subject
                   for f in self.facts)

    def get_all_subjects(self, predicate: str) -> list[str]:
        """Get all subjects that have a given predicate."""
        return list({f.subject for f in self.facts if f.predicate == predicate})

    def get_all_facts(self) -> list[Fact]:
        """Return all facts as a sorted list (for display)."""
        return sorted(self.facts, key=lambda f: (f.predicate, f.subject))

    def get_derived_facts(self) -> list[Fact]:
        """Return only facts that were derived by inference."""
        return sorted(self._derived_facts, key=lambda f: (f.predicate, f.subject))

    def get_inference_log(self) -> list[str]:
        """Return the log of rule firings from the last forward chain run."""
        return self._inference_log.copy()

    def clear_derived(self):
        """Remove all derived facts and reset inference state."""
        self.facts -= self._derived_facts
        self._derived_facts.clear()
        self._inference_log.clear()

    # ================================================================
    # Load from Pipeline
    # ================================================================

    def load_from_pipeline(self, pipeline: Pipeline):
        """
        Populate the KB with facts extracted from a Pipeline object.

        Adds facts about:
        - Task properties (duration, CPU, memory, priority, SLA, status)
        - Dependencies between tasks
        - Resource properties (capacity)
        """
        # Task facts
        for task_id, task in pipeline.tasks.items():
            self.add_fact("task_exists", task_id, True)
            self.add_fact("task_status", task_id, task.status.value)
            self.add_fact("duration_estimate", task_id, task.duration_estimate)
            self.add_fact("requires_cpu", task_id, task.cpu_required)
            self.add_fact("requires_memory", task_id, task.memory_required)
            self.add_fact("priority", task_id, task.priority)
            if task.sla_deadline is not None:
                self.add_fact("has_sla", task_id, task.sla_deadline)

        # Dependency facts
        for upstream, downstream in pipeline.dag.edges():
            self.add_fact("depends_on", downstream, upstream)

        # Resource facts
        for resource in pipeline.resources:
            self.add_fact("resource_exists", resource.resource_id, True)
            self.add_fact("resource_cpu_capacity", resource.resource_id,
                          resource.cpu_capacity)
            self.add_fact("resource_memory_capacity", resource.resource_id,
                          resource.memory_capacity)

    def update_task_status(self, task_id: str, new_status: str):
        """Update a task's status fact (removes old, adds new)."""
        self.remove_facts_by_predicate("task_status", task_id)
        self.add_fact("task_status", task_id, new_status)

    # ================================================================
    # Rule Registration
    # ================================================================

    def add_rule(self, rule: Rule):
        """Register an inference rule."""
        self.rules.append(rule)

    def register_default_rules(self):
        """Register the standard set of scheduling inference rules."""
        self.rules.clear()

        # Rule 1: Task is ready to run
        # IF task is pending AND all dependencies are completed → ready_to_run
        self.add_rule(Rule(
            name="ready_to_run",
            description="IF task is pending AND all dependencies completed THEN ready_to_run",
            condition_fn=self._condition_ready_to_run,
            conclusion_fn=self._conclude_ready_to_run,
            applies_to="tasks"
        ))

        # Rule 2: Task is blocked
        # IF task depends on a failed task → blocked
        self.add_rule(Rule(
            name="blocked_by_failure",
            description="IF task depends on a failed task THEN blocked",
            condition_fn=self._condition_blocked,
            conclusion_fn=self._conclude_blocked,
            applies_to="tasks"
        ))

        # Rule 3: SLA at risk (simple estimate)
        # IF task has SLA AND cumulative estimated time exceeds deadline → sla_at_risk
        self.add_rule(Rule(
            name="sla_at_risk",
            description="IF estimated completion > SLA deadline THEN sla_at_risk",
            condition_fn=self._condition_sla_at_risk,
            conclusion_fn=self._conclude_sla_at_risk,
            applies_to="tasks"
        ))

        # Rule 4: Resource insufficient
        # IF task requires more CPU than any single resource → resource_insufficient
        self.add_rule(Rule(
            name="resource_insufficient",
            description="IF task needs more CPU than any resource has THEN resource_insufficient",
            condition_fn=self._condition_resource_insufficient,
            conclusion_fn=self._conclude_resource_insufficient,
            applies_to="tasks"
        ))

        # Rule 5: High priority and pending
        # IF task has priority >= 8 AND is pending → urgent
        self.add_rule(Rule(
            name="urgent_task",
            description="IF priority >= 8 AND pending THEN urgent",
            condition_fn=self._condition_urgent,
            conclusion_fn=self._conclude_urgent,
            applies_to="tasks"
        ))

        # Rule 6: All dependencies met (helper fact)
        # IF every upstream task is completed → all_deps_met
        self.add_rule(Rule(
            name="all_deps_met",
            description="IF all upstream tasks completed THEN all_deps_met",
            condition_fn=self._condition_all_deps_met,
            conclusion_fn=self._conclude_all_deps_met,
            applies_to="tasks"
        ))

        # Rule 7: Cascading risk
        # IF a task is sla_at_risk AND another task depends on it AND that
        # task also has an SLA → cascading_sla_risk
        self.add_rule(Rule(
            name="cascading_sla_risk",
            description="IF upstream is sla_at_risk AND downstream has SLA THEN cascading_sla_risk",
            condition_fn=self._condition_cascading_risk,
            conclusion_fn=self._conclude_cascading_risk,
            applies_to="tasks"
        ))

        # Rule 8: Recommend retry
        # IF task is failed AND has no "no_retry" flag → recommend_retry
        self.add_rule(Rule(
            name="recommend_retry",
            description="IF task failed THEN recommend_retry",
            condition_fn=self._condition_recommend_retry,
            conclusion_fn=self._conclude_recommend_retry,
            applies_to="tasks"
        ))

    # ================================================================
    # Rule Condition & Conclusion Functions
    # ================================================================

    def _condition_ready_to_run(self, subject: str) -> bool:
        status = self.query_single("task_status", subject)
        if status != "pending":
            return False
        deps = self.query("depends_on", subject)
        if not deps:
            return True  # no dependencies → ready
        return all(
            self.query_single("task_status", dep) == "completed"
            for dep in deps
        )

    def _conclude_ready_to_run(self, subject: str) -> list[Fact]:
        return [Fact("ready_to_run", subject, True)]

    def _condition_blocked(self, subject: str) -> bool:
        status = self.query_single("task_status", subject)
        if status not in ("pending", "ready"):
            return False
        deps = self.query("depends_on", subject)
        return any(
            self.query_single("task_status", dep) == "failed"
            for dep in deps
        )

    def _conclude_blocked(self, subject: str) -> list[Fact]:
        # Find which dependency failed
        deps = self.query("depends_on", subject)
        failed_deps = [d for d in deps
                       if self.query_single("task_status", d) == "failed"]
        return [Fact("blocked", subject, failed_dep) for failed_dep in failed_deps]

    def _condition_sla_at_risk(self, subject: str) -> bool:
        sla = self.query_single("has_sla", subject)
        if sla is None:
            return False
        # Calculate cumulative time: sum of estimates along dependency chain
        cumulative = self._estimate_cumulative_time(subject)
        return cumulative > sla

    def _conclude_sla_at_risk(self, subject: str) -> list[Fact]:
        sla = self.query_single("has_sla", subject)
        cumulative = self._estimate_cumulative_time(subject)
        return [
            Fact("sla_at_risk", subject, True),
            Fact("sla_slack", subject, sla - cumulative),  # negative = overdue
        ]

    def _condition_resource_insufficient(self, subject: str) -> bool:
        cpu_needed = self.query_single("requires_cpu", subject)
        if cpu_needed is None:
            return False
        resources = self.get_all_subjects("resource_cpu_capacity")
        if not resources:
            return True
        max_capacity = max(
            self.query_single("resource_cpu_capacity", r) or 0
            for r in resources
        )
        return cpu_needed > max_capacity

    def _conclude_resource_insufficient(self, subject: str) -> list[Fact]:
        return [Fact("resource_insufficient", subject, True)]

    def _condition_urgent(self, subject: str) -> bool:
        status = self.query_single("task_status", subject)
        priority = self.query_single("priority", subject)
        return status == "pending" and priority is not None and priority >= 8

    def _conclude_urgent(self, subject: str) -> list[Fact]:
        return [Fact("urgent", subject, True)]

    def _condition_all_deps_met(self, subject: str) -> bool:
        deps = self.query("depends_on", subject)
        if not deps:
            return True
        return all(
            self.query_single("task_status", dep) == "completed"
            for dep in deps
        )

    def _conclude_all_deps_met(self, subject: str) -> list[Fact]:
        return [Fact("all_deps_met", subject, True)]

    def _condition_cascading_risk(self, subject: str) -> bool:
        # Does this task have an SLA?
        sla = self.query_single("has_sla", subject)
        if sla is None:
            return False
        # Does any upstream task have sla_at_risk?
        deps = self.query("depends_on", subject)
        return any(
            self.has_fact("sla_at_risk", dep, True) for dep in deps
        )

    def _conclude_cascading_risk(self, subject: str) -> list[Fact]:
        return [Fact("cascading_sla_risk", subject, True)]

    def _condition_recommend_retry(self, subject: str) -> bool:
        status = self.query_single("task_status", subject)
        return status == "failed" and not self.has_fact("no_retry", subject, True)

    def _conclude_recommend_retry(self, subject: str) -> list[Fact]:
        return [Fact("recommend_retry", subject, True)]

    # ================================================================
    # Helper: Cumulative Time Estimation
    # ================================================================

    def _estimate_cumulative_time(self, task_id: str,
                                   visited: Optional[set] = None) -> float:
        """
        Estimate total time from pipeline start to task completion.
        Uses longest path through dependencies (critical path to this task).
        """
        if visited is None:
            visited = set()
        if task_id in visited:
            return 0  # cycle protection
        visited.add(task_id)

        own_duration = self.query_single("duration_estimate", task_id) or 0
        deps = self.query("depends_on", task_id)

        if not deps:
            return own_duration

        # Longest upstream path (critical path to this node)
        max_upstream = max(
            self._estimate_cumulative_time(dep, visited.copy())
            for dep in deps
        )
        return max_upstream + own_duration

    # ================================================================
    # Forward Chaining
    # ================================================================

    def forward_chain(self, max_iterations: int = 50) -> int:
        """
        Forward chaining: apply all rules repeatedly until no new facts
        are derived (fixpoint) or max iterations reached.

        Returns:
            Number of new facts derived.
        """
        self._inference_log.clear()
        total_new = 0

        for iteration in range(max_iterations):
            new_facts_this_round = []

            for rule in self.rules:
                # Determine which subjects to apply the rule to
                if rule.applies_to == "tasks":
                    subjects = self.get_all_subjects("task_exists")
                elif rule.applies_to == "resources":
                    subjects = self.get_all_subjects("resource_exists")
                else:
                    subjects = list(
                        set(self.get_all_subjects("task_exists")) |
                        set(self.get_all_subjects("resource_exists"))
                    )

                for subject in subjects:
                    # Check if rule condition is met
                    if rule.condition_fn(subject):
                        # Get conclusion facts
                        conclusions = rule.conclusion_fn(subject)
                        for fact in conclusions:
                            if fact not in self.facts:
                                new_facts_this_round.append((rule.name, fact))

            if not new_facts_this_round:
                break  # fixpoint reached

            for rule_name, fact in new_facts_this_round:
                self.facts.add(fact)
                self._derived_facts.add(fact)
                self._inference_log.append(
                    f"[{rule_name}] Derived: {fact}"
                )
                total_new += 1

        return total_new

    # ================================================================
    # Backward Chaining
    # ================================================================

    def backward_chain(self, goal_predicate: str, goal_subject: str,
                        goal_value: Any = None, depth: int = 0,
                        max_depth: int = 20) -> tuple[bool, list[str]]:
        """
        Backward chaining: given a goal fact, try to prove it by tracing
        back through rules.

        Args:
            goal_predicate: The predicate to prove.
            goal_subject: The subject to prove it for.
            goal_value: Specific value to match (None = any value).
            depth: Current recursion depth.
            max_depth: Maximum recursion depth to prevent infinite loops.

        Returns:
            (proven: bool, explanation: list[str]) where explanation traces
            the reasoning chain.
        """
        indent = "  " * depth
        explanation = []

        if depth > max_depth:
            return False, [f"{indent}Max depth reached."]

        # Check if fact already exists in KB
        if goal_value is not None:
            if self.has_fact(goal_predicate, goal_subject, goal_value):
                explanation.append(
                    f"{indent}KNOWN: {goal_predicate}({goal_subject}, {goal_value})"
                )
                return True, explanation
        else:
            results = self.query(goal_predicate, goal_subject)
            if results:
                explanation.append(
                    f"{indent}KNOWN: {goal_predicate}({goal_subject}, {results[0]})"
                )
                return True, explanation

        # Try to derive via rules
        for rule in self.rules:
            # Check if any conclusion of this rule could produce our goal
            # We test by checking the condition and seeing if the conclusion matches
            if rule.applies_to == "tasks":
                subjects_pool = self.get_all_subjects("task_exists")
            elif rule.applies_to == "resources":
                subjects_pool = self.get_all_subjects("resource_exists")
            else:
                subjects_pool = list(
                    set(self.get_all_subjects("task_exists")) |
                    set(self.get_all_subjects("resource_exists"))
                )

            if goal_subject not in subjects_pool:
                continue

            if rule.condition_fn(goal_subject):
                conclusions = rule.conclusion_fn(goal_subject)
                for fact in conclusions:
                    if fact.predicate == goal_predicate and fact.subject == goal_subject:
                        if goal_value is None or fact.value == goal_value:
                            explanation.append(
                                f"{indent}RULE [{rule.name}]: {rule.description}"
                            )
                            explanation.append(
                                f"{indent}  → DERIVED: {fact}"
                            )
                            # Add the fact to KB
                            self.facts.add(fact)
                            self._derived_facts.add(fact)
                            return True, explanation

        explanation.append(
            f"{indent}CANNOT PROVE: {goal_predicate}({goal_subject}, {goal_value})"
        )
        return False, explanation

    # ================================================================
    # High-Level Query Methods (used by the scheduler agent)
    # ================================================================

    def get_ready_tasks(self) -> list[str]:
        """Get all tasks that are ready to run (after forward chaining)."""
        return [f.subject for f in self.facts
                if f.predicate == "ready_to_run" and f.value is True]

    def get_blocked_tasks(self) -> list[tuple[str, str]]:
        """Get all blocked tasks and what's blocking them."""
        return [(f.subject, f.value) for f in self.facts
                if f.predicate == "blocked"]

    def get_sla_risks(self) -> list[dict]:
        """Get all tasks with SLA at risk, with details."""
        risks = []
        for f in self.facts:
            if f.predicate == "sla_at_risk" and f.value is True:
                task_id = f.subject
                sla = self.query_single("has_sla", task_id)
                slack = self.query_single("sla_slack", task_id)
                risks.append({
                    "task_id": task_id,
                    "sla_deadline": sla,
                    "slack_minutes": slack,
                    "cascading": self.has_fact("cascading_sla_risk", task_id, True),
                })
        return risks

    def get_urgent_tasks(self) -> list[str]:
        """Get all tasks flagged as urgent."""
        return [f.subject for f in self.facts
                if f.predicate == "urgent" and f.value is True]

    def get_retry_recommendations(self) -> list[str]:
        """Get all tasks recommended for retry."""
        return [f.subject for f in self.facts
                if f.predicate == "recommend_retry" and f.value is True]

    # ================================================================
    # Display / Debug
    # ================================================================

    def summary(self) -> str:
        """Return a human-readable summary of the KB state after inference."""
        lines = [
            f"Knowledge Base Summary",
            f"  Total facts: {len(self.facts)}",
            f"  Derived facts: {len(self._derived_facts)}",
            f"  Rules: {len(self.rules)}",
        ]

        ready = self.get_ready_tasks()
        if ready:
            lines.append(f"  Ready to run: {ready}")

        blocked = self.get_blocked_tasks()
        if blocked:
            lines.append(f"  Blocked: {blocked}")

        risks = self.get_sla_risks()
        if risks:
            for r in risks:
                lines.append(
                    f"  SLA AT RISK: {r['task_id']} "
                    f"(deadline={r['sla_deadline']:.0f}m, "
                    f"slack={r['slack_minutes']:.0f}m"
                    f"{', CASCADING' if r['cascading'] else ''})"
                )

        urgent = self.get_urgent_tasks()
        if urgent:
            lines.append(f"  Urgent: {urgent}")

        retries = self.get_retry_recommendations()
        if retries:
            lines.append(f"  Recommend retry: {retries}")

        return "\n".join(lines)
