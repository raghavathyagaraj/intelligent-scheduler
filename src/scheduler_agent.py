"""
scheduler_agent.py — The Intelligent Scheduler Agent.

Orchestrates all AI components:
1. Load pipeline → populate Knowledge Base
2. Forward chain → identify ready tasks, SLA risks, blocked tasks
3. CSP Solver → generate initial schedule
4. Simulator → execute schedule (with potential failures)
5. Search Planner → re-plan if failures occur
6. Learning → record actual runtimes, improve estimates

Agent type: Learning utility-based agent.
- Utility-based: evaluates schedule quality (SLA adherence, makespan, utilization).
- Learning: improves runtime estimates over iterations via EWMA.
"""

from dataclasses import dataclass, field
from typing import Optional
from src.task_dag import Pipeline, Schedule, ScheduleEntry
from src.knowledge_base import KnowledgeBase
from src.csp_solver import CSPSolver, CSPResult
from src.search_planner import SearchPlanner, SearchResult
from src.learning import RuntimeEstimator
from src.simulator import Simulator, SimResult


@dataclass
class AgentMetrics:
    """Metrics from a single agent run."""
    makespan: float = 0.0
    sla_adherence: float = 0.0
    resource_utilization: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_total: int = 0
    csp_nodes_explored: int = 0
    csp_backtracks: int = 0
    replan_count: int = 0
    search_nodes_explored: int = 0
    learning_mae: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "Agent Metrics:",
            f"  Makespan: {self.makespan:.0f} minutes",
            f"  SLA adherence: {self.sla_adherence:.1f}%",
            f"  Tasks: {self.tasks_completed}/{self.tasks_total} completed, "
            f"{self.tasks_failed} failed",
            f"  CSP: {self.csp_nodes_explored} nodes, {self.csp_backtracks} backtracks",
            f"  Re-plans: {self.replan_count}",
            f"  Search nodes (total): {self.search_nodes_explored}",
        ]
        if self.learning_mae is not None:
            lines.append(f"  Learning MAE: {self.learning_mae:.2f} minutes")
        return "\n".join(lines)


@dataclass
class AgentRunResult:
    """Complete result of an agent run (one iteration)."""
    iteration: int
    schedule: Optional[Schedule] = None
    sim_result: Optional[SimResult] = None
    replan_results: list[SearchResult] = field(default_factory=list)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    kb_summary: str = ""
    log: list[str] = field(default_factory=list)

    def print_log(self):
        for entry in self.log:
            print(entry)


class SchedulerAgent:
    """
    The main intelligent scheduler agent.

    Orchestrates: KB → CSP → Simulate → Detect failures → Replan → Learn.

    Usage:
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="normal")
        print(result.metrics.summary())

        # Multi-iteration with learning:
        for i in range(10):
            result = agent.run_iteration(i+1, mode="stochastic")
    """

    def __init__(self, pipeline: Pipeline,
                 time_horizon: float = 480.0,
                 time_step: float = 15.0,
                 ewma_alpha: float = 0.3):
        self.pipeline = pipeline
        self.time_horizon = time_horizon
        self.time_step = time_step

        # Initialize components
        self.kb = KnowledgeBase()
        self.estimator = RuntimeEstimator(alpha=ewma_alpha)
        self.estimator.set_initial_estimates_from_pipeline(pipeline)

        # Track iteration count
        self._iteration = 0

    # ================================================================
    # Main Run Loop
    # ================================================================

    def run(self, mode: str = "normal", failure_rate: float = 0.1,
            seed: Optional[int] = None) -> AgentRunResult:
        """
        Execute one full agent cycle:
        1. Build KB → forward chain → detect risks
        2. Solve CSP → get initial schedule
        3. Simulate execution
        4. If failures → re-plan using A* search
        5. Record runtimes → update learning

        Args:
            mode: Simulation mode ("normal", "stochastic", "failure", "spike").
            failure_rate: Probability of failure per task (for failure mode).
            seed: Random seed for reproducibility.

        Returns:
            AgentRunResult with full details.
        """
        self._iteration += 1
        result = AgentRunResult(iteration=self._iteration)
        log = result.log

        log.append(f"{'='*60}")
        log.append(f"AGENT RUN — Iteration {self._iteration} (mode={mode})")
        log.append(f"{'='*60}")

        # Step 1: Knowledge Base
        log.append("\n[STEP 1] Building Knowledge Base...")
        self.pipeline.reset_all_statuses()
        self.kb = KnowledgeBase()
        self.kb.load_from_pipeline(self.pipeline)
        self.kb.register_default_rules()

        # Update KB with learned estimates
        for task_id in self.pipeline.tasks:
            learned_est = self.estimator.get_estimate(task_id)
            self.kb.remove_facts_by_predicate("duration_estimate", task_id)
            self.kb.add_fact("duration_estimate", task_id, learned_est)

        new_facts = self.kb.forward_chain()
        log.append(f"  Derived {new_facts} facts via forward chaining")

        ready = self.kb.get_ready_tasks()
        log.append(f"  Ready tasks: {ready}")

        risks = self.kb.get_sla_risks()
        if risks:
            for r in risks:
                log.append(f"  ⚠ SLA AT RISK: {r['task_id']} "
                           f"(slack={r['slack_minutes']:.0f}m)")

        result.kb_summary = self.kb.summary()

        # Step 2: CSP Solver
        log.append("\n[STEP 2] Solving CSP for initial schedule...")

        # Update pipeline with learned estimates for CSP
        for task_id, task in self.pipeline.tasks.items():
            task.duration_estimate = self.estimator.get_estimate(task_id)

        solver = CSPSolver(
            self.pipeline,
            time_horizon=self.time_horizon,
            time_step=self.time_step
        )
        csp_result = solver.solve()

        result.metrics.csp_nodes_explored = csp_result.nodes_explored
        result.metrics.csp_backtracks = csp_result.backtracks

        if not csp_result.success:
            log.append(f"  ✗ CSP failed: {csp_result.message}")
            log.append("  Attempting without AC-3...")
            csp_result = solver.solve(use_ac3=False)

        if not csp_result.success:
            log.append(f"  ✗ CSP failed completely: {csp_result.message}")
            result.metrics.tasks_total = len(self.pipeline.tasks)
            return result

        result.schedule = csp_result.schedule
        log.append(f"  ✓ Schedule found: {csp_result.message}")
        log.append(f"  Makespan: {csp_result.schedule.get_makespan():.0f}m")

        # Step 3: Simulate
        log.append(f"\n[STEP 3] Simulating execution (mode={mode})...")
        simulator = Simulator(
            self.pipeline,
            mode=mode,
            failure_rate=failure_rate,
            seed=seed
        )
        sim_result = simulator.run(csp_result.schedule)
        result.sim_result = sim_result

        log.append(f"  Completed: {len(sim_result.completed_tasks)} tasks")
        if sim_result.failed_tasks:
            log.append(f"  ✗ Failed: {sim_result.failed_tasks}")
        if sim_result.sla_violations:
            log.append(f"  ⚠ SLA violations: {sim_result.sla_violations}")

        # Step 4: Re-plan if failures occurred
        final_schedule = csp_result.schedule
        if sim_result.failed_tasks:
            log.append(f"\n[STEP 4] Re-planning after failures...")
            planner = SearchPlanner(
                self.pipeline,
                current_time=sim_result.total_time
            )
            search_result = planner.replan_from_failure(
                failed_task_ids=sim_result.failed_tasks,
                completed_task_ids=sim_result.completed_tasks,
                current_time=sim_result.total_time
            )
            result.replan_results.append(search_result)
            result.metrics.replan_count += 1
            result.metrics.search_nodes_explored += search_result.nodes_explored

            if search_result.success:
                log.append(f"  ✓ Recovery plan found: {search_result.message}")
                for action in search_result.actions:
                    log.append(f"    → {action.description}")
                if search_result.schedule:
                    final_schedule = search_result.schedule
            else:
                log.append(f"  ✗ No recovery plan: {search_result.message}")
        else:
            log.append("\n[STEP 4] No failures — skipping re-plan.")

        # Step 5: Learning
        log.append("\n[STEP 5] Recording runtimes for learning...")
        for task_id, actual in sim_result.actual_durations.items():
            record = self.estimator.record_runtime(
                task_id, actual, self._iteration
            )
            old_est = record.estimated_duration
            new_est = self.estimator.get_estimate(task_id)
            log.append(f"  {task_id}: actual={actual:.1f}m, "
                       f"old_est={old_est:.1f}m → new_est={new_est:.1f}m")

        # Compute metrics
        metrics = result.metrics
        metrics.tasks_total = len(self.pipeline.tasks)
        metrics.tasks_completed = len(sim_result.completed_tasks)
        metrics.tasks_failed = len(sim_result.failed_tasks)
        metrics.makespan = final_schedule.get_makespan() if final_schedule else 0
        metrics.sla_adherence = final_schedule.get_sla_adherence_rate(
            self.pipeline.tasks
        ) if final_schedule else 0
        metrics.learning_mae = self.estimator.get_global_mae()

        log.append(f"\n{metrics.summary()}")
        log.append(f"{'='*60}\n")

        return result

    # ================================================================
    # Multi-Iteration Run
    # ================================================================

    def run_iterations(self, n: int, mode: str = "stochastic",
                        failure_rate: float = 0.1,
                        base_seed: int = 42) -> list[AgentRunResult]:
        """
        Run the agent for n iterations with learning across runs.

        Args:
            n: Number of iterations.
            mode: Simulation mode.
            failure_rate: For failure mode.
            base_seed: Base random seed (incremented per iteration).

        Returns:
            List of AgentRunResult, one per iteration.
        """
        results = []
        for i in range(n):
            result = self.run(
                mode=mode,
                failure_rate=failure_rate,
                seed=base_seed + i
            )
            results.append(result)
        return results

    # ================================================================
    # Naive Baseline (for comparison)
    # ================================================================

    def run_naive_baseline(self, mode: str = "normal",
                            failure_rate: float = 0.1,
                            seed: Optional[int] = None) -> AgentRunResult:
        """
        Run a naive baseline scheduler for comparison:
        - Topological sort → FIFO assignment to first available resource.
        - No KB reasoning, no CSP, no re-planning, no learning.
        """
        result = AgentRunResult(iteration=0)
        log = result.log

        log.append("NAIVE BASELINE SCHEDULER")
        log.append("=" * 40)

        self.pipeline.reset_all_statuses()

        # Simple topological order scheduling
        topo_order = self.pipeline.get_topological_order()
        schedule = Schedule()
        resource_end_times = {r.resource_id: 0.0 for r in self.pipeline.resources}
        task_end_times = {}

        for task_id in topo_order:
            task = self.pipeline.tasks[task_id]

            # Earliest start: after all dependencies finish
            deps = self.pipeline.get_dependencies(task_id)
            earliest_start = max(
                (task_end_times.get(d, 0.0) for d in deps),
                default=0.0
            )

            # Find first available resource that fits
            best_resource = None
            best_start = float('inf')
            for resource in self.pipeline.resources:
                if (task.cpu_required <= resource.cpu_capacity and
                        task.memory_required <= resource.memory_capacity):
                    start = max(earliest_start, resource_end_times[resource.resource_id])
                    if start < best_start:
                        best_start = start
                        best_resource = resource

            if best_resource is None:
                log.append(f"  Cannot schedule {task_id} — no fitting resource")
                continue

            end_time = best_start + task.duration_estimate
            schedule.add_entry(ScheduleEntry(
                task_id, best_resource.resource_id, best_start, end_time
            ))
            resource_end_times[best_resource.resource_id] = end_time
            task_end_times[task_id] = end_time

        result.schedule = schedule
        log.append(f"  Scheduled {len(schedule.entries)} tasks")
        log.append(f"  Makespan: {schedule.get_makespan():.0f}m")

        # Simulate
        simulator = Simulator(self.pipeline, mode=mode,
                              failure_rate=failure_rate, seed=seed)
        sim_result = simulator.run(schedule)
        result.sim_result = sim_result

        # Metrics (no re-planning for baseline)
        metrics = result.metrics
        metrics.tasks_total = len(self.pipeline.tasks)
        metrics.tasks_completed = len(sim_result.completed_tasks)
        metrics.tasks_failed = len(sim_result.failed_tasks)
        metrics.makespan = schedule.get_makespan()
        metrics.sla_adherence = schedule.get_sla_adherence_rate(self.pipeline.tasks)

        log.append(f"\n{metrics.summary()}")
        return result

    # ================================================================
    # Getters
    # ================================================================

    def get_estimator(self) -> RuntimeEstimator:
        return self.estimator

    def get_kb(self) -> KnowledgeBase:
        return self.kb

    def get_iteration(self) -> int:
        return self._iteration
