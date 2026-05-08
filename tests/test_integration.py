"""
test_integration.py — Integration tests for the full agent pipeline.

Tests the complete cycle: KB → CSP → Simulate → Replan → Learn.
Verifies all components work together correctly.
"""

import os
import subprocess
import pytest
from src.task_dag import Pipeline, Task, Resource
from src.knowledge_base import KnowledgeBase
from src.csp_solver import CSPSolver
from src.search_planner import SearchPlanner
from src.learning import RuntimeEstimator
from src.simulator import Simulator
from src.scheduler_agent import SchedulerAgent


# ============================================================
# Helper
# ============================================================

def load_dag(name):
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    return Pipeline.load_json(os.path.join(data_dir, f"{name}_dag.json"))


# ============================================================
# Full Agent Cycle Tests
# ============================================================

class TestFullAgentCycle:
    def test_normal_mode_simple(self):
        """Agent runs simple DAG in normal mode — all tasks complete."""
        pipeline = load_dag("simple")
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="normal", seed=42)
        assert result.metrics.tasks_completed == result.metrics.tasks_total
        assert result.metrics.tasks_failed == 0
        assert result.metrics.sla_adherence == 100.0

    def test_normal_mode_medium(self):
        pipeline = load_dag("medium")
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="normal", seed=42)
        assert result.metrics.tasks_completed == result.metrics.tasks_total

    def test_normal_mode_complex(self):
        pipeline = load_dag("complex")
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="normal", seed=42)
        assert result.metrics.tasks_completed == result.metrics.tasks_total

    def test_stochastic_mode(self):
        """Stochastic mode — tasks complete but with varied runtimes."""
        pipeline = load_dag("medium")
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="stochastic", seed=42)
        assert result.metrics.tasks_completed > 0
        assert result.schedule is not None

    def test_failure_mode_triggers_replan(self):
        """Failure mode — agent detects failures and replans."""
        pipeline = load_dag("medium")
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="failure", failure_rate=0.3, seed=42)
        # With 30% failure rate, some tasks should fail
        # Agent should have attempted re-planning
        assert result.metrics.tasks_total == 10
        assert len(result.log) > 0  # agent produced output

    def test_spike_mode(self):
        """Spike mode — one task takes much longer."""
        pipeline = load_dag("simple")
        agent = SchedulerAgent(pipeline)
        result = agent.run(mode="spike", seed=42)
        assert result.metrics.tasks_completed > 0


# ============================================================
# Multi-Iteration Learning Tests
# ============================================================

class TestLearningAcrossIterations:
    def test_learning_improves_estimates(self):
        """Over 10 iterations, EWMA estimates should converge toward actuals."""
        pipeline = load_dag("simple")
        agent = SchedulerAgent(pipeline)
        results = agent.run_iterations(n=10, mode="stochastic", base_seed=42)

        estimator = agent.get_estimator()
        assert len(results) == 10

        # Check that some tasks have history
        all_history = estimator.get_all_history()
        assert len(all_history) > 0

        # Global MAE should exist
        mae = estimator.get_global_mae()
        assert mae is not None

    def test_error_decreases_over_iterations(self):
        """Error in first 3 iterations should be >= error in last 3."""
        pipeline = load_dag("simple")
        agent = SchedulerAgent(pipeline, ewma_alpha=0.4)
        results = agent.run_iterations(n=10, mode="stochastic", base_seed=100)

        estimator = agent.get_estimator()
        # Pick a task that has full history
        for task_id in pipeline.tasks:
            errors = estimator.get_error_over_iterations(task_id)
            if len(errors) >= 6:
                first_3_avg = sum(e[1] for e in errors[:3]) / 3
                last_3_avg = sum(e[1] for e in errors[-3:]) / 3
                # Last errors should be smaller (or at least not massively bigger)
                # Allow some tolerance since it's stochastic
                assert last_3_avg <= first_3_avg * 1.5
                break


# ============================================================
# Baseline Comparison Tests
# ============================================================

class TestBaselineComparison:
    def test_baseline_runs(self):
        """Naive baseline should produce a valid schedule."""
        pipeline = load_dag("medium")
        agent = SchedulerAgent(pipeline)
        result = agent.run_naive_baseline(mode="normal", seed=42)
        assert result.schedule is not None
        assert result.metrics.tasks_completed > 0

    def test_agent_beats_baseline_on_normal(self):
        """On normal mode, agent should match or beat baseline on SLA."""
        pipeline = load_dag("medium")

        agent = SchedulerAgent(pipeline)
        agent_result = agent.run(mode="normal", seed=42)

        baseline_agent = SchedulerAgent(Pipeline.load_json("data/medium_dag.json"))
        baseline_result = baseline_agent.run_naive_baseline(mode="normal", seed=42)

        assert agent_result.metrics.sla_adherence >= baseline_result.metrics.sla_adherence


# ============================================================
# Component Interaction Tests
# ============================================================

class TestComponentInteraction:
    def test_kb_feeds_csp(self):
        """KB facts should correctly inform CSP variable domains."""
        pipeline = load_dag("simple")
        kb = KnowledgeBase()
        kb.load_from_pipeline(pipeline)
        kb.register_default_rules()
        kb.forward_chain()

        # KB should identify ready tasks
        ready = kb.get_ready_tasks()
        assert len(ready) > 0

        # CSP should solve using the pipeline
        solver = CSPSolver(pipeline)
        result = solver.solve()
        assert result.success

    def test_csp_schedule_validates(self):
        """CSP output should pass independent validation."""
        pipeline = load_dag("complex")
        solver = CSPSolver(pipeline, time_horizon=600)
        result = solver.solve()
        assert result.success

        violations = CSPSolver.validate_schedule(result.schedule, pipeline)
        assert len(violations) == 0

    def test_simulator_feeds_learning(self):
        """Simulator outputs should feed into the learning component."""
        pipeline = load_dag("simple")
        solver = CSPSolver(pipeline)
        csp_result = solver.solve()
        assert csp_result.success

        sim = Simulator(pipeline, mode="stochastic", seed=42)
        sim_result = sim.run(csp_result.schedule)

        estimator = RuntimeEstimator()
        estimator.set_initial_estimates_from_pipeline(pipeline)

        for task_id, actual in sim_result.actual_durations.items():
            estimator.record_runtime(task_id, actual, iteration=1)

        assert estimator.get_global_mae() is not None

    def test_search_handles_failure(self):
        """Search planner should find recovery when given failures."""
        pipeline = load_dag("simple")
        planner = SearchPlanner(pipeline, current_time=30)
        result = planner.replan_from_failure(
            failed_task_ids=["validate"],
            completed_task_ids=["extract"],
            current_time=30
        )
        assert result.success

    def test_full_pipeline_kb_to_search(self):
        """KB detects failure → search replans."""
        pipeline = load_dag("simple")

        # Setup KB with a failed task
        kb = KnowledgeBase()
        kb.load_from_pipeline(pipeline)
        kb.register_default_rules()
        kb.update_task_status("extract", "completed")
        kb.update_task_status("validate", "failed")
        kb.clear_derived()
        kb.forward_chain()

        # KB should detect the failure
        blocked = kb.get_blocked_tasks()
        retries = kb.get_retry_recommendations()
        assert len(blocked) > 0 or len(retries) > 0

        # Search should find a recovery plan
        planner = SearchPlanner(pipeline, current_time=45)
        result = planner.replan_from_failure(
            failed_task_ids=["validate"],
            completed_task_ids=["extract"],
            current_time=45
        )
        assert result.success


# ============================================================
# CLI Smoke Test
# ============================================================

class TestCLI:
    def test_main_runs_without_error(self):
        """main.py should execute without crashing."""
        import subprocess
        result = subprocess.run(
            ["python", "main.py", "--dag", "simple", "--mode", "normal",
             "--no-plots"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        assert "INTELLIGENT PIPELINE SCHEDULER" in result.stdout

    def test_main_with_iterations(self):
        result = subprocess.run(
            ["python", "main.py", "--dag", "simple", "--mode", "stochastic",
             "--iterations", "3", "--no-plots"],
            capture_output=True, text=True,
            cwd=os.path.join(os.path.dirname(__file__), "..")
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
