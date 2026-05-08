"""
test_csp_solver.py — Unit tests for the CSP Solver module.

Tests cover: domain generation, dependency constraints, resource capacity,
SLA constraints, MRV variable ordering, forward checking, AC-3,
infeasibility detection, schedule validation, and solving all sample DAGs.
"""

import os
import pytest
from src.task_dag import Task, Resource, Pipeline, Schedule, ScheduleEntry
from src.csp_solver import CSPSolver, CSPResult


# ============================================================
# Helper: Build test pipelines
# ============================================================

def make_linear_pipeline():
    """A → B → C, 1 resource."""
    p = Pipeline(name="linear")
    p.add_task(Task("A", "A", 30, cpu_required=2, memory_required=4.0))
    p.add_task(Task("B", "B", 45, cpu_required=4, memory_required=8.0))
    p.add_task(Task("C", "C", 15, cpu_required=2, memory_required=4.0, sla_deadline=360))
    p.add_dependency("A", "B")
    p.add_dependency("B", "C")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8, memory_capacity=32.0))
    return p


def make_parallel_pipeline():
    """A and B can run in parallel, both feed into C."""
    p = Pipeline(name="parallel")
    p.add_task(Task("A", "A", 30, cpu_required=4))
    p.add_task(Task("B", "B", 30, cpu_required=4))
    p.add_task(Task("C", "C", 15, cpu_required=2, sla_deadline=360))
    p.add_dependency("A", "C")
    p.add_dependency("B", "C")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8, memory_capacity=32.0))
    return p


def make_resource_contention_pipeline():
    """Two tasks that each need 6 CPUs on an 8-CPU server — can't run together."""
    p = Pipeline(name="contention")
    p.add_task(Task("heavy1", "Heavy 1", 30, cpu_required=6))
    p.add_task(Task("heavy2", "Heavy 2", 30, cpu_required=6))
    p.add_task(Task("light", "Light", 10, cpu_required=2, sla_deadline=360))
    p.add_dependency("heavy1", "light")
    p.add_dependency("heavy2", "light")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8, memory_capacity=32.0))
    return p


def make_infeasible_pipeline():
    """Task needs 16 CPUs but max resource has 8 — impossible."""
    p = Pipeline(name="infeasible")
    p.add_task(Task("huge", "Huge Task", 30, cpu_required=16))
    p.add_resource(Resource("r1", "Server", cpu_capacity=8))
    return p


def make_tight_sla_pipeline():
    """SLA is too tight to fit the critical path."""
    p = Pipeline(name="tight_sla")
    p.add_task(Task("A", "A", 200))
    p.add_task(Task("B", "B", 200, sla_deadline=300))
    p.add_dependency("A", "B")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8))
    return p


def make_multi_resource_pipeline():
    """Multiple resources available — solver should distribute tasks."""
    p = Pipeline(name="multi_resource")
    p.add_task(Task("A", "A", 30, cpu_required=4))
    p.add_task(Task("B", "B", 30, cpu_required=4))
    p.add_task(Task("C", "C", 30, cpu_required=4))
    p.add_task(Task("D", "D", 15, cpu_required=2, sla_deadline=360))
    p.add_dependency("A", "D")
    p.add_dependency("B", "D")
    p.add_dependency("C", "D")
    p.add_resource(Resource("r1", "Server A", cpu_capacity=4, memory_capacity=16.0))
    p.add_resource(Resource("r2", "Server B", cpu_capacity=4, memory_capacity=16.0))
    return p


# ============================================================
# Basic Solving Tests
# ============================================================

class TestCSPSolverBasic:
    def test_solve_linear(self):
        result = CSPSolver(make_linear_pipeline()).solve()
        assert result.success is True
        assert len(result.assignments) == 3
        assert result.schedule is not None

    def test_solve_parallel(self):
        result = CSPSolver(make_parallel_pipeline()).solve()
        assert result.success is True
        assert len(result.assignments) == 3

    def test_solve_multi_resource(self):
        result = CSPSolver(make_multi_resource_pipeline()).solve()
        assert result.success is True
        assert len(result.assignments) == 4

    def test_infeasible_resource(self):
        result = CSPSolver(make_infeasible_pipeline()).solve()
        assert result.success is False
        assert "no valid" in result.message.lower()

    def test_infeasible_sla(self):
        """Critical path 400m but SLA at 300m — should fail."""
        result = CSPSolver(make_tight_sla_pipeline(), time_horizon=480).solve()
        assert result.success is False

    def test_nodes_explored_positive(self):
        result = CSPSolver(make_linear_pipeline()).solve()
        assert result.nodes_explored > 0

    def test_message_on_success(self):
        result = CSPSolver(make_linear_pipeline()).solve()
        assert "Solution found" in result.message


# ============================================================
# Constraint Validation Tests
# ============================================================

class TestConstraintValidation:
    def test_dependency_respected(self):
        """All dependency orderings must be respected in the solution."""
        p = make_linear_pipeline()
        result = CSPSolver(p).solve()
        assert result.success

        a = result.schedule.get_entry_for_task("A")
        b = result.schedule.get_entry_for_task("B")
        c = result.schedule.get_entry_for_task("C")

        assert b.start_time >= a.end_time, "B must start after A ends"
        assert c.start_time >= b.end_time, "C must start after B ends"

    def test_sla_respected(self):
        """SLA-bound tasks must finish before their deadline."""
        p = make_linear_pipeline()
        result = CSPSolver(p).solve()
        assert result.success

        c = result.schedule.get_entry_for_task("C")
        task_c = p.tasks["C"]
        assert c.end_time <= task_c.sla_deadline

    def test_resource_capacity_respected(self):
        """Heavy tasks that exceed combined capacity must not overlap."""
        p = make_resource_contention_pipeline()
        result = CSPSolver(p).solve()
        assert result.success

        h1 = result.schedule.get_entry_for_task("heavy1")
        h2 = result.schedule.get_entry_for_task("heavy2")

        # If on same resource, they must NOT overlap (6+6=12 > 8)
        if h1.resource_id == h2.resource_id:
            overlaps = h1.overlaps_with(h2)
            assert not overlaps, "Heavy tasks overlap on same resource!"

    def test_parallel_tasks_can_overlap_if_capacity_allows(self):
        """Two 4-CPU tasks on an 8-CPU server CAN run simultaneously."""
        p = make_parallel_pipeline()
        result = CSPSolver(p).solve()
        assert result.success
        # A and B could overlap since 4+4=8 = capacity — that's fine

    def test_validate_schedule_clean(self):
        """Validate that a solved schedule has zero violations."""
        p = make_linear_pipeline()
        result = CSPSolver(p).solve()
        assert result.success

        violations = CSPSolver.validate_schedule(result.schedule, p)
        assert len(violations) == 0, f"Violations found: {violations}"

    def test_validate_schedule_with_violation(self):
        """Manually build a bad schedule and verify validation catches it."""
        p = make_linear_pipeline()
        bad_schedule = Schedule()
        # B starts at 0 but depends on A (which also starts at 0)
        bad_schedule.add_entry(ScheduleEntry("A", "r1", 0, 30))
        bad_schedule.add_entry(ScheduleEntry("B", "r1", 0, 45))  # violation!
        bad_schedule.add_entry(ScheduleEntry("C", "r1", 45, 60))

        violations = CSPSolver.validate_schedule(bad_schedule, p)
        assert len(violations) > 0
        assert any("Dependency violation" in v for v in violations)


# ============================================================
# Forward Checking vs Plain Backtracking
# ============================================================

class TestForwardChecking:
    def test_fc_solves_same_as_plain(self):
        """Both methods should find a valid solution."""
        p = make_linear_pipeline()
        result_fc = CSPSolver(p).solve(use_ac3=False)
        result_plain = CSPSolver(p).solve_without_fc(use_ac3=False)
        assert result_fc.success is True
        assert result_plain.success is True

    def test_fc_explores_fewer_nodes(self):
        """Forward checking should explore fewer nodes than plain backtracking."""
        p = make_resource_contention_pipeline()
        solver_fc = CSPSolver(p)
        result_fc = solver_fc.solve(use_ac3=False)

        solver_plain = CSPSolver(p)
        result_plain = solver_plain.solve_without_fc(use_ac3=False)

        assert result_fc.success
        assert result_plain.success
        # FC should be at least as efficient (usually more)
        assert result_fc.nodes_explored <= result_plain.nodes_explored


# ============================================================
# AC-3 Tests
# ============================================================

class TestAC3:
    def test_ac3_preserves_solvability(self):
        """AC-3 should not make a solvable problem unsolvable."""
        p = make_linear_pipeline()
        result = CSPSolver(p).solve(use_ac3=True)
        assert result.success is True

    def test_ac3_detects_infeasibility(self):
        """AC-3 should detect when no solution exists."""
        p = make_infeasible_pipeline()
        result = CSPSolver(p).solve(use_ac3=True)
        assert result.success is False

    def test_ac3_vs_no_ac3(self):
        """Both should produce valid solutions on solvable problems."""
        p = make_parallel_pipeline()
        result_ac3 = CSPSolver(p).solve(use_ac3=True)
        result_no = CSPSolver(p).solve(use_ac3=False)
        assert result_ac3.success is True
        assert result_no.success is True


# ============================================================
# Sample DAG Tests
# ============================================================

class TestSampleDAGs:
    def _solve_dag(self, filename):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            pytest.skip(f"{filename} not found")
        p = Pipeline.load_json(filepath)
        return CSPSolver(p, time_horizon=600, time_step=15).solve(), p

    def test_solve_simple_dag(self):
        result, p = self._solve_dag("simple_dag.json")
        assert result.success, result.message
        violations = CSPSolver.validate_schedule(result.schedule, p)
        assert len(violations) == 0, f"Violations: {violations}"

    def test_solve_medium_dag(self):
        result, p = self._solve_dag("medium_dag.json")
        assert result.success, result.message
        violations = CSPSolver.validate_schedule(result.schedule, p)
        assert len(violations) == 0, f"Violations: {violations}"

    def test_solve_complex_dag(self):
        result, p = self._solve_dag("complex_dag.json")
        assert result.success, result.message
        violations = CSPSolver.validate_schedule(result.schedule, p)
        assert len(violations) == 0, f"Violations: {violations}"

    def test_complex_dag_sla_compliance(self):
        result, p = self._solve_dag("complex_dag.json")
        assert result.success
        sla_rate = result.schedule.get_sla_adherence_rate(p.tasks)
        assert sla_rate == 100.0, f"SLA adherence only {sla_rate}%"


# ============================================================
# Time Step Granularity
# ============================================================

class TestTimeGranularity:
    def test_finer_granularity_still_solves(self):
        p = make_linear_pipeline()
        result = CSPSolver(p, time_step=5).solve()
        assert result.success

    def test_coarser_granularity_still_solves(self):
        p = make_linear_pipeline()
        result = CSPSolver(p, time_step=30).solve()
        assert result.success
