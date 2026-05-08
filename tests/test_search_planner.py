"""
test_search_planner.py — Unit tests for the Search & Planning module.

Tests cover: initial state creation, goal detection, action generation,
A* search, greedy search, failure recovery, SLA-aware planning,
cascade skipping, and A* vs greedy comparison.
"""

import pytest
from src.task_dag import Task, TaskStatus, Resource, Pipeline
from src.search_planner import SearchPlanner, PlannerState, SearchResult


# ============================================================
# Helper pipelines
# ============================================================

def make_linear():
    """A → B → C."""
    p = Pipeline(name="linear")
    p.add_task(Task("A", "A", 30, cpu_required=2))
    p.add_task(Task("B", "B", 20, cpu_required=2))
    p.add_task(Task("C", "C", 10, cpu_required=2, sla_deadline=360))
    p.add_dependency("A", "B")
    p.add_dependency("B", "C")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8))
    return p


def make_diamond():
    """A → B, A → C, B → D, C → D."""
    p = Pipeline(name="diamond")
    p.add_task(Task("A", "A", 15))
    p.add_task(Task("B", "B", 20))
    p.add_task(Task("C", "C", 25))
    p.add_task(Task("D", "D", 10, sla_deadline=360))
    p.add_dependency("A", "B")
    p.add_dependency("A", "C")
    p.add_dependency("B", "D")
    p.add_dependency("C", "D")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8))
    return p


def make_multi_branch():
    """A → B → D, A → C → D, plus independent E."""
    p = Pipeline(name="multi")
    p.add_task(Task("A", "A", 10))
    p.add_task(Task("B", "B", 20))
    p.add_task(Task("C", "C", 15))
    p.add_task(Task("D", "D", 10, sla_deadline=300))
    p.add_task(Task("E", "E", 25))  # independent
    p.add_dependency("A", "B")
    p.add_dependency("A", "C")
    p.add_dependency("B", "D")
    p.add_dependency("C", "D")
    p.add_resource(Resource("r1", "Server", cpu_capacity=8))
    return p


# ============================================================
# State & Goal Tests
# ============================================================

class TestStateAndGoal:
    def test_create_initial_state(self):
        p = make_linear()
        planner = SearchPlanner(p, current_time=0)
        state = planner.create_initial_state()
        assert state.task_statuses["A"] == "pending"
        assert state.task_statuses["B"] == "pending"
        assert state.current_time == 0

    def test_create_state_with_overrides(self):
        p = make_linear()
        planner = SearchPlanner(p, current_time=60)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "failed", "C": "pending"}
        )
        assert state.task_statuses["A"] == "completed"
        assert state.task_statuses["B"] == "failed"

    def test_goal_all_completed(self):
        p = make_linear()
        planner = SearchPlanner(p)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "completed", "C": "completed"}
        )
        assert planner._is_goal(state) is True

    def test_goal_with_skipped(self):
        p = make_linear()
        planner = SearchPlanner(p)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "skipped", "C": "skipped"}
        )
        assert planner._is_goal(state) is True

    def test_not_goal_with_pending(self):
        p = make_linear()
        planner = SearchPlanner(p)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "pending", "C": "pending"}
        )
        assert planner._is_goal(state) is False


# ============================================================
# A* Search Tests
# ============================================================

class TestAStarSearch:
    def test_astar_all_pending(self):
        """A* should schedule all tasks from scratch."""
        p = make_linear()
        planner = SearchPlanner(p, current_time=0)
        result = planner.search_astar()
        assert result.success is True
        assert result.nodes_explored > 0
        assert result.schedule is not None
        assert len(result.actions) > 0

    def test_astar_failure_recovery(self):
        """A* should find a plan when a task has failed."""
        p = make_linear()
        planner = SearchPlanner(p, current_time=30)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "failed", "C": "pending"}
        )
        result = planner.search_astar(state)
        assert result.success is True
        # Plan should include retrying B or skipping it
        action_descs = " ".join(a.description for a in result.actions)
        assert "B" in action_descs

    def test_astar_respects_dependencies(self):
        """Tasks should only be scheduled after dependencies complete."""
        p = make_linear()
        planner = SearchPlanner(p, current_time=0)
        result = planner.search_astar()
        assert result.success

        # Verify order: A before B, B before C in schedule
        entries = {e.task_id: e for e in result.schedule.entries}
        if "A" in entries and "B" in entries:
            assert entries["B"].start_time >= entries["A"].end_time
        if "B" in entries and "C" in entries:
            assert entries["C"].start_time >= entries["B"].end_time

    def test_astar_diamond_dag(self):
        """A* should handle diamond DAG correctly."""
        p = make_diamond()
        planner = SearchPlanner(p, current_time=0)
        result = planner.search_astar()
        assert result.success is True
        # All 4 tasks should be scheduled
        scheduled_tasks = {e.task_id for e in result.schedule.entries}
        assert scheduled_tasks == {"A", "B", "C", "D"}

    def test_astar_returns_nodes_explored(self):
        p = make_linear()
        planner = SearchPlanner(p)
        result = planner.search_astar()
        assert result.nodes_explored > 0

    def test_astar_sla_adherence(self):
        """A* should report SLA adherence percentage."""
        p = make_linear()
        planner = SearchPlanner(p, current_time=0)
        result = planner.search_astar()
        assert result.success
        # C has SLA at 360, total work is 60 — should be met
        assert result.sla_adherence == 100.0


# ============================================================
# Greedy Best-First Search Tests
# ============================================================

class TestGreedySearch:
    def test_greedy_all_pending(self):
        p = make_linear()
        planner = SearchPlanner(p, current_time=0)
        result = planner.search_greedy()
        assert result.success is True

    def test_greedy_failure_recovery(self):
        p = make_linear()
        planner = SearchPlanner(p, current_time=30)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "failed", "C": "pending"}
        )
        result = planner.search_greedy(state)
        assert result.success is True

    def test_greedy_diamond(self):
        p = make_diamond()
        planner = SearchPlanner(p, current_time=0)
        result = planner.search_greedy()
        assert result.success is True


# ============================================================
# A* vs Greedy Comparison
# ============================================================

class TestAStarVsGreedy:
    def test_both_find_solution(self):
        """Both should find a valid solution for the same problem."""
        p = make_diamond()
        planner = SearchPlanner(p, current_time=0)
        astar = planner.search_astar()
        greedy = planner.search_greedy()
        assert astar.success is True
        assert greedy.success is True

    def test_astar_cost_leq_greedy(self):
        """A* should find a plan with cost ≤ greedy (optimal vs heuristic)."""
        p = make_diamond()
        planner_a = SearchPlanner(p, current_time=0)
        planner_g = SearchPlanner(p, current_time=0)
        astar = planner_a.search_astar()
        greedy = planner_g.search_greedy()
        assert astar.success and greedy.success
        assert astar.total_cost <= greedy.total_cost + 0.001  # tiny epsilon


# ============================================================
# Failure Recovery (replan_from_failure) Tests
# ============================================================

class TestReplanFromFailure:
    def test_replan_single_failure(self):
        """Replan after one task fails."""
        p = make_linear()
        planner = SearchPlanner(p)
        result = planner.replan_from_failure(
            failed_task_ids=["B"],
            completed_task_ids=["A"],
            current_time=30
        )
        assert result.success is True
        # B should be retried (or skipped + C skipped)
        scheduled = {e.task_id for e in result.schedule.entries}
        # Either B was retried and C scheduled, or both skipped
        assert "A" in scheduled or len(scheduled) >= 1

    def test_replan_multiple_failures(self):
        """Replan after multiple tasks fail."""
        p = make_diamond()
        planner = SearchPlanner(p)
        result = planner.replan_from_failure(
            failed_task_ids=["B", "C"],
            completed_task_ids=["A"],
            current_time=15
        )
        assert result.success is True

    def test_replan_preserves_completed(self):
        """Completed tasks should stay completed in the plan."""
        p = make_linear()
        planner = SearchPlanner(p)
        result = planner.replan_from_failure(
            failed_task_ids=["B"],
            completed_task_ids=["A"],
            current_time=30
        )
        assert result.success
        assert result.final_state.task_statuses["A"] == "completed"

    def test_replan_no_failures(self):
        """If nothing failed, should just schedule remaining tasks."""
        p = make_linear()
        planner = SearchPlanner(p)
        result = planner.replan_from_failure(
            failed_task_ids=[],
            completed_task_ids=["A"],
            current_time=30
        )
        assert result.success is True


# ============================================================
# Cascade Skip Tests
# ============================================================

class TestCascadeSkip:
    def test_skip_cascades_to_dependents(self):
        """If A is skipped, B and C (which depend on A) should also be skipped."""
        p = make_linear()
        planner = SearchPlanner(p, current_time=0)
        state = planner.create_initial_state(
            task_statuses={"A": "failed", "B": "pending", "C": "pending"}
        )
        # Search should skip A, which cascades to B, then C
        result = planner.search_astar(state)
        assert result.success is True

    def test_partial_cascade_diamond(self):
        """In diamond: if B fails and is skipped, D should still wait for C."""
        p = make_diamond()
        planner = SearchPlanner(p, current_time=15)
        state = planner.create_initial_state(
            task_statuses={
                "A": "completed", "B": "failed",
                "C": "pending", "D": "pending"
            }
        )
        result = planner.search_astar(state)
        assert result.success is True


# ============================================================
# Edge Cases
# ============================================================

class TestEdgeCases:
    def test_single_task_pipeline(self):
        p = Pipeline(name="single")
        p.add_task(Task("only", "Only Task", 10))
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))
        planner = SearchPlanner(p)
        result = planner.search_astar()
        assert result.success is True

    def test_already_complete(self):
        """All tasks already done — should immediately be at goal."""
        p = make_linear()
        planner = SearchPlanner(p)
        state = planner.create_initial_state(
            task_statuses={"A": "completed", "B": "completed", "C": "completed"}
        )
        result = planner.search_astar(state)
        assert result.success is True
        assert result.nodes_explored <= 2  # should find goal immediately

    def test_independent_tasks(self):
        """Tasks with no dependencies can all be scheduled freely."""
        p = Pipeline(name="independent")
        p.add_task(Task("X", "X", 10))
        p.add_task(Task("Y", "Y", 15))
        p.add_task(Task("Z", "Z", 20))
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))
        planner = SearchPlanner(p)
        result = planner.search_astar()
        assert result.success is True
        scheduled = {e.task_id for e in result.schedule.entries}
        assert scheduled == {"X", "Y", "Z"}
