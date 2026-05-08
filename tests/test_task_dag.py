"""
test_task_dag.py — Unit tests for the core data model.

Tests cover: Task creation, Resource creation, Schedule operations,
Pipeline DAG operations (dependencies, topological sort, critical path,
ready tasks, validation, JSON serialization).
"""

import os
import json
import tempfile
import pytest
from src.task_dag import (
    Task, TaskStatus, Resource, ScheduleEntry, Schedule, Pipeline
)


# ============================================================
# Task Tests
# ============================================================

class TestTask:
    def test_create_task_defaults(self):
        t = Task(task_id="t1", name="Test Task", duration_estimate=30)
        assert t.task_id == "t1"
        assert t.cpu_required == 2
        assert t.memory_required == 4.0
        assert t.priority == 5
        assert t.sla_deadline is None
        assert t.status == TaskStatus.PENDING

    def test_create_task_custom(self):
        t = Task(
            task_id="t2", name="Heavy Task", duration_estimate=60,
            cpu_required=8, memory_required=32.0, priority=10,
            sla_deadline=360.0
        )
        assert t.cpu_required == 8
        assert t.sla_deadline == 360.0
        assert t.priority == 10

    def test_task_equality(self):
        t1 = Task(task_id="t1", name="A", duration_estimate=10)
        t2 = Task(task_id="t1", name="B", duration_estimate=20)
        t3 = Task(task_id="t3", name="A", duration_estimate=10)
        assert t1 == t2      # same task_id
        assert t1 != t3      # different task_id

    def test_task_hashable(self):
        t1 = Task(task_id="t1", name="A", duration_estimate=10)
        t2 = Task(task_id="t2", name="B", duration_estimate=20)
        s = {t1, t2}
        assert len(s) == 2

    def test_task_to_dict(self):
        t = Task(task_id="t1", name="Test", duration_estimate=30, sla_deadline=360)
        d = t.to_dict()
        assert d["task_id"] == "t1"
        assert d["sla_deadline"] == 360
        assert d["status"] == "pending"


# ============================================================
# Resource Tests
# ============================================================

class TestResource:
    def test_create_resource_defaults(self):
        r = Resource(resource_id="r1", name="Server 1")
        assert r.cpu_capacity == 8
        assert r.memory_capacity == 32.0

    def test_resource_equality(self):
        r1 = Resource(resource_id="r1", name="A")
        r2 = Resource(resource_id="r1", name="B")
        assert r1 == r2

    def test_resource_hashable(self):
        r1 = Resource(resource_id="r1", name="A")
        r2 = Resource(resource_id="r2", name="B")
        s = {r1, r2}
        assert len(s) == 2


# ============================================================
# ScheduleEntry Tests
# ============================================================

class TestScheduleEntry:
    def test_duration(self):
        e = ScheduleEntry(task_id="t1", resource_id="r1", start_time=60, end_time=90)
        assert e.duration == 30

    def test_overlaps(self):
        e1 = ScheduleEntry(task_id="t1", resource_id="r1", start_time=60, end_time=90)
        e2 = ScheduleEntry(task_id="t2", resource_id="r1", start_time=80, end_time=110)
        e3 = ScheduleEntry(task_id="t3", resource_id="r1", start_time=90, end_time=120)
        assert e1.overlaps_with(e2) is True     # overlap at 80-90
        assert e1.overlaps_with(e3) is False    # e3 starts exactly when e1 ends

    def test_no_overlap(self):
        e1 = ScheduleEntry(task_id="t1", resource_id="r1", start_time=0, end_time=30)
        e2 = ScheduleEntry(task_id="t2", resource_id="r1", start_time=60, end_time=90)
        assert e1.overlaps_with(e2) is False


# ============================================================
# Schedule Tests
# ============================================================

class TestSchedule:
    def _make_schedule(self):
        s = Schedule()
        s.add_entry(ScheduleEntry("t1", "r1", 0, 30))
        s.add_entry(ScheduleEntry("t2", "r1", 30, 60))
        s.add_entry(ScheduleEntry("t3", "r2", 0, 45))
        return s

    def test_add_and_get(self):
        s = self._make_schedule()
        assert len(s.entries) == 3
        assert s.get_entry_for_task("t1").start_time == 0
        assert s.get_entry_for_task("missing") is None

    def test_entries_for_resource(self):
        s = self._make_schedule()
        r1_entries = s.get_entries_for_resource("r1")
        assert len(r1_entries) == 2
        assert r1_entries[0].task_id == "t1"  # sorted by start_time
        assert r1_entries[1].task_id == "t2"

    def test_makespan(self):
        s = self._make_schedule()
        assert s.get_makespan() == 60  # 0 to 60

    def test_empty_makespan(self):
        s = Schedule()
        assert s.get_makespan() == 0.0

    def test_sla_compliance(self):
        s = Schedule()
        s.add_entry(ScheduleEntry("t1", "r1", 0, 50))   # finishes at 50
        s.add_entry(ScheduleEntry("t2", "r1", 50, 120))  # finishes at 120

        tasks = {
            "t1": Task(task_id="t1", name="A", duration_estimate=50, sla_deadline=60),
            "t2": Task(task_id="t2", name="B", duration_estimate=70, sla_deadline=100),
        }
        compliance = s.check_sla_compliance(tasks)
        assert compliance["t1"] is True    # 50 <= 60
        assert compliance["t2"] is False   # 120 > 100

    def test_sla_adherence_rate(self):
        s = Schedule()
        s.add_entry(ScheduleEntry("t1", "r1", 0, 50))
        s.add_entry(ScheduleEntry("t2", "r1", 50, 120))
        tasks = {
            "t1": Task(task_id="t1", name="A", duration_estimate=50, sla_deadline=60),
            "t2": Task(task_id="t2", name="B", duration_estimate=70, sla_deadline=100),
        }
        assert s.get_sla_adherence_rate(tasks) == 50.0

    def test_sla_adherence_no_sla_tasks(self):
        s = Schedule()
        s.add_entry(ScheduleEntry("t1", "r1", 0, 50))
        tasks = {
            "t1": Task(task_id="t1", name="A", duration_estimate=50),  # no SLA
        }
        assert s.get_sla_adherence_rate(tasks) == 100.0


# ============================================================
# Pipeline Tests
# ============================================================

class TestPipeline:
    def _make_linear_pipeline(self):
        """A → B → C (linear chain)"""
        p = Pipeline(name="linear")
        p.add_task(Task(task_id="A", name="Task A", duration_estimate=10))
        p.add_task(Task(task_id="B", name="Task B", duration_estimate=20))
        p.add_task(Task(task_id="C", name="Task C", duration_estimate=30))
        p.add_dependency("A", "B")
        p.add_dependency("B", "C")
        p.add_resource(Resource(resource_id="r1", name="Server"))
        return p

    def _make_diamond_pipeline(self):
        """A diamond-shaped DAG: A splits to B and C, both merge into D."""
        p = Pipeline(name="diamond")
        p.add_task(Task(task_id="A", name="A", duration_estimate=10))
        p.add_task(Task(task_id="B", name="B", duration_estimate=20))
        p.add_task(Task(task_id="C", name="C", duration_estimate=30))
        p.add_task(Task(task_id="D", name="D", duration_estimate=15))
        p.add_dependency("A", "B")
        p.add_dependency("A", "C")
        p.add_dependency("B", "D")
        p.add_dependency("C", "D")
        p.add_resource(Resource(resource_id="r1", name="Server"))
        return p

    # --- Dependencies ---

    def test_add_dependency(self):
        p = self._make_linear_pipeline()
        assert p.get_dependencies("B") == ["A"]
        assert p.get_dependents("A") == ["B"]

    def test_dependency_missing_task(self):
        p = Pipeline()
        p.add_task(Task(task_id="A", name="A", duration_estimate=10))
        with pytest.raises(ValueError, match="not found"):
            p.add_dependency("A", "missing")

    def test_self_dependency(self):
        p = Pipeline()
        p.add_task(Task(task_id="A", name="A", duration_estimate=10))
        with pytest.raises(ValueError, match="cannot depend on itself"):
            p.add_dependency("A", "A")

    def test_cycle_detection(self):
        p = Pipeline()
        p.add_task(Task(task_id="A", name="A", duration_estimate=10))
        p.add_task(Task(task_id="B", name="B", duration_estimate=10))
        p.add_dependency("A", "B")
        with pytest.raises(ValueError, match="cycle"):
            p.add_dependency("B", "A")

    # --- Topological sort ---

    def test_topological_order_linear(self):
        p = self._make_linear_pipeline()
        order = p.get_topological_order()
        assert order.index("A") < order.index("B") < order.index("C")

    def test_topological_order_diamond(self):
        p = self._make_diamond_pipeline()
        order = p.get_topological_order()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    # --- Critical path ---

    def test_critical_path_linear(self):
        p = self._make_linear_pipeline()
        path, duration = p.get_critical_path()
        assert path == ["A", "B", "C"]
        assert duration == 60  # 10 + 20 + 30

    def test_critical_path_diamond(self):
        p = self._make_diamond_pipeline()
        path, duration = p.get_critical_path()
        # Longest: A(10) → C(30) → D(15) = 55
        # Shorter: A(10) → B(20) → D(15) = 45
        assert path == ["A", "C", "D"]
        assert duration == 55

    # --- Ready tasks ---

    def test_ready_tasks_initial(self):
        p = self._make_diamond_pipeline()
        ready = p.get_ready_tasks()
        assert ready == ["A"]  # only root is ready

    def test_ready_tasks_after_completion(self):
        p = self._make_diamond_pipeline()
        p.tasks["A"].status = TaskStatus.COMPLETED
        ready = p.get_ready_tasks()
        assert set(ready) == {"B", "C"}  # both branches unlocked

    def test_ready_tasks_partial_completion(self):
        p = self._make_diamond_pipeline()
        p.tasks["A"].status = TaskStatus.COMPLETED
        p.tasks["B"].status = TaskStatus.COMPLETED
        # C is still pending, so D is not ready (D needs both B and C)
        ready = p.get_ready_tasks()
        assert ready == ["C"]

    def test_ready_tasks_all_deps_met(self):
        p = self._make_diamond_pipeline()
        p.tasks["A"].status = TaskStatus.COMPLETED
        p.tasks["B"].status = TaskStatus.COMPLETED
        p.tasks["C"].status = TaskStatus.COMPLETED
        ready = p.get_ready_tasks()
        assert ready == ["D"]

    # --- Roots and leaves ---

    def test_roots_and_leaves_linear(self):
        p = self._make_linear_pipeline()
        assert p.get_all_roots() == ["A"]
        assert p.get_all_leaves() == ["C"]

    def test_roots_and_leaves_diamond(self):
        p = self._make_diamond_pipeline()
        assert p.get_all_roots() == ["A"]
        assert p.get_all_leaves() == ["D"]

    # --- Reset ---

    def test_reset_statuses(self):
        p = self._make_linear_pipeline()
        p.tasks["A"].status = TaskStatus.COMPLETED
        p.tasks["B"].status = TaskStatus.RUNNING
        p.reset_all_statuses()
        assert all(t.status == TaskStatus.PENDING for t in p.tasks.values())

    # --- Validation ---

    def test_validate_healthy_pipeline(self):
        p = self._make_linear_pipeline()
        issues = p.validate()
        assert len(issues) == 0

    def test_validate_no_resources(self):
        p = Pipeline()
        p.add_task(Task(task_id="A", name="A", duration_estimate=10))
        p.resources = []
        issues = p.validate()
        assert any("No resources" in i for i in issues)

    def test_validate_task_exceeds_resources(self):
        p = Pipeline()
        p.add_task(Task(task_id="big", name="Big Task", duration_estimate=10,
                        cpu_required=32, memory_required=128.0))
        p.add_resource(Resource(resource_id="r1", name="Small", cpu_capacity=4))
        issues = p.validate()
        assert any("no resource" in i.lower() for i in issues)

    # --- JSON serialization ---

    def test_save_and_load_json(self):
        p = self._make_diamond_pipeline()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
        try:
            p.save_json(filepath)
            loaded = Pipeline.load_json(filepath)
            assert loaded.name == "diamond"
            assert len(loaded.tasks) == 4
            assert loaded.dag.number_of_edges() == 4
            assert len(loaded.resources) == 1
            assert loaded.get_dependencies("D") == ["B", "C"] or \
                   set(loaded.get_dependencies("D")) == {"B", "C"}
        finally:
            os.remove(filepath)

    def test_load_sample_dags(self):
        """Verify all 3 sample DAGs load without errors."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        for filename in ["simple_dag.json", "medium_dag.json", "complex_dag.json"]:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                p = Pipeline.load_json(filepath)
                assert len(p.tasks) > 0
                assert len(p.resources) > 0
                issues = p.validate()
                errors = [i for i in issues if i.startswith("ERROR")]
                assert len(errors) == 0, f"{filename} has errors: {errors}"

    # --- Summary ---

    def test_summary(self):
        p = self._make_linear_pipeline()
        s = p.summary()
        assert "linear" in s
        assert "Tasks: 3" in s
        assert "Critical path" in s
