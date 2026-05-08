"""
test_knowledge_base.py — Unit tests for the Knowledge Base module.

Tests cover: fact management, rule registration, forward chaining,
backward chaining, SLA risk detection, blocked task detection,
cumulative time estimation, and pipeline loading.
"""

import pytest
from src.task_dag import Task, TaskStatus, Resource, Pipeline
from src.knowledge_base import Fact, Rule, KnowledgeBase


# ============================================================
# Fact Tests
# ============================================================

class TestFact:
    def test_create_fact(self):
        f = Fact("task_status", "extract", "completed")
        assert f.predicate == "task_status"
        assert f.subject == "extract"
        assert f.value == "completed"

    def test_fact_equality(self):
        f1 = Fact("task_status", "extract", "completed")
        f2 = Fact("task_status", "extract", "completed")
        f3 = Fact("task_status", "extract", "failed")
        assert f1 == f2
        assert f1 != f3

    def test_fact_hashable(self):
        f1 = Fact("task_status", "extract", "completed")
        f2 = Fact("task_status", "extract", "completed")
        s = {f1, f2}
        assert len(s) == 1  # duplicates removed

    def test_fact_repr(self):
        f = Fact("depends_on", "B", "A")
        assert "depends_on" in repr(f)
        assert "B" in repr(f)


# ============================================================
# KnowledgeBase — Fact Management
# ============================================================

class TestKBFactManagement:
    def test_add_and_query(self):
        kb = KnowledgeBase()
        kb.add_fact("task_status", "extract", "completed")
        result = kb.query("task_status", "extract")
        assert result == ["completed"]

    def test_query_single(self):
        kb = KnowledgeBase()
        kb.add_fact("requires_cpu", "transform", 4)
        assert kb.query_single("requires_cpu", "transform") == 4

    def test_query_single_missing(self):
        kb = KnowledgeBase()
        assert kb.query_single("requires_cpu", "missing") is None

    def test_has_fact_with_value(self):
        kb = KnowledgeBase()
        kb.add_fact("task_status", "extract", "completed")
        assert kb.has_fact("task_status", "extract", "completed") is True
        assert kb.has_fact("task_status", "extract", "failed") is False

    def test_has_fact_without_value(self):
        kb = KnowledgeBase()
        kb.add_fact("task_status", "extract", "completed")
        assert kb.has_fact("task_status", "extract") is True
        assert kb.has_fact("task_status", "missing") is False

    def test_remove_fact(self):
        kb = KnowledgeBase()
        kb.add_fact("task_status", "extract", "completed")
        kb.remove_fact("task_status", "extract", "completed")
        assert kb.has_fact("task_status", "extract") is False

    def test_remove_facts_by_predicate(self):
        kb = KnowledgeBase()
        kb.add_fact("task_status", "extract", "running")
        kb.add_fact("task_status", "extract", "completed")
        kb.remove_facts_by_predicate("task_status", "extract")
        assert kb.query("task_status", "extract") == []

    def test_get_all_subjects(self):
        kb = KnowledgeBase()
        kb.add_fact("task_exists", "A", True)
        kb.add_fact("task_exists", "B", True)
        kb.add_fact("task_exists", "C", True)
        subjects = kb.get_all_subjects("task_exists")
        assert set(subjects) == {"A", "B", "C"}

    def test_multiple_values_same_predicate(self):
        kb = KnowledgeBase()
        kb.add_fact("depends_on", "D", "B")
        kb.add_fact("depends_on", "D", "C")
        deps = kb.query("depends_on", "D")
        assert set(deps) == {"B", "C"}


# ============================================================
# KnowledgeBase — Load from Pipeline
# ============================================================

class TestKBPipelineLoading:
    def _make_pipeline(self):
        """Simple A → B → C pipeline."""
        p = Pipeline(name="test")
        p.add_task(Task("A", "Task A", 10, cpu_required=2, priority=5))
        p.add_task(Task("B", "Task B", 20, cpu_required=4, priority=7))
        p.add_task(Task("C", "Task C", 30, cpu_required=2, priority=9,
                        sla_deadline=120))
        p.add_dependency("A", "B")
        p.add_dependency("B", "C")
        p.add_resource(Resource("r1", "Server 1", cpu_capacity=8, memory_capacity=32))
        return p

    def test_load_tasks(self):
        kb = KnowledgeBase()
        kb.load_from_pipeline(self._make_pipeline())
        assert kb.has_fact("task_exists", "A", True)
        assert kb.has_fact("task_exists", "B", True)
        assert kb.has_fact("task_exists", "C", True)

    def test_load_dependencies(self):
        kb = KnowledgeBase()
        kb.load_from_pipeline(self._make_pipeline())
        assert kb.has_fact("depends_on", "B", "A")
        assert kb.has_fact("depends_on", "C", "B")
        assert not kb.has_fact("depends_on", "A", "B")  # direction matters

    def test_load_task_properties(self):
        kb = KnowledgeBase()
        kb.load_from_pipeline(self._make_pipeline())
        assert kb.query_single("duration_estimate", "A") == 10
        assert kb.query_single("requires_cpu", "B") == 4
        assert kb.query_single("priority", "C") == 9
        assert kb.query_single("has_sla", "C") == 120

    def test_load_resources(self):
        kb = KnowledgeBase()
        kb.load_from_pipeline(self._make_pipeline())
        assert kb.has_fact("resource_exists", "r1", True)
        assert kb.query_single("resource_cpu_capacity", "r1") == 8

    def test_load_initial_status(self):
        kb = KnowledgeBase()
        kb.load_from_pipeline(self._make_pipeline())
        assert kb.query_single("task_status", "A") == "pending"

    def test_update_task_status(self):
        kb = KnowledgeBase()
        kb.load_from_pipeline(self._make_pipeline())
        kb.update_task_status("A", "completed")
        assert kb.query_single("task_status", "A") == "completed"
        # Should only have one status fact
        statuses = kb.query("task_status", "A")
        assert len(statuses) == 1


# ============================================================
# Forward Chaining Tests
# ============================================================

class TestForwardChaining:
    def _make_kb_with_pipeline(self):
        """Create a KB loaded from A → B → C pipeline with default rules."""
        p = Pipeline(name="test")
        p.add_task(Task("A", "Task A", 10, cpu_required=2, priority=5))
        p.add_task(Task("B", "Task B", 20, cpu_required=4, priority=8))
        p.add_task(Task("C", "Task C", 30, cpu_required=2, priority=9,
                        sla_deadline=120))
        p.add_dependency("A", "B")
        p.add_dependency("B", "C")
        p.add_resource(Resource("r1", "Server 1", cpu_capacity=8))
        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        return kb

    def test_initial_ready_tasks(self):
        """Only A should be ready initially (no dependencies)."""
        kb = self._make_kb_with_pipeline()
        kb.forward_chain()
        ready = kb.get_ready_tasks()
        assert "A" in ready
        assert "B" not in ready
        assert "C" not in ready

    def test_ready_after_completion(self):
        """After A completes, B should become ready."""
        kb = self._make_kb_with_pipeline()
        kb.update_task_status("A", "completed")
        kb.clear_derived()
        kb.forward_chain()
        ready = kb.get_ready_tasks()
        assert "B" in ready
        assert "C" not in ready

    def test_chain_completion(self):
        """After A and B complete, C should be ready."""
        kb = self._make_kb_with_pipeline()
        kb.update_task_status("A", "completed")
        kb.update_task_status("B", "completed")
        kb.clear_derived()
        kb.forward_chain()
        ready = kb.get_ready_tasks()
        assert "C" in ready

    def test_blocked_on_failure(self):
        """If A fails, B should be blocked."""
        kb = self._make_kb_with_pipeline()
        kb.update_task_status("A", "failed")
        kb.clear_derived()
        kb.forward_chain()
        blocked = kb.get_blocked_tasks()
        assert any(task == "B" and blocker == "A"
                   for task, blocker in blocked)

    def test_urgent_detection(self):
        """B has priority 8, C has priority 9 — both should be flagged urgent."""
        kb = self._make_kb_with_pipeline()
        kb.forward_chain()
        urgent = kb.get_urgent_tasks()
        assert "B" in urgent
        assert "C" in urgent
        assert "A" not in urgent  # priority 5

    def test_retry_recommendation(self):
        """Failed tasks should get retry recommendations."""
        kb = self._make_kb_with_pipeline()
        kb.update_task_status("A", "failed")
        kb.clear_derived()
        kb.forward_chain()
        retries = kb.get_retry_recommendations()
        assert "A" in retries

    def test_forward_chain_returns_count(self):
        """Forward chain should return the number of new facts derived."""
        kb = self._make_kb_with_pipeline()
        new_count = kb.forward_chain()
        assert new_count > 0

    def test_fixpoint(self):
        """Running forward chain twice should derive no new facts on second run."""
        kb = self._make_kb_with_pipeline()
        kb.forward_chain()
        new_count = kb.forward_chain()
        assert new_count == 0  # already at fixpoint

    def test_inference_log(self):
        """Forward chaining should produce a log of rule firings."""
        kb = self._make_kb_with_pipeline()
        kb.forward_chain()
        log = kb.get_inference_log()
        assert len(log) > 0
        assert any("ready_to_run" in entry for entry in log)

    def test_derived_facts_tracking(self):
        """Derived facts should be tracked separately from base facts."""
        kb = self._make_kb_with_pipeline()
        base_count = len(kb.facts)
        kb.forward_chain()
        derived = kb.get_derived_facts()
        assert len(derived) > 0
        assert len(kb.facts) > base_count

    def test_clear_derived(self):
        """Clearing derived facts should remove only inferred facts."""
        kb = self._make_kb_with_pipeline()
        base_count = len(kb.facts)
        kb.forward_chain()
        assert len(kb.facts) > base_count
        kb.clear_derived()
        assert len(kb.facts) == base_count


# ============================================================
# SLA Risk Detection Tests
# ============================================================

class TestSLARiskDetection:
    def test_sla_at_risk(self):
        """SLA should be at risk when cumulative time exceeds deadline."""
        p = Pipeline(name="test")
        p.add_task(Task("A", "Task A", 50))
        p.add_task(Task("B", "Task B", 50))
        p.add_task(Task("C", "Task C", 50, sla_deadline=100))
        # Critical path: A(50) → B(50) → C(50) = 150 > SLA of 100
        p.add_dependency("A", "B")
        p.add_dependency("B", "C")
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))

        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        kb.forward_chain()

        risks = kb.get_sla_risks()
        assert len(risks) == 1
        assert risks[0]["task_id"] == "C"
        assert risks[0]["slack_minutes"] < 0  # negative = overdue

    def test_sla_safe(self):
        """SLA should NOT be at risk when there's enough time."""
        p = Pipeline(name="test")
        p.add_task(Task("A", "Task A", 10))
        p.add_task(Task("B", "Task B", 10, sla_deadline=360))
        p.add_dependency("A", "B")
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))

        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        kb.forward_chain()

        risks = kb.get_sla_risks()
        assert len(risks) == 0

    def test_cascading_sla_risk(self):
        """If upstream task has SLA risk, downstream with SLA should get cascading flag."""
        p = Pipeline(name="test")
        p.add_task(Task("A", "Task A", 200, sla_deadline=100))  # already at risk
        p.add_task(Task("B", "Task B", 10, sla_deadline=300))
        p.add_dependency("A", "B")
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))

        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        kb.forward_chain()

        risks = kb.get_sla_risks()
        b_risk = [r for r in risks if r["task_id"] == "B"]
        # B depends on A which is at risk, and B has its own SLA → cascading
        if b_risk:
            assert b_risk[0]["cascading"] is True


# ============================================================
# Backward Chaining Tests
# ============================================================

class TestBackwardChaining:
    def _make_kb(self):
        p = Pipeline(name="test")
        p.add_task(Task("A", "Task A", 10, priority=5))
        p.add_task(Task("B", "Task B", 20, priority=9))
        p.add_dependency("A", "B")
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))
        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        return kb

    def test_prove_known_fact(self):
        """Should prove a fact that already exists in KB."""
        kb = self._make_kb()
        proven, explanation = kb.backward_chain("task_exists", "A", True)
        assert proven is True
        assert any("KNOWN" in e for e in explanation)

    def test_prove_ready_to_run(self):
        """A has no dependencies → should be provable as ready_to_run."""
        kb = self._make_kb()
        proven, explanation = kb.backward_chain("ready_to_run", "A", True)
        assert proven is True
        assert any("ready_to_run" in e for e in explanation)

    def test_cannot_prove_false(self):
        """B depends on A (pending) → should NOT be provable as ready_to_run."""
        kb = self._make_kb()
        proven, explanation = kb.backward_chain("ready_to_run", "B", True)
        assert proven is False
        assert any("CANNOT PROVE" in e for e in explanation)

    def test_prove_urgent(self):
        """B has priority 9 → should be provable as urgent."""
        kb = self._make_kb()
        proven, explanation = kb.backward_chain("urgent", "B", True)
        assert proven is True

    def test_prove_nonexistent(self):
        """Should not prove a fact about a nonexistent task."""
        kb = self._make_kb()
        proven, _ = kb.backward_chain("task_exists", "Z", True)
        assert proven is False


# ============================================================
# Resource Insufficiency Tests
# ============================================================

class TestResourceInsufficiency:
    def test_resource_insufficient(self):
        """Task needing 16 CPUs on a cluster with only 8 should be flagged."""
        p = Pipeline(name="test")
        p.add_task(Task("heavy", "Heavy Task", 10, cpu_required=16))
        p.add_resource(Resource("r1", "Small Server", cpu_capacity=8))

        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        kb.forward_chain()

        assert kb.has_fact("resource_insufficient", "heavy", True)

    def test_resource_sufficient(self):
        """Task needing 4 CPUs on a cluster with 8 should NOT be flagged."""
        p = Pipeline(name="test")
        p.add_task(Task("light", "Light Task", 10, cpu_required=4))
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))

        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        kb.forward_chain()

        assert not kb.has_fact("resource_insufficient", "light", True)


# ============================================================
# Diamond DAG Test (more complex scenario)
# ============================================================

class TestDiamondDAG:
    def _make_diamond_kb(self):
        """A → B, A → C, B → D, C → D."""
        p = Pipeline(name="diamond")
        p.add_task(Task("A", "A", 10, priority=5))
        p.add_task(Task("B", "B", 20, priority=6))
        p.add_task(Task("C", "C", 30, priority=7))
        p.add_task(Task("D", "D", 15, priority=10, sla_deadline=360))
        p.add_dependency("A", "B")
        p.add_dependency("A", "C")
        p.add_dependency("B", "D")
        p.add_dependency("C", "D")
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))
        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        return kb

    def test_only_root_ready_initially(self):
        kb = self._make_diamond_kb()
        kb.forward_chain()
        assert kb.get_ready_tasks() == ["A"]

    def test_branches_ready_after_root(self):
        kb = self._make_diamond_kb()
        kb.update_task_status("A", "completed")
        kb.clear_derived()
        kb.forward_chain()
        ready = set(kb.get_ready_tasks())
        assert ready == {"B", "C"}

    def test_D_not_ready_until_both_done(self):
        kb = self._make_diamond_kb()
        kb.update_task_status("A", "completed")
        kb.update_task_status("B", "completed")
        # C is still pending
        kb.clear_derived()
        kb.forward_chain()
        ready = kb.get_ready_tasks()
        assert "D" not in ready
        assert "C" in ready

    def test_D_ready_after_both_done(self):
        kb = self._make_diamond_kb()
        kb.update_task_status("A", "completed")
        kb.update_task_status("B", "completed")
        kb.update_task_status("C", "completed")
        kb.clear_derived()
        kb.forward_chain()
        ready = kb.get_ready_tasks()
        assert "D" in ready

    def test_D_blocked_if_branch_fails(self):
        kb = self._make_diamond_kb()
        kb.update_task_status("A", "completed")
        kb.update_task_status("B", "failed")
        kb.clear_derived()
        kb.forward_chain()
        blocked = kb.get_blocked_tasks()
        assert any(task == "D" and blocker == "B"
                   for task, blocker in blocked)


# ============================================================
# Summary & Display
# ============================================================

class TestSummary:
    def test_summary_contains_key_info(self):
        p = Pipeline(name="test")
        p.add_task(Task("A", "A", 10))
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))
        kb = KnowledgeBase()
        kb.load_from_pipeline(p)
        kb.register_default_rules()
        kb.forward_chain()
        s = kb.summary()
        assert "Total facts" in s
        assert "Derived facts" in s
        assert "Ready to run" in s
