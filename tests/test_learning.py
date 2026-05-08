"""
test_learning.py — Unit tests for the Learning module.

Tests cover: initial estimates, EWMA updates, SMA updates, error tracking,
improvement over static, pipeline loading, reset, and convergence behavior.
"""

import pytest
from src.task_dag import Task, Pipeline, Resource
from src.learning import RuntimeEstimator, RuntimeRecord


# ============================================================
# Basic Functionality
# ============================================================

class TestBasic:
    def test_set_initial_estimate(self):
        est = RuntimeEstimator()
        est.set_initial_estimate("A", 60.0)
        assert est.get_estimate("A") == 60.0

    def test_get_static_estimate(self):
        est = RuntimeEstimator()
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 80.0, iteration=1)
        assert est.get_static_estimate("A") == 60.0  # never changes

    def test_record_returns_record(self):
        est = RuntimeEstimator()
        est.set_initial_estimate("A", 60.0)
        record = est.record_runtime("A", 72.0, iteration=1)
        assert isinstance(record, RuntimeRecord)
        assert record.actual_duration == 72.0
        assert record.estimated_duration == 60.0
        assert record.error == -12.0  # 60 - 72

    def test_history_tracking(self):
        est = RuntimeEstimator()
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 72.0, iteration=1)
        est.record_runtime("A", 68.0, iteration=2)
        history = est.get_history("A")
        assert len(history) == 2

    def test_iteration_count(self):
        est = RuntimeEstimator()
        est.set_initial_estimate("A", 60.0)
        assert est.get_iteration_count("A") == 0
        est.record_runtime("A", 72.0, iteration=1)
        assert est.get_iteration_count("A") == 1

    def test_empty_history(self):
        est = RuntimeEstimator()
        assert est.get_history("nonexistent") == []
        assert est.get_absolute_error("nonexistent") is None

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            RuntimeEstimator(alpha=0.0)
        with pytest.raises(ValueError):
            RuntimeEstimator(alpha=1.0)
        with pytest.raises(ValueError):
            RuntimeEstimator(alpha=-0.5)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            RuntimeEstimator(method="invalid")


# ============================================================
# EWMA Tests
# ============================================================

class TestEWMA:
    def test_ewma_single_observation(self):
        est = RuntimeEstimator(alpha=0.3, method="ewma")
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 90.0, iteration=1)
        # EWMA: 0.3 * 90 + 0.7 * 60 = 27 + 42 = 69
        assert abs(est.get_estimate("A") - 69.0) < 0.01

    def test_ewma_multiple_observations(self):
        est = RuntimeEstimator(alpha=0.3, method="ewma")
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 90.0, iteration=1)
        # After 1: 0.3*90 + 0.7*60 = 69.0
        est.record_runtime("A", 90.0, iteration=2)
        # After 2: 0.3*90 + 0.7*69 = 27 + 48.3 = 75.3
        assert abs(est.get_estimate("A") - 75.3) < 0.01

    def test_ewma_converges_toward_actual(self):
        """EWMA should converge toward the true runtime over many iterations."""
        est = RuntimeEstimator(alpha=0.3, method="ewma")
        est.set_initial_estimate("A", 60.0)
        true_runtime = 90.0

        for i in range(20):
            est.record_runtime("A", true_runtime, iteration=i + 1)

        estimate = est.get_estimate("A")
        # After 20 iterations, should be very close to 90
        assert abs(estimate - true_runtime) < 1.0

    def test_ewma_higher_alpha_converges_faster(self):
        """Higher alpha should converge faster."""
        est_fast = RuntimeEstimator(alpha=0.5, method="ewma")
        est_slow = RuntimeEstimator(alpha=0.1, method="ewma")

        est_fast.set_initial_estimate("A", 60.0)
        est_slow.set_initial_estimate("A", 60.0)

        true_runtime = 90.0
        for i in range(5):
            est_fast.record_runtime("A", true_runtime, iteration=i + 1)
            est_slow.record_runtime("A", true_runtime, iteration=i + 1)

        # Fast should be closer to 90 after 5 iterations
        fast_error = abs(est_fast.get_estimate("A") - true_runtime)
        slow_error = abs(est_slow.get_estimate("A") - true_runtime)
        assert fast_error < slow_error

    def test_ewma_adapts_to_change(self):
        """EWMA should adapt when the true runtime changes."""
        est = RuntimeEstimator(alpha=0.3, method="ewma")
        est.set_initial_estimate("A", 60.0)

        # First: runtime is 60
        for i in range(5):
            est.record_runtime("A", 60.0, iteration=i + 1)

        # Then: runtime jumps to 120
        for i in range(10):
            est.record_runtime("A", 120.0, iteration=i + 6)

        # Should have adapted toward 120
        estimate = est.get_estimate("A")
        assert estimate > 100  # should be close to 120


# ============================================================
# SMA Tests
# ============================================================

class TestSMA:
    def test_sma_single_observation(self):
        est = RuntimeEstimator(method="sma")
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 90.0, iteration=1)
        assert est.get_estimate("A") == 90.0  # average of [90]

    def test_sma_multiple_observations(self):
        est = RuntimeEstimator(method="sma")
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 80.0, iteration=1)
        est.record_runtime("A", 100.0, iteration=2)
        assert est.get_estimate("A") == 90.0  # average of [80, 100]

    def test_sma_falls_back_to_initial(self):
        est = RuntimeEstimator(method="sma")
        est.set_initial_estimate("A", 60.0)
        assert est.get_estimate("A") == 60.0  # no history, use initial


# ============================================================
# Error Metrics Tests
# ============================================================

class TestErrorMetrics:
    def test_absolute_error(self):
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 72.0, iteration=1)  # error = |60-72| = 12
        est.record_runtime("A", 65.0, iteration=2)  # error = |69-65| ≈ 4 (uses EWMA)
        mae = est.get_absolute_error("A")
        assert mae is not None
        assert mae > 0

    def test_global_mae(self):
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        est.set_initial_estimate("B", 30.0)
        est.record_runtime("A", 72.0, iteration=1)
        est.record_runtime("B", 35.0, iteration=1)
        mae = est.get_global_mae()
        assert mae is not None
        assert mae > 0

    def test_global_mae_none_when_empty(self):
        est = RuntimeEstimator()
        assert est.get_global_mae() is None

    def test_error_over_iterations(self):
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        for i in range(5):
            est.record_runtime("A", 80.0, iteration=i + 1)
        errors = est.get_error_over_iterations("A")
        assert len(errors) == 5
        # Errors should generally decrease as EWMA converges
        first_error = errors[0][1]
        last_error = errors[-1][1]
        assert last_error < first_error

    def test_improvement_over_static(self):
        """Learning should show improvement over static estimates."""
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        # True runtime is 90 — static estimate of 60 is always wrong by 30
        for i in range(10):
            est.record_runtime("A", 90.0, iteration=i + 1)

        improvement = est.get_improvement_over_static("A")
        assert improvement is not None
        assert improvement > 0  # learning should be better than static

    def test_all_errors(self):
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        est.set_initial_estimate("B", 30.0)
        est.record_runtime("A", 72.0, iteration=1)
        est.record_runtime("B", 35.0, iteration=1)
        errors = est.get_all_errors()
        assert "A" in errors
        assert "B" in errors


# ============================================================
# Pipeline Integration
# ============================================================

class TestPipelineIntegration:
    def test_load_from_pipeline(self):
        p = Pipeline(name="test")
        p.add_task(Task("A", "A", 30))
        p.add_task(Task("B", "B", 45))
        p.add_resource(Resource("r1", "Server", cpu_capacity=8))

        est = RuntimeEstimator()
        est.set_initial_estimates_from_pipeline(p)
        assert est.get_estimate("A") == 30.0
        assert est.get_estimate("B") == 45.0


# ============================================================
# Reset
# ============================================================

class TestReset:
    def test_reset_clears_history(self):
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 80.0, iteration=1)
        assert est.get_iteration_count("A") == 1

        est.reset()
        assert est.get_iteration_count("A") == 0
        assert est.get_estimate("A") == 60.0  # back to initial


# ============================================================
# Summary
# ============================================================

class TestSummary:
    def test_summary_contains_info(self):
        est = RuntimeEstimator(alpha=0.3)
        est.set_initial_estimate("A", 60.0)
        est.record_runtime("A", 72.0, iteration=1)
        s = est.summary()
        assert "ewma" in s
        assert "alpha=0.3" in s
        assert "A" in s
