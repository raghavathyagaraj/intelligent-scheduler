"""
learning.py — Learning Component for Runtime Estimation.

Tracks actual task runtimes and improves future estimates using:
- Simple Moving Average (SMA): baseline estimator.
- Exponential Weighted Moving Average (EWMA): recency-biased estimator.
- Error tracking: measures prediction accuracy over iterations.

The scheduler agent uses these improved estimates to make better
scheduling decisions (feed into CSP domains and search heuristics).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeRecord:
    """A single observed runtime for a task."""
    task_id: str
    estimated_duration: float   # what we predicted
    actual_duration: float      # what actually happened
    iteration: int              # which run this was
    error: float = 0.0         # estimated - actual

    def __post_init__(self):
        self.error = self.estimated_duration - self.actual_duration


class RuntimeEstimator:
    """
    Learns task runtime estimates from historical observations.

    Supports two estimation strategies:
    - SMA (Simple Moving Average): equal weight to all past observations.
    - EWMA (Exponential Weighted Moving Average): recent observations
      weighted more heavily (controlled by alpha parameter).

    Usage:
        estimator = RuntimeEstimator(alpha=0.3)
        estimator.set_initial_estimate("task_A", 60.0)

        # After task runs:
        estimator.record_runtime("task_A", actual_duration=72.0, iteration=1)

        # Get improved estimate for next run:
        next_estimate = estimator.get_estimate("task_A")  # uses EWMA
    """

    def __init__(self, alpha: float = 0.3, method: str = "ewma"):
        """
        Args:
            alpha: EWMA smoothing factor (0 < alpha < 1).
                   Higher alpha = more weight on recent observations.
                   Typical values: 0.2-0.4.
            method: "ewma" or "sma" (simple moving average).
        """
        if not 0 < alpha < 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {alpha}")
        if method not in ("ewma", "sma"):
            raise ValueError(f"Method must be 'ewma' or 'sma', got {method}")

        self.alpha = alpha
        self.method = method

        # Storage
        self._initial_estimates: dict[str, float] = {}
        self._history: dict[str, list[RuntimeRecord]] = {}
        self._ewma_estimates: dict[str, float] = {}

    # ================================================================
    # Initial Estimates
    # ================================================================

    def set_initial_estimate(self, task_id: str, estimate: float):
        """Set the initial (static) estimate for a task."""
        self._initial_estimates[task_id] = estimate
        if task_id not in self._ewma_estimates:
            self._ewma_estimates[task_id] = estimate

    def set_initial_estimates_from_pipeline(self, pipeline) -> None:
        """Load initial estimates from a Pipeline object."""
        for task_id, task in pipeline.tasks.items():
            self.set_initial_estimate(task_id, task.duration_estimate)

    # ================================================================
    # Record Observations
    # ================================================================

    def record_runtime(self, task_id: str, actual_duration: float,
                        iteration: int) -> RuntimeRecord:
        """
        Record an observed runtime and update estimates.

        Args:
            task_id: The task that ran.
            actual_duration: How long it actually took (minutes).
            iteration: Which iteration/run number this is.

        Returns:
            The RuntimeRecord created.
        """
        # Get current estimate
        current_estimate = self.get_estimate(task_id)

        # Create record
        record = RuntimeRecord(
            task_id=task_id,
            estimated_duration=current_estimate,
            actual_duration=actual_duration,
            iteration=iteration
        )

        # Store in history
        if task_id not in self._history:
            self._history[task_id] = []
        self._history[task_id].append(record)

        # Update EWMA estimate
        if self.method == "ewma":
            self._ewma_estimates[task_id] = (
                self.alpha * actual_duration +
                (1 - self.alpha) * current_estimate
            )
        # SMA doesn't need incremental update — computed on the fly

        return record

    # ================================================================
    # Get Estimates
    # ================================================================

    def get_estimate(self, task_id: str) -> float:
        """
        Get the current best estimate for a task's runtime.

        Returns EWMA or SMA estimate depending on configured method.
        Falls back to initial estimate if no history exists.
        """
        if self.method == "ewma":
            return self._get_ewma_estimate(task_id)
        else:
            return self._get_sma_estimate(task_id)

    def _get_ewma_estimate(self, task_id: str) -> float:
        """Get EWMA estimate (already maintained incrementally)."""
        if task_id in self._ewma_estimates:
            return self._ewma_estimates[task_id]
        return self._initial_estimates.get(task_id, 0.0)

    def _get_sma_estimate(self, task_id: str) -> float:
        """Get Simple Moving Average estimate."""
        if task_id in self._history and self._history[task_id]:
            actuals = [r.actual_duration for r in self._history[task_id]]
            return sum(actuals) / len(actuals)
        return self._initial_estimates.get(task_id, 0.0)

    def get_static_estimate(self, task_id: str) -> float:
        """Get the original static estimate (never changes)."""
        return self._initial_estimates.get(task_id, 0.0)

    # ================================================================
    # Error Metrics
    # ================================================================

    def get_absolute_error(self, task_id: str) -> Optional[float]:
        """
        Get the mean absolute error for a task's estimates.
        Returns None if no history exists.
        """
        if task_id not in self._history or not self._history[task_id]:
            return None
        errors = [abs(r.error) for r in self._history[task_id]]
        return sum(errors) / len(errors)

    def get_error_over_iterations(self, task_id: str) -> list[tuple[int, float]]:
        """
        Get the absolute error at each iteration for a task.

        Returns:
            List of (iteration, absolute_error) tuples.
        """
        if task_id not in self._history:
            return []
        return [
            (r.iteration, abs(r.error))
            for r in self._history[task_id]
        ]

    def get_all_errors(self) -> dict[str, float]:
        """Get mean absolute error for all tasks with history."""
        errors = {}
        for task_id in self._history:
            err = self.get_absolute_error(task_id)
            if err is not None:
                errors[task_id] = err
        return errors

    def get_global_mae(self) -> Optional[float]:
        """Get the global Mean Absolute Error across all tasks and iterations."""
        all_errors = []
        for records in self._history.values():
            for r in records:
                all_errors.append(abs(r.error))
        if not all_errors:
            return None
        return sum(all_errors) / len(all_errors)

    def get_improvement_over_static(self, task_id: str) -> Optional[float]:
        """
        Compare learned estimate error vs static estimate error.

        Returns:
            Percentage improvement. Positive = learning is better.
            None if no history.
        """
        if task_id not in self._history or not self._history[task_id]:
            return None

        static_est = self.get_static_estimate(task_id)
        records = self._history[task_id]

        # Static error: what the error would be if we always used initial estimate
        static_errors = [abs(static_est - r.actual_duration) for r in records]
        static_mae = sum(static_errors) / len(static_errors)

        # Learned error: what the actual errors were with our adaptive estimates
        learned_errors = [abs(r.error) for r in records]
        learned_mae = sum(learned_errors) / len(learned_errors)

        if static_mae == 0:
            return 0.0

        improvement = ((static_mae - learned_mae) / static_mae) * 100.0
        return improvement

    # ================================================================
    # History Access
    # ================================================================

    def get_history(self, task_id: str) -> list[RuntimeRecord]:
        """Get all runtime records for a task."""
        return self._history.get(task_id, [])

    def get_all_history(self) -> dict[str, list[RuntimeRecord]]:
        """Get complete history for all tasks."""
        return dict(self._history)

    def get_iteration_count(self, task_id: str) -> int:
        """How many observations we have for a task."""
        return len(self._history.get(task_id, []))

    # ================================================================
    # Reset
    # ================================================================

    def reset(self):
        """Clear all history and reset estimates to initial values."""
        self._history.clear()
        self._ewma_estimates = dict(self._initial_estimates)

    # ================================================================
    # Summary
    # ================================================================

    def summary(self) -> str:
        """Human-readable summary of the learning state."""
        lines = [
            f"Runtime Estimator (method={self.method}, alpha={self.alpha})",
            f"  Tasks tracked: {len(self._initial_estimates)}",
            f"  Tasks with history: {len(self._history)}",
        ]

        total_records = sum(len(v) for v in self._history.values())
        lines.append(f"  Total observations: {total_records}")

        mae = self.get_global_mae()
        if mae is not None:
            lines.append(f"  Global MAE: {mae:.2f} minutes")

        for task_id in sorted(self._history.keys()):
            static = self.get_static_estimate(task_id)
            current = self.get_estimate(task_id)
            err = self.get_absolute_error(task_id)
            improvement = self.get_improvement_over_static(task_id)
            n = self.get_iteration_count(task_id)
            lines.append(
                f"  {task_id}: static={static:.0f}m → learned={current:.1f}m "
                f"(MAE={err:.1f}m, improvement={improvement:+.1f}%, "
                f"n={n})"
            )

        return "\n".join(lines)
