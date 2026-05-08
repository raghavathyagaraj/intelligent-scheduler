"""
evaluate.py — Run all evaluation experiments and generate results.

Experiments:
1. Correctness: CSP produces zero-violation schedules.
2. Search comparison: A* vs Greedy — nodes explored, cost.
3. Learning improvement: Error decreases over 10 iterations.
4. Scalability: Solve time vs DAG size (5, 10, 20 tasks).
5. Failure recovery: SLA adherence with 1, 2, 3 failures.
6. Agent vs Naive baseline: makespan, SLA, tasks completed.

Run: python evaluation/evaluate.py
Outputs: evaluation/results/*.csv, evaluation/results/*.png
"""

import os
import sys
import csv
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.task_dag import Pipeline
from src.csp_solver import CSPSolver
from src.search_planner import SearchPlanner
from src.scheduler_agent import SchedulerAgent
from src.learning import RuntimeEstimator
from src.visualizer import (
    plot_learning_curve, plot_metrics_comparison,
    plot_search_comparison, plot_scalability, plot_gantt, plot_dag
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def ensure_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def load(name):
    return Pipeline.load_json(os.path.join(DATA_DIR, f"{name}_dag.json"))


# ================================================================
# Experiment 1: Correctness
# ================================================================

def experiment_1_correctness():
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Correctness — Zero Constraint Violations")
    print("=" * 60)

    results = []
    for dag_name in ["simple", "medium", "complex"]:
        pipeline = load(dag_name)
        solver = CSPSolver(pipeline, time_horizon=600)
        csp_result = solver.solve()
        violations = CSPSolver.validate_schedule(csp_result.schedule, pipeline) \
            if csp_result.success else ["CSP failed"]
        sla_rate = csp_result.schedule.get_sla_adherence_rate(pipeline.tasks) \
            if csp_result.success else 0

        row = {
            "dag": dag_name,
            "tasks": len(pipeline.tasks),
            "solved": csp_result.success,
            "violations": len(violations),
            "sla_adherence": f"{sla_rate:.1f}%",
            "nodes_explored": csp_result.nodes_explored,
            "backtracks": csp_result.backtracks,
        }
        results.append(row)
        status = "✓" if len(violations) == 0 and csp_result.success else "✗"
        print(f"  {status} {dag_name}: {len(pipeline.tasks)} tasks, "
              f"{len(violations)} violations, SLA={sla_rate:.1f}%")

    _write_csv("exp1_correctness.csv", results)
    return results


# ================================================================
# Experiment 2: A* vs Greedy Search Comparison
# ================================================================

def experiment_2_search_comparison():
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: A* vs Greedy Best-First Search")
    print("=" * 60)

    results = []
    for dag_name in ["simple", "medium", "complex"]:
        pipeline = load(dag_name)

        # Create a failure scenario
        task_ids = list(pipeline.tasks.keys())
        first_task = task_ids[0]
        second_task = task_ids[1] if len(task_ids) > 1 else task_ids[0]

        planner_a = SearchPlanner(pipeline, current_time=30)
        astar = planner_a.replan_from_failure(
            failed_task_ids=[second_task],
            completed_task_ids=[first_task],
            current_time=30
        )

        planner_g = SearchPlanner(pipeline, current_time=30)
        greedy = planner_g.search_greedy(
            planner_g.create_initial_state(
                task_statuses={
                    t: "completed" if t == first_task
                    else ("failed" if t == second_task else "pending")
                    for t in task_ids
                }
            )
        )

        row = {
            "dag": dag_name,
            "astar_nodes": astar.nodes_explored,
            "greedy_nodes": greedy.nodes_explored,
            "astar_cost": f"{astar.total_cost:.1f}",
            "greedy_cost": f"{greedy.total_cost:.1f}",
            "astar_sla": f"{astar.sla_adherence:.1f}%",
            "greedy_sla": f"{greedy.sla_adherence:.1f}%",
        }
        results.append(row)
        print(f"  {dag_name}: A*={astar.nodes_explored} nodes "
              f"(cost={astar.total_cost:.1f}), "
              f"Greedy={greedy.nodes_explored} nodes "
              f"(cost={greedy.total_cost:.1f})")

    _write_csv("exp2_search_comparison.csv", results)

    # Plot for the last (complex) DAG
    if results:
        last = results[-1]
        plot_search_comparison(
            int(last["astar_nodes"]), int(last["greedy_nodes"]),
            float(last["astar_cost"]), float(last["greedy_cost"]),
            output_path=os.path.join(RESULTS_DIR, "exp2_search_comparison.png")
        )
    return results


# ================================================================
# Experiment 3: Learning Improvement Over Iterations
# ================================================================

def experiment_3_learning():
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Learning — Error Decreasing Over Iterations")
    print("=" * 60)

    pipeline = load("medium")
    agent = SchedulerAgent(pipeline, ewma_alpha=0.3)
    results_list = agent.run_iterations(n=10, mode="stochastic", base_seed=42)

    estimator = agent.get_estimator()

    rows = []
    for i, r in enumerate(results_list):
        mae = r.metrics.learning_mae
        rows.append({
            "iteration": i + 1,
            "global_mae": f"{mae:.2f}" if mae else "N/A",
            "tasks_completed": r.metrics.tasks_completed,
        })
        print(f"  Iteration {i+1}: MAE={mae:.2f}m" if mae else
              f"  Iteration {i+1}: MAE=N/A")

    _write_csv("exp3_learning.csv", rows)

    # Plot learning curve
    plot_learning_curve(
        estimator,
        output_path=os.path.join(RESULTS_DIR, "exp3_learning_curve.png")
    )

    # Check improvement
    for task_id in list(pipeline.tasks.keys())[:3]:
        improvement = estimator.get_improvement_over_static(task_id)
        if improvement is not None:
            print(f"  {task_id}: {improvement:+.1f}% improvement over static")

    return rows


# ================================================================
# Experiment 4: Scalability
# ================================================================

def experiment_4_scalability():
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Scalability — Solve Time vs DAG Size")
    print("=" * 60)

    results = []
    task_counts = []
    csp_times = []
    search_times = []

    for dag_name in ["simple", "medium", "complex"]:
        pipeline = load(dag_name)
        n_tasks = len(pipeline.tasks)
        task_counts.append(n_tasks)

        # CSP time
        start = time.time()
        solver = CSPSolver(pipeline, time_horizon=600)
        csp_result = solver.solve()
        csp_time = time.time() - start
        csp_times.append(csp_time)

        # Search time
        task_ids = list(pipeline.tasks.keys())
        start = time.time()
        planner = SearchPlanner(pipeline, current_time=0)
        search_result = planner.search_astar()
        search_time = time.time() - start
        search_times.append(search_time)

        row = {
            "dag": dag_name,
            "tasks": n_tasks,
            "csp_time_sec": f"{csp_time:.4f}",
            "csp_nodes": csp_result.nodes_explored,
            "search_time_sec": f"{search_time:.4f}",
            "search_nodes": search_result.nodes_explored,
        }
        results.append(row)
        print(f"  {dag_name} ({n_tasks} tasks): CSP={csp_time:.4f}s, "
              f"Search={search_time:.4f}s")

    _write_csv("exp4_scalability.csv", results)
    plot_scalability(
        task_counts, csp_times, search_times,
        output_path=os.path.join(RESULTS_DIR, "exp4_scalability.png")
    )
    return results


# ================================================================
# Experiment 5: Failure Recovery
# ================================================================

def experiment_5_failure_recovery():
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Failure Recovery — SLA Under Failures")
    print("=" * 60)

    pipeline = load("medium")
    results = []

    for failure_rate in [0.0, 0.1, 0.2, 0.3]:
        agent = SchedulerAgent(Pipeline.load_json(
            os.path.join(DATA_DIR, "medium_dag.json")
        ))
        result = agent.run(mode="failure", failure_rate=failure_rate, seed=42)

        row = {
            "failure_rate": f"{failure_rate:.0%}",
            "tasks_completed": result.metrics.tasks_completed,
            "tasks_failed": result.metrics.tasks_failed,
            "sla_adherence": f"{result.metrics.sla_adherence:.1f}%",
            "replan_count": result.metrics.replan_count,
        }
        results.append(row)
        print(f"  Failure rate={failure_rate:.0%}: "
              f"completed={result.metrics.tasks_completed}/{result.metrics.tasks_total}, "
              f"SLA={result.metrics.sla_adherence:.1f}%, "
              f"replans={result.metrics.replan_count}")

    _write_csv("exp5_failure_recovery.csv", results)
    return results


# ================================================================
# Experiment 6: Agent vs Naive Baseline
# ================================================================

def experiment_6_baseline_comparison():
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: Intelligent Agent vs Naive Baseline")
    print("=" * 60)

    results = []

    for dag_name in ["simple", "medium", "complex"]:
        dag_path = os.path.join(DATA_DIR, f"{dag_name}_dag.json")

        # Agent
        agent = SchedulerAgent(Pipeline.load_json(dag_path))
        agent_result = agent.run(mode="normal", seed=42)

        # Baseline
        baseline = SchedulerAgent(Pipeline.load_json(dag_path))
        baseline_result = baseline.run_naive_baseline(mode="normal", seed=42)

        row = {
            "dag": dag_name,
            "agent_makespan": f"{agent_result.metrics.makespan:.0f}",
            "baseline_makespan": f"{baseline_result.metrics.makespan:.0f}",
            "agent_sla": f"{agent_result.metrics.sla_adherence:.1f}%",
            "baseline_sla": f"{baseline_result.metrics.sla_adherence:.1f}%",
            "agent_completed": agent_result.metrics.tasks_completed,
            "baseline_completed": baseline_result.metrics.tasks_completed,
        }
        results.append(row)
        print(f"  {dag_name}: Agent makespan={agent_result.metrics.makespan:.0f}m "
              f"vs Baseline={baseline_result.metrics.makespan:.0f}m | "
              f"Agent SLA={agent_result.metrics.sla_adherence:.1f}% "
              f"vs Baseline={baseline_result.metrics.sla_adherence:.1f}%")

        # Generate comparison chart for complex DAG
        if dag_name == "complex":
            plot_metrics_comparison(
                agent_result.metrics, baseline_result.metrics,
                output_path=os.path.join(RESULTS_DIR, "exp6_comparison.png")
            )

    # Generate DAG and Gantt for complex dag
    pipeline = Pipeline.load_json(os.path.join(DATA_DIR, "complex_dag.json"))
    agent = SchedulerAgent(pipeline)
    result = agent.run(mode="normal", seed=42)
    if result.schedule:
        plot_dag(pipeline, output_path=os.path.join(RESULTS_DIR, "exp6_dag.png"))
        plot_gantt(result.schedule, pipeline,
                   output_path=os.path.join(RESULTS_DIR, "exp6_gantt.png"))

    _write_csv("exp6_baseline_comparison.csv", results)
    return results


# ================================================================
# CSV Helper
# ================================================================

def _write_csv(filename, rows):
    if not rows:
        return
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Saved {filepath}")


# ================================================================
# Main
# ================================================================

def main():
    ensure_dir()

    print("\n" + "#" * 60)
    print("# INTELLIGENT PIPELINE SCHEDULER — EVALUATION SUITE")
    print("#" * 60)

    experiment_1_correctness()
    experiment_2_search_comparison()
    experiment_3_learning()
    experiment_4_scalability()
    experiment_5_failure_recovery()
    experiment_6_baseline_comparison()

    print("\n" + "#" * 60)
    print("# ALL EXPERIMENTS COMPLETE")
    print(f"# Results saved to: {RESULTS_DIR}/")
    print("#" * 60)


if __name__ == "__main__":
    main()
