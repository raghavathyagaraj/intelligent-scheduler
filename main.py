"""
main.py — Entry point for the Intelligent Pipeline Scheduler.

Usage:
    python main.py --dag simple
    python main.py --dag medium --mode failure
    python main.py --dag complex --mode stochastic --iterations 10
    python main.py --dag medium --mode failure --compare-baseline
"""

import argparse
import os
from src.task_dag import Pipeline
from src.scheduler_agent import SchedulerAgent
from src.visualizer import (
    plot_dag, plot_gantt, plot_metrics_comparison, plot_learning_curve
)


DAG_FILES = {
    "simple": "data/simple_dag.json",
    "medium": "data/medium_dag.json",
    "complex": "data/complex_dag.json",
}

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="Intelligent Pipeline Scheduler")
    parser.add_argument(
        "--dag", choices=["simple", "medium", "complex"], default="simple",
        help="Which sample DAG to use (default: simple)"
    )
    parser.add_argument(
        "--mode", choices=["normal", "stochastic", "failure", "spike"],
        default="normal",
        help="Simulation mode (default: normal)"
    )
    parser.add_argument(
        "--failure-rate", type=float, default=0.1,
        help="Probability of task failure in failure mode (default: 0.1)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1,
        help="Number of iterations for learning (default: 1)"
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Also run naive baseline for comparison"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip generating visualization PNGs"
    )
    args = parser.parse_args()

    # Load pipeline
    dag_path = DAG_FILES[args.dag]
    if not os.path.exists(dag_path):
        print(f"Error: DAG file not found: {dag_path}")
        return

    pipeline = Pipeline.load_json(dag_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("INTELLIGENT PIPELINE SCHEDULER")
    print("=" * 60)
    print()
    print(pipeline.summary())
    print()

    # Create agent
    agent = SchedulerAgent(pipeline)

    # Run iterations
    if args.iterations > 1:
        print(f"Running {args.iterations} iterations with learning...\n")
        results = agent.run_iterations(
            n=args.iterations,
            mode=args.mode,
            failure_rate=args.failure_rate,
            base_seed=args.seed
        )
        for r in results:
            r.print_log()
    else:
        result = agent.run(
            mode=args.mode,
            failure_rate=args.failure_rate,
            seed=args.seed
        )
        result.print_log()
        results = [result]

    # Compare with baseline
    if args.compare_baseline:
        print("\n" + "=" * 60)
        print("RUNNING NAIVE BASELINE FOR COMPARISON")
        print("=" * 60 + "\n")

        baseline_agent = SchedulerAgent(Pipeline.load_json(dag_path))
        baseline_result = baseline_agent.run_naive_baseline(
            mode=args.mode,
            failure_rate=args.failure_rate,
            seed=args.seed
        )
        baseline_result.print_log()

        if not args.no_plots and results[-1].metrics and baseline_result.metrics:
            plot_metrics_comparison(
                results[-1].metrics,
                baseline_result.metrics,
                output_path=os.path.join(RESULTS_DIR, "comparison.png")
            )
            print(f"\nComparison chart saved to {RESULTS_DIR}/comparison.png")

    # Generate visualizations
    if not args.no_plots:
        # DAG
        plot_dag(pipeline,
                 output_path=os.path.join(RESULTS_DIR, "dag.png"),
                 title=f"Pipeline: {pipeline.name}")
        print(f"DAG diagram saved to {RESULTS_DIR}/dag.png")

        # Gantt (from last run)
        last_result = results[-1]
        if last_result.schedule:
            plot_gantt(last_result.schedule, pipeline,
                       output_path=os.path.join(RESULTS_DIR, "gantt.png"),
                       title=f"Schedule: {pipeline.name} ({args.mode} mode)")
            print(f"Gantt chart saved to {RESULTS_DIR}/gantt.png")

        # Learning curve (if multiple iterations)
        if args.iterations > 1:
            plot_learning_curve(
                agent.get_estimator(),
                output_path=os.path.join(RESULTS_DIR, "learning_curve.png")
            )
            print(f"Learning curve saved to {RESULTS_DIR}/learning_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
