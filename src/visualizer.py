"""
visualizer.py — Visualization for the Intelligent Pipeline Scheduler.

Generates static PNG charts:
- DAG diagram: task dependency graph with status colors.
- Gantt chart: schedule as horizontal bars, colored by resource.
- Metrics dashboard: bar charts comparing agent vs baseline.
- Learning curve: estimation error decreasing over iterations.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import networkx as nx
from typing import Optional
from src.task_dag import Pipeline, Schedule, TaskStatus
from src.scheduler_agent import AgentRunResult, AgentMetrics
from src.learning import RuntimeEstimator


# Color palette
COLORS = {
    "completed": "#4CAF50",
    "running": "#FFC107",
    "failed": "#F44336",
    "pending": "#9E9E9E",
    "skipped": "#FF9800",
    "ready": "#2196F3",
}

RESOURCE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def plot_dag(pipeline: Pipeline,
             output_path: str = "results/dag.png",
             title: Optional[str] = None,
             figsize: tuple = (12, 8)):
    """
    Plot the pipeline DAG with task status colors.

    Args:
        pipeline: The pipeline to visualize.
        output_path: Where to save the PNG.
        title: Chart title (defaults to pipeline name).
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    G = pipeline.dag

    # Layout
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot") if _has_graphviz() else \
        _hierarchical_layout(G)

    # Node colors based on status
    node_colors = []
    for node in G.nodes():
        status = pipeline.tasks[node].status.value
        node_colors.append(COLORS.get(status, COLORS["pending"]))

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#CCCCCC",
                           arrows=True, arrowsize=20, width=1.5,
                           connectionstyle="arc3,rad=0.1")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=2000, edgecolors="#333333", linewidths=1.5)

    # Labels: task_id + duration
    labels = {}
    for task_id, task in pipeline.tasks.items():
        labels[task_id] = f"{task_id}\n({task.duration_estimate:.0f}m)"
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=8,
                            font_weight="bold")

    # Legend
    legend_patches = [
        mpatches.Patch(color=COLORS[s], label=s.capitalize())
        for s in ["completed", "running", "failed", "pending"]
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    ax.set_title(title or f"Pipeline DAG: {pipeline.name}", fontsize=14,
                 fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    _save(fig, output_path)


def plot_gantt(schedule: Schedule, pipeline: Pipeline,
               output_path: str = "results/gantt.png",
               title: Optional[str] = None,
               figsize: tuple = (14, 8)):
    """
    Plot a Gantt chart of the schedule.

    Args:
        schedule: The schedule to visualize.
        pipeline: The pipeline (for task names and SLA info).
        output_path: Where to save the PNG.
        title: Chart title.
        figsize: Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Map resources to colors
    resources = list({e.resource_id for e in schedule.entries})
    resources.sort()
    res_color_map = {r: RESOURCE_COLORS[i % len(RESOURCE_COLORS)]
                     for i, r in enumerate(resources)}

    # Sort tasks by start time
    sorted_entries = sorted(schedule.entries, key=lambda e: e.start_time)
    task_ids = [e.task_id for e in sorted_entries]

    for i, entry in enumerate(sorted_entries):
        color = res_color_map[entry.resource_id]
        ax.barh(i, entry.duration, left=entry.start_time,
                color=color, edgecolor="#333333", linewidth=0.5,
                height=0.6, alpha=0.85)

        # Task label inside bar
        mid = entry.start_time + entry.duration / 2
        ax.text(mid, i, f"{entry.task_id}", ha="center", va="center",
                fontsize=7, fontweight="bold", color="white")

    # SLA deadline lines
    for entry in sorted_entries:
        task = pipeline.tasks.get(entry.task_id)
        if task and task.sla_deadline is not None:
            idx = task_ids.index(entry.task_id)
            ax.axvline(x=task.sla_deadline, color="red", linestyle="--",
                       alpha=0.5, linewidth=1)

    ax.set_yticks(range(len(task_ids)))
    ax.set_yticklabels(task_ids, fontsize=9)
    ax.set_xlabel("Time (minutes from midnight)", fontsize=11)
    ax.set_title(title or "Schedule — Gantt Chart", fontsize=14,
                 fontweight="bold")

    # Legend for resources
    legend_patches = [
        mpatches.Patch(color=res_color_map[r], label=r)
        for r in resources
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _save(fig, output_path)


def plot_metrics_comparison(agent_metrics: AgentMetrics,
                             baseline_metrics: AgentMetrics,
                             output_path: str = "results/comparison.png",
                             figsize: tuple = (12, 6)):
    """
    Bar chart comparing agent vs naive baseline on key metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    categories = ["Intelligent Agent", "Naive Baseline"]
    bar_colors = ["#2196F3", "#9E9E9E"]

    # Makespan
    values = [agent_metrics.makespan, baseline_metrics.makespan]
    axes[0].bar(categories, values, color=bar_colors, edgecolor="#333", width=0.5)
    axes[0].set_title("Makespan (minutes)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Minutes")
    for i, v in enumerate(values):
        axes[0].text(i, v + 2, f"{v:.0f}", ha="center", fontsize=10,
                     fontweight="bold")

    # SLA Adherence
    values = [agent_metrics.sla_adherence, baseline_metrics.sla_adherence]
    axes[1].bar(categories, values, color=bar_colors, edgecolor="#333", width=0.5)
    axes[1].set_title("SLA Adherence (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Percentage")
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(values):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=10,
                     fontweight="bold")

    # Tasks Completed
    values = [agent_metrics.tasks_completed, baseline_metrics.tasks_completed]
    axes[2].bar(categories, values, color=bar_colors, edgecolor="#333", width=0.5)
    axes[2].set_title("Tasks Completed", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Count")
    for i, v in enumerate(values):
        axes[2].text(i, v + 0.2, f"{v}", ha="center", fontsize=10,
                     fontweight="bold")

    plt.suptitle("Intelligent Agent vs Naive Baseline", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_learning_curve(estimator: RuntimeEstimator,
                         task_ids: Optional[list[str]] = None,
                         output_path: str = "results/learning_curve.png",
                         figsize: tuple = (10, 6)):
    """
    Plot estimation error over iterations showing learning improvement.

    Args:
        estimator: The RuntimeEstimator with history.
        task_ids: Specific tasks to plot (None = all with history).
        output_path: Where to save the PNG.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if task_ids is None:
        task_ids = list(estimator.get_all_history().keys())

    for task_id in task_ids:
        errors = estimator.get_error_over_iterations(task_id)
        if not errors:
            continue
        iterations = [e[0] for e in errors]
        abs_errors = [e[1] for e in errors]
        ax.plot(iterations, abs_errors, marker="o", markersize=4,
                label=task_id, linewidth=1.5)

    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Absolute Error (minutes)", fontsize=11)
    ax.set_title("Learning: Estimation Error Over Iterations", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, output_path)


def plot_search_comparison(astar_nodes: int, greedy_nodes: int,
                            astar_cost: float, greedy_cost: float,
                            output_path: str = "results/search_comparison.png",
                            figsize: tuple = (10, 5)):
    """
    Compare A* vs Greedy best-first on nodes explored and plan cost.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    categories = ["A*", "Greedy"]
    colors = ["#2196F3", "#FF9800"]

    # Nodes explored
    ax1.bar(categories, [astar_nodes, greedy_nodes], color=colors,
            edgecolor="#333", width=0.4)
    ax1.set_title("Nodes Explored", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Count")
    for i, v in enumerate([astar_nodes, greedy_nodes]):
        ax1.text(i, v + 1, str(v), ha="center", fontsize=10, fontweight="bold")

    # Plan cost
    ax2.bar(categories, [astar_cost, greedy_cost], color=colors,
            edgecolor="#333", width=0.4)
    ax2.set_title("Plan Cost", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cost")
    for i, v in enumerate([astar_cost, greedy_cost]):
        ax2.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10,
                 fontweight="bold")

    plt.suptitle("A* vs Greedy Best-First Search", fontsize=14,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, output_path)


def plot_scalability(task_counts: list[int],
                      csp_times: list[float],
                      search_times: list[float],
                      output_path: str = "results/scalability.png",
                      figsize: tuple = (10, 6)):
    """
    Plot CSP and search solve times vs DAG size.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(task_counts, csp_times, marker="s", label="CSP Solve Time",
            color="#2196F3", linewidth=2, markersize=8)
    ax.plot(task_counts, search_times, marker="o", label="Search Time",
            color="#FF9800", linewidth=2, markersize=8)

    ax.set_xlabel("Number of Tasks", fontsize=11)
    ax.set_ylabel("Time (seconds)", fontsize=11)
    ax.set_title("Scalability: Solve Time vs DAG Size", fontsize=14,
                 fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save(fig, output_path)


# ================================================================
# Helpers
# ================================================================

def _save(fig, path: str):
    """Save figure and create directory if needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _has_graphviz() -> bool:
    """Check if graphviz layout is available."""
    try:
        import pygraphviz
        return True
    except ImportError:
        return False


def _hierarchical_layout(G) -> dict:
    """Simple hierarchical layout for DAGs without graphviz."""
    # Use topological generations for Y position
    try:
        generations = list(nx.topological_generations(G))
    except Exception:
        return nx.spring_layout(G)

    pos = {}
    for y, gen in enumerate(generations):
        for x, node in enumerate(sorted(gen)):
            # Center each generation
            x_offset = (x - len(gen) / 2) * 2
            pos[node] = (x_offset, -y * 2)
    return pos
