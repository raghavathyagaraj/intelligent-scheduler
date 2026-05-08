# Intelligent Data Pipeline Scheduler

An AI agent that intelligently schedules data pipeline tasks across limited compute resources using search, knowledge representation, constraint satisfaction, and learning.

## Team Members
- Sanjana
- Raghava Thyagaraj
- Monisha
- Diogo Dcosta
- Rakshitha

## Course
Graduate AI — Final Project (Option D: Planning & Scheduling Systems)
Pace University — CS 627 Artificial Intelligence — May 2026

---

## Problem Description

Static pipeline schedulers (cron, Airflow) follow fixed timetables and cannot adapt when tasks fail, runtimes vary, or resources are contended. Our intelligent agent solves this by:

1. **Reasoning** about task dependencies, resource limits, and SLA deadlines using first-order logic (forward/backward chaining)
2. **Scheduling** optimally by formulating the problem as a Constraint Satisfaction Problem (backtracking + MRV + LCV + forward checking + AC-3)
3. **Recovering** from failures using A* search to find optimal re-planning strategies
4. **Learning** from actual runtimes using EWMA to improve future scheduling accuracy

---

## System Requirements

- **Python**: 3.10 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 4 GB RAM minimum
- **No cloud services, APIs, or external tools required**

## Dependencies

All dependencies are listed in `requirements.txt`:
- `networkx` — DAG representation and graph algorithms
- `matplotlib` — Visualization (Gantt charts, DAG plots, metrics)
- `numpy` — Numerical computation for learning module
- `pytest` — Automated testing framework

---

## Installation Instructions

```bash
# 1. Clone or unzip the project
cd intelligent-scheduler

# 2. Create a virtual environment
python3 -m venv venv

# 3. Activate it
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify — run all tests (expect 176 passed)
pytest tests/ -v
```

---

## Usage Instructions with Examples

### Basic: Schedule a pipeline
```bash
python main.py --dag simple
```

### Stochastic mode (runtimes vary ±30%)
```bash
python main.py --dag medium --mode stochastic
```

### Failure mode (random task crashes)
```bash
python main.py --dag complex --mode failure --failure-rate 0.2
```

### Learning over 10 iterations
```bash
python main.py --dag medium --mode stochastic --iterations 10
```

### Compare agent vs naive baseline
```bash
python main.py --dag medium --compare-baseline
```

### Run all 6 evaluation experiments
```bash
python evaluation/evaluate.py
```

---

## File Structure

```
intelligent-scheduler/
├── main.py                        # CLI entry point
├── requirements.txt               # Dependencies with versions
├── README.md                      # This file
│
├── src/                           # Source code (8 modules, ~3,700 LOC)
│   ├── task_dag.py                # Data model: Task, Resource, Pipeline, Schedule
│   ├── knowledge_base.py          # FOL knowledge base + inference engine
│   ├── csp_solver.py              # CSP solver + backtracking + MRV/LCV/FC/AC-3
│   ├── search_planner.py          # A* and greedy search for failure recovery
│   ├── learning.py                # EWMA runtime estimation
│   ├── scheduler_agent.py         # Agent orchestrator
│   ├── simulator.py               # Pipeline execution simulator (4 modes)
│   └── visualizer.py              # DAG, Gantt, metrics charts
│
├── data/                          # Sample pipelines (JSON)
│   ├── simple_dag.json            # 5 tasks, linear, 1 resource
│   ├── medium_dag.json            # 10 tasks, diamond, 2 resources
│   └── complex_dag.json           # 20 tasks, multi-branch, 3 resources, 4 SLAs
│
├── tests/                         # 176 automated tests
│   ├── test_task_dag.py           # 39 tests
│   ├── test_knowledge_base.py     # 46 tests
│   ├── test_csp_solver.py         # 24 tests
│   ├── test_search_planner.py     # 25 tests
│   ├── test_learning.py           # 25 tests
│   └── test_integration.py        # 17 tests
│
├── evaluation/                    # Experiments + results
│   ├── evaluate.py                # Runs all 6 experiments
│   └── results/                   # CSVs + PNG charts
│
├── examples/                      # Example input/output files
│   ├── example_input.json         # Sample pipeline
│   ├── example_output_schedule.json
│   └── example_output_terminal.txt
│
├── docs/                          # Diagrams + API docs
│   ├── architecture.png
│   ├── component_interaction.png
│   ├── data_flow.png
│   ├── agent_type.png
│   └── api_docs.md
│
├── report/Final_Report.pdf
└── demo/demo_video.mp4
```

---

## Running Tests

```bash
# All 176 tests
pytest tests/ -v

# Specific module
pytest tests/test_csp_solver.py -v

# By keyword
pytest tests/ -k "failure" -v
```

| Test File | Tests | What's Verified |
|---|---|---|
| test_task_dag.py | 39 | DAG operations, critical path, JSON I/O |
| test_knowledge_base.py | 46 | FOL rules, chaining, SLA risk detection |
| test_csp_solver.py | 24 | All constraints, MRV, FC, AC-3 |
| test_search_planner.py | 25 | A*, greedy, failure recovery |
| test_learning.py | 25 | EWMA convergence, error tracking |
| test_integration.py | 17 | Full agent cycle, baseline, CLI |

---

## AI Techniques

| Technique | Module | Chapter | Purpose |
|---|---|---|---|
| Knowledge Representation (FOL) | knowledge_base.py | Ch. 7-9 | Proactive risk detection |
| Constraint Satisfaction (CSP) | csp_solver.py | Ch. 5-6 | Valid, optimal schedules |
| Search & Planning (A*) | search_planner.py | Ch. 3-4 | Failure recovery |
| Learning (EWMA) | learning.py | — | Improve estimates over time |
