# API Documentation — Major Classes and Functions

## task_dag.py

### Class: Task
Represents a single unit of work in a data pipeline.
```python
Task(task_id: str, name: str, duration_estimate: float,
     cpu_required: int = 2, memory_required: float = 4.0,
     priority: int = 5, sla_deadline: Optional[float] = None)
```
- `task_id` — Unique identifier (e.g., "extract_orders")
- `duration_estimate` — Estimated runtime in minutes
- `cpu_required` — CPU cores needed
- `priority` — 1 (lowest) to 10 (critical)
- `sla_deadline` — Minutes from midnight (360 = 6:00 AM), None = no deadline

### Class: Resource
Represents a compute server/cluster.
```python
Resource(resource_id: str, name: str,
         cpu_capacity: int = 8, memory_capacity: float = 32.0)
```

### Class: Schedule
A collection of ScheduleEntry objects.
```python
schedule.add_entry(ScheduleEntry(task_id, resource_id, start_time, end_time))
schedule.get_makespan() → float              # Total time from first start to last end
schedule.get_sla_adherence_rate(tasks) → float   # % of SLA-bound tasks meeting deadline
schedule.check_sla_compliance(tasks) → dict      # {task_id: True/False}
schedule.get_entries_for_resource(resource_id) → list
```

### Class: Pipeline
A DAG of tasks with dependency edges.
```python
pipeline = Pipeline.load_json("data/medium_dag.json")
pipeline.add_task(task)
pipeline.add_dependency("upstream_id", "downstream_id")
pipeline.get_dependencies(task_id) → list[str]       # Upstream tasks
pipeline.get_dependents(task_id) → list[str]         # Downstream tasks
pipeline.get_topological_order() → list[str]         # Valid execution order
pipeline.get_ready_tasks() → list[str]               # Tasks with all deps completed
pipeline.get_critical_path() → (path, duration)      # Longest path through DAG
pipeline.validate() → list[str]                      # Warnings/errors
pipeline.summary() → str                             # Human-readable overview
pipeline.save_json(filepath)
```

---

## knowledge_base.py

### Class: KnowledgeBase
First-order logic knowledge base with inference.
```python
kb = KnowledgeBase()
kb.load_from_pipeline(pipeline)                # Populate facts from pipeline
kb.register_default_rules()                    # Register 8 inference rules
kb.forward_chain() → int                      # Returns number of new facts derived

# Query facts
kb.query(predicate, subject) → list[Any]
kb.query_single(predicate, subject) → Any
kb.has_fact(predicate, subject, value) → bool

# Update
kb.update_task_status(task_id, "completed")
kb.add_fact(predicate, subject, value)

# High-level queries (after forward chaining)
kb.get_ready_tasks() → list[str]
kb.get_blocked_tasks() → list[tuple[str, str]]     # (task, blocker)
kb.get_sla_risks() → list[dict]                    # {task_id, deadline, slack}
kb.get_urgent_tasks() → list[str]
kb.get_retry_recommendations() → list[str]

# Backward chaining
kb.backward_chain(predicate, subject, value) → (proven: bool, explanation: list[str])

# Debugging
kb.get_inference_log() → list[str]            # Rule firing trace
kb.get_derived_facts() → list[Fact]           # Only facts from inference
kb.summary() → str
```

### 8 Inference Rules
1. `ready_to_run` — Task pending + all deps completed
2. `blocked_by_failure` — Upstream task failed
3. `sla_at_risk` — Cumulative time exceeds deadline
4. `resource_insufficient` — Task needs more CPU than any resource
5. `urgent_task` — Priority >= 8 and pending
6. `all_deps_met` — All dependencies completed
7. `cascading_sla_risk` — Upstream SLA risk + downstream has SLA
8. `recommend_retry` — Failed task should be retried

---

## csp_solver.py

### Class: CSPSolver
Constraint satisfaction solver for scheduling.
```python
solver = CSPSolver(pipeline, time_horizon=480.0, time_step=15.0, start_time=0.0)
result = solver.solve(use_ac3=True) → CSPResult
result = solver.solve_without_fc() → CSPResult   # Plain backtracking (for experiments)

# CSPResult fields
result.success → bool
result.schedule → Schedule
result.nodes_explored → int
result.backtracks → int
result.message → str

# Validate independently
CSPSolver.validate_schedule(schedule, pipeline) → list[str]   # Violation messages
```

### Constraint Types
1. Dependency: start_time(B) >= end_time(A) for every edge A→B
2. CPU capacity: concurrent tasks on same resource ≤ CPU capacity
3. Memory capacity: concurrent tasks on same resource ≤ memory capacity
4. SLA: end_time(task) ≤ sla_deadline

### Optimizations
- MRV: Pick variable with smallest domain first
- LCV: Try value that eliminates fewest neighbor options
- Forward checking: Prune domains after each assignment
- AC-3: Arc consistency preprocessing

---

## search_planner.py

### Class: SearchPlanner
A* and greedy best-first search for failure recovery.
```python
planner = SearchPlanner(pipeline, current_time=60.0, max_nodes=5000)
result = planner.search_astar(initial_state) → SearchResult
result = planner.search_greedy(initial_state) → SearchResult
result = planner.replan_from_failure(
    failed_task_ids=["task_b"],
    completed_task_ids=["task_a"],
    current_time=60.0
) → SearchResult

# SearchResult fields
result.success → bool
result.actions → list[PlannerAction]
result.schedule → Schedule
result.nodes_explored → int
result.sla_adherence → float
result.total_cost → float
```

### Actions and Costs
- schedule: cost 1.0 — assign ready task to resource
- retry: cost 5.0 — re-run failed task
- skip: cost 50.0 — accept failure, cascade to dependents
- wait: cost 0.5 — advance time by one step

### Heuristic
h(state) = remaining_work / num_resources + 100.0 × sla_violations
Admissible: guarantees A* optimality.

---

## learning.py

### Class: RuntimeEstimator
EWMA-based runtime estimation.
```python
estimator = RuntimeEstimator(alpha=0.3, method="ewma")
estimator.set_initial_estimates_from_pipeline(pipeline)

# Record and learn
record = estimator.record_runtime(task_id, actual_duration, iteration) → RuntimeRecord

# Get estimates
estimator.get_estimate(task_id) → float          # Learned estimate
estimator.get_static_estimate(task_id) → float   # Original (never changes)

# Error metrics
estimator.get_absolute_error(task_id) → float
estimator.get_global_mae() → float
estimator.get_error_over_iterations(task_id) → list[(iteration, error)]
estimator.get_improvement_over_static(task_id) → float  # % improvement

# EWMA formula: estimate = alpha * actual + (1-alpha) * old_estimate
```

---

## scheduler_agent.py

### Class: SchedulerAgent
Central orchestrator wiring all AI components.
```python
agent = SchedulerAgent(pipeline, time_horizon=480, time_step=15, ewma_alpha=0.3)

# Single run
result = agent.run(mode="normal", failure_rate=0.1, seed=42) → AgentRunResult

# Multi-iteration with learning
results = agent.run_iterations(n=10, mode="stochastic", base_seed=42) → list[AgentRunResult]

# Naive baseline for comparison
result = agent.run_naive_baseline(mode="normal", seed=42) → AgentRunResult

# AgentRunResult fields
result.schedule → Schedule
result.sim_result → SimResult
result.metrics → AgentMetrics  # makespan, sla_adherence, tasks_completed, etc.
result.log → list[str]        # Full execution trace
result.print_log()            # Print to terminal
```

---

## simulator.py

### Class: Simulator
Simulates pipeline execution.
```python
sim = Simulator(pipeline, mode="failure", failure_rate=0.1, seed=42)
result = sim.run(schedule) → SimResult

# SimResult fields
result.completed_tasks → list[str]
result.failed_tasks → list[str]
result.actual_durations → dict[str, float]
result.sla_violations → list[str]
result.events → list[SimEvent]       # Full event log
```

### Modes
- `normal` — exact estimated durations
- `stochastic` — runtimes vary ±30%
- `failure` — random task failures
- `spike` — one random task takes 4x longer
