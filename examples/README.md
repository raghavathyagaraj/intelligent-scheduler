# Example Input/Output Files

## example_input.json
A sample pipeline definition (copy of simple_dag.json).
5 tasks in a linear chain with 1 resource and 1 SLA deadline.

## example_output_schedule.json
The schedule produced by the agent when run on the input.
Contains task assignments (resource, start_time, end_time) and metrics.

## example_output_terminal.txt
Full terminal output from running:
```bash
python main.py --dag simple --mode normal
```
Shows all 5 agent steps: KB reasoning, CSP solving, simulation, re-planning check, and learning.

## How to reproduce
```bash
python main.py --dag simple --mode normal --no-plots
```
