"""Graph wiring for the Planner/Executor multi-agent workflow.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes import executor_node, planner_node, supervisor_node
from state import AgentState


def build_planner_executor_graph():
	"""Build and compile the Planner/Executor LangGraph state machine.

	Nodes
	-----
	- "Supervisor": router node using ``next_step`` to decide what happens.
	- "Planner":   specialised agent that creates a plan from ``user_goal``.
	- "Executor":  specialised agent that iterates over the plan steps.

	Edges
	-----
	- START → "Supervisor".
	- "Supervisor" → (conditional via ``next_step``):
		- "Planner"   → "Planner" node.
		- "Executor"  → "Executor" node.
		- "FINISH"    → ``END`` terminal state.
	- "Planner"  → "Supervisor" (so that we immediately continue after planning).
	- "Executor" → "Supervisor" (loop until all steps are executed).
	"""

	workflow: StateGraph[AgentState] = StateGraph(AgentState)

	# Register nodes.
	workflow.add_node("Supervisor", supervisor_node)
	workflow.add_node("Planner", planner_node)
	workflow.add_node("Executor", executor_node)

	# START edge.
	workflow.add_edge(START, "Supervisor")

	# Conditional routing based on the ``next_step`` field, similar to the
	# reference Dev/QA example.
	workflow.add_conditional_edges(
		"Supervisor",
		lambda x: x["next_step"],
		{"Planner": "Planner", "Executor": "Executor", "FINISH": END},
	)

	# Loop back to Supervisor after each specialised agent finishes.
	workflow.add_edge("Planner", "Supervisor")
	workflow.add_edge("Executor", "Supervisor")

	memory = MemorySaver()
	return workflow.compile(checkpointer=memory)


# Expose a default compiled graph for convenience.
graph = build_planner_executor_graph()

