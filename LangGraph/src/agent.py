"""Interactive entrypoint for the Planner/Executor multi-agent workflow.
"""

from __future__ import annotations

from pprint import pprint

from langchain_core.messages import HumanMessage

from graph import graph
from state import AgentState


def run_once(user_goal: str) -> AgentState:
	"""Run the Planner/Executor graph once for the given user goal.

	Returns the final ``AgentState`` after the graph has reached ``END``.
	"""

	config = {"configurable": {"thread_id": "planner_executor_run_1"}}

	# Initial state resembles the style from the reference examples: a single
	# HumanMessage plus explicit state fields.
	initial_state: AgentState = {
		"messages": [HumanMessage(content=user_goal)],
		"user_goal": user_goal,
		"plan": [],
		"current_step_index": 0,
		"execution_log": [],
		"next_step": "",
		"status": "created",
	}

	final_state: AgentState | None = None

	# Stream values so you can see the Supervisor/Planner/Executor loop as it
	# unfolds, similar to the Dev/QA example.
	for s in graph.stream(initial_state, config=config, stream_mode="values"):  # type: ignore[assignment]
		final_state = s  # type: ignore[assignment]

		messages = s.get("messages", [])
		last_msg = messages[-1] if messages else None
		last_name = getattr(last_msg, "name", None) if last_msg else None

		if last_name == "Planner":
			print("\n[Planner Output] Planned steps updated.")
		elif last_name == "Executor":
			idx = s.get("current_step_index", 0)
			print(f"\n[Executor Output] Completed step index {idx}.")

	if final_state is None:
		raise RuntimeError("Graph did not produce a final state.")

	return final_state


if __name__ == "__main__":
	print("--- Planner/Executor Multi-Agent Workflow ---")
	user_input = input("Goal: ")

	state = run_once(user_input)

	print("\n--- Final Plan ---")
	for i, step in enumerate(state.get("plan", []), start=1):
		print(f"  {i}. {step}")

	print("\n--- Execution Log ---")
	for entry in state.get("execution_log", []):
		print("-" * 40)
		print(entry)

	print("\n--- Final State (raw) ---")
	pprint(state)

