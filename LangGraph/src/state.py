"""Shared state definition for the Planner/Executor LangGraph workflow.
"""

from typing import Annotated, List, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
	"""State passed between nodes in the Planner/Executor workflow.

	Fields
	------
	messages:
		Conversation history, managed with ``add_messages`` so that LangGraph
		automatically appends new messages returned by nodes.
	user_goal:
		High-level user objective (e.g. "Set up a basic ML experiment tracking workflow").
	plan:
		Ordered list of textual steps produced by the Planner agent.
	current_step_index:
		Zero-based index of the plan step that the Executor should handle next.
	execution_log:
		Human-readable log entries describing how each step was (virtually)
		executed by the Executor agent.
	next_step:
		Routing hint set by the Supervisor node. One of: "Planner",
		"Executor" or "FINISH".
	status:
		High-level status string (e.g. "created", "planning_done",
		"executing", "done").
	"""

	messages: Annotated[List, add_messages]
	user_goal: str
	plan: List[str]
	current_step_index: int
	execution_log: List[str]
	next_step: str
	status: str

