"""Utility tools for the Planner/Executor example.
"""

from langchain_core.tools import tool


@tool
def summarize_step(step: str) -> str:
	"""Return a short, human-readable summary of a plan step.
	"""

	return f"Plan step summary: {step.strip()}"

