"""Agent node implementations for the Planner/Executor multi-agent workflow.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state import AgentState


def _get_llm() -> ChatOllama:
	"""Return a shared ChatOllama instance using a local mistral model.

	This assumes an Ollama server is running locally and exposes the
	``mistral:latest`` model. No API keys are required for this setup.
	"""

	return ChatOllama(model="mistral:latest", temperature=0.0)


def _parse_plan(text: str) -> List[str]:
	"""Parse an LLM-generated plan into a list of clean step strings."""

	steps: List[str] = []
	for raw_line in text.splitlines():
		line = raw_line.strip()
		if not line:
			continue
		for prefix in ("- ", "* ", "• "):
			if line.startswith(prefix):
				line = line[len(prefix) :].strip()
		if ". " in line and line.split(". ", 1)[0].isdigit():
			line = line.split(". ", 1)[1].strip()
		steps.append(line)
	return steps


def supervisor_node(state: AgentState) -> Dict:
	"""Router node that decides which specialised agent should act next.

	Routing policy (using the same "next_step" pattern as in the reference):

	- If this is the first turn (no named message yet), route to ``Planner``.
	- After ``Planner`` runs, route to ``Executor``.
	- After ``Executor`` runs:
	  - If there are remaining steps (``current_step_index < len(plan)``),
		route back to ``Executor`` to continue the loop.
	  - Otherwise, route to ``FINISH`` to end the workflow.
	"""

	messages = state.get("messages", [])
	last_message = messages[-1] if messages else None
	last_name = getattr(last_message, "name", None) if last_message else None

	plan = state.get("plan", [])
	current_index = state.get("current_step_index", 0)

	if not last_name:
		next_step = "Planner"
	elif last_name == "Planner":
		next_step = "Executor"
	elif last_name == "Executor":
		if current_index < len(plan):
			next_step = "Executor"
		else:
			next_step = "FINISH"
	else:
		next_step = "FINISH"

	print(
		f"[Supervisor] last={last_name!r}, plan_len={len(plan)}, "
		f"current_step_index={current_index} -> next_step={next_step}"
	)

	return {"next_step": next_step, "status": "supervising"}


def planner_node(state: AgentState) -> Dict:
	"""Planner agent that turns a high-level goal into an ordered list of steps."""

	llm = _get_llm()

	user_goal = state.get("user_goal")
	if not user_goal:
		# Fallback: derive it from the first user message if possible.
		messages = state.get("messages", [])
		if messages:
			user_goal = str(messages[0].content)
		else:
			user_goal = "Set up a basic ML experiment tracking workflow."

	print(f"[Planner] Generating plan for goal: {user_goal}")

	messages = [
		SystemMessage(
			content=(
				"You are an MLOps planning assistant. Given a high-level "
				"goal, produce a short ordered list of concrete steps to "
				"achieve it. Focus on practical, implementation-oriented "
				"actions suitable for a small lab environment."
			)
		),
		HumanMessage(
			content=(
				"User goal: "
				+ user_goal
				+ "\n\nRespond with 4-8 steps as a numbered list."
			)
		),
	]

	response = llm.invoke(messages)
	raw_plan = response.content if isinstance(response.content, str) else str(response.content)

	steps = _parse_plan(raw_plan)
	if not steps:
		steps = [
			"Clarify the goal and constraints.",
			"Draft a minimal implementation plan.",
			"Implement the plan in a step-by-step fashion.",
			"Verify that the goal has been achieved.",
		]

	print("[Planner] Planned steps:")
	for i, step in enumerate(steps, start=1):
		print(f"  {i}. {step}")

	return {
		"messages": [HumanMessage(content=raw_plan, name="Planner")],
		"plan": steps,
		"current_step_index": 0,
		"status": "planning_done",
		"user_goal": user_goal,
	}


def executor_node(state: AgentState) -> Dict:
	"""Executor agent that iterates over plan steps and simulates execution.

	For each step, it asks the LLM to describe how to carry it out in practice
	and appends the result to ``execution_log``.
	"""

	llm = _get_llm()

	plan = state.get("plan", [])
	current_index = state.get("current_step_index", 0)

	if not plan or current_index >= len(plan):
		print("[Executor] No remaining steps to execute.")
		return {"status": "nothing_to_execute"}

	step_text = plan[current_index]
	print(f"[Executor] Executing step {current_index + 1}/{len(plan)}: {step_text}")

	messages = [
		SystemMessage(
			content=(
				"You are an MLOps execution assistant. For the given plan "
				"step, describe briefly (3-5 sentences) how you would carry "
				"it out in practice, including any relevant tools, commands, "
				"or configuration considerations."
			)
		),
		HumanMessage(
			content=(
				"Overall user goal: "
				+ state.get("user_goal", "(unknown)")
				+ "\nCurrent step: "
				+ step_text
			)
		),
	]

	response = llm.invoke(messages)
	explanation = response.content if isinstance(response.content, str) else str(response.content)

	previous_log = list(state.get("execution_log", []))
	entry = f"Step {current_index + 1}: {step_text}\n{explanation}"
	previous_log.append(entry)

	print("[Executor] Appended execution log entry.")

	return {
		"messages": [HumanMessage(content=explanation, name="Executor")],
		"execution_log": previous_log,
		"current_step_index": current_index + 1,
		"status": "executing",
	}

