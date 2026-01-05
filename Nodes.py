"""
Node implementations for the Financial Analysis Agent graph.
Contains all LangGraph nodes:  planner, researcher, analyst, reviewer, writer.
"""

import json
import logging
from typing import Any, Dict

from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from AnalysisTools import analysis_tools
# Import from local modules
from Constants import (ALL_TOOL_NAMES, ANALYSIS_TOOL_NAMES, MAX_CALLS_PER_TOOL,
                       MAX_ITERATIONS, RESEARCH_TOOL_NAMES)
from ResearchTools import research_tools
from Schemas import AgentState, PlanOutput, ReviewDecision
from SystemPrompts import (ANALYST_PROMPT, PLANNER_SYSTEM_PROMPT,
                           RESEARCHER_PROMPT, REVIEWER_PROMPT, WRITER_PROMPT)
from Utilities import (build_context, can_call_tool, check_data_status,
                       check_limits, create_default_plan,
                       create_fallback_review_decision,
                       extract_data_from_messages, get_available_tools_info,
                       get_data_summary, get_tool_status_for_reviewer,
                       safe_json_serialize)

# Setup logging
logger = logging.getLogger(__name__)


###############################################################################
# LLM Configuration
###############################################################################

import os

from dotenv import load_dotenv

load_dotenv()

AUTH_CODE = os.getenv("AUTH_CODE")
BASE_URL = os.getenv("BASE_URL")


def get_llm(
    temperature: float = 0,
    reasoning_effort: str = "high",
) -> ChatOpenAI:
    """Create a configured ChatOpenAI instance."""
    kwargs = {
        "temperature": temperature,
        "model": "gpt-oss-120b",
    }
    if BASE_URL:
        kwargs["base_url"] = BASE_URL
    if AUTH_CODE:
        kwargs["api_key"] = AUTH_CODE

    return ChatOpenAI(**kwargs)


# Initialize LLMs
_base_planner_llm = get_llm(temperature=0.1)
_base_reviewer_llm = get_llm(temperature=0)
_base_researcher_llm = get_llm(temperature=0)
_base_analyst_llm = get_llm(temperature=0)
_base_writer_llm = get_llm(temperature=0.2)

# LLMs with structured output
planner_llm_structured = _base_planner_llm.with_structured_output(PlanOutput)
reviewer_llm_structured = _base_reviewer_llm.with_structured_output(ReviewDecision)

# LLMs with tool binding
researcher_llm = _base_researcher_llm.bind_tools(research_tools)
analyst_llm = _base_analyst_llm.bind_tools(analysis_tools)
writer_llm = _base_writer_llm


###############################################################################
# Helper Functions
###############################################################################


def _print_node_header(node_name: str) -> None:
    """Print a formatted node header."""
    print("")
    print("=" * 70)
    print(f"{node_name} NODE")
    print("=" * 70)


def _print_node_footer() -> None:
    """Print a formatted node footer."""
    print("-" * 70)


def _log_tool_calls(response: AIMessage) -> None:
    """Log tool calls from an AI response."""
    if response.tool_calls:
        print(f"  Tool calls: {len(response.tool_calls)}")
        for tc in response.tool_calls:
            print(f"    - {tc['name']}")


###############################################################################
# Planner Node
###############################################################################


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner node using structured output for deterministic plan creation.
    """
    _print_node_header("PLANNER")

    user_request = state["user_request"]

    try:
        plan_output: PlanOutput = planner_llm_structured.invoke(
            [
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Create a detailed analysis plan for: {user_request}"
                ),
            ]
        )

        plan = plan_output.model_dump()
        logger.info(f"Structured output successful: {plan. get('ticker')}")
        print("  ✓ Structured output successful")

    except (ValidationError, AttributeError, Exception) as e:
        logger.warning(f"Structured output failed, using fallback: {e}")
        print(f"  ⚠ Structured output failed: {e}")
        print("  Using fallback plan...")

        plan_output = create_default_plan()
        plan = plan_output.model_dump()

    print(f"  Ticker: {plan.get('ticker')}")
    print(f"  Years:  {plan.get('years')}")
    print(f"  Data Requirements: {plan.get('data_requirements', [])}")
    _print_node_footer()

    return {
        "messages": [AIMessage(content=f"Plan created: {safe_json_serialize(plan)}")],
        "plan": plan,
        "current_phase": "researching",
        "current_node": "researcher",
    }


###############################################################################
# Researcher Node
###############################################################################


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcher node that gathers financial data using research tools.
    """
    _print_node_header("RESEARCHER")

    limits = check_limits(state)
    data_status = check_data_status(state)
    tool_status = get_available_tools_info(state, RESEARCH_TOOL_NAMES)

    print(f"  Iteration:  {limits['iteration_count']}/{MAX_ITERATIONS}")

    # Get plan details
    plan = state.get("plan", {})
    ticker = plan.get("ticker", "NVDA")
    years = plan.get("years", [2023, 2024, 2025])

    # Build context
    context = build_context(state)
    pending = state.get("pending_action")

    # Determine specific instructions
    specific_instruction = _build_researcher_instruction(
        pending=pending, data_status=data_status, ticker=ticker, years=years
    )

    if pending:
        print(f"  Pending action: {pending}")

    # Build data status string for prompt
    data_status_str = f"""
    - Financial Data:  {"Collected" if data_status["has_financials"] else "MISSING"}
    - Price Data: {"Collected" if data_status["has_price"] else "MISSING"}
    - News Data: {"Collected" if data_status["has_news"] else "MISSING"}
    - Analysis Results: {data_status["analysis_count"]} completed
    """

    prompt = RESEARCHER_PROMPT.format(
        tool_status=tool_status, data_status=data_status_str
    )

    response = researcher_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"{context}{specific_instruction}"),
        ]
    )

    _log_tool_calls(response)
    _print_node_footer()

    return {
        "messages": [response],
        "current_phase": "researching",
        "current_node": "researcher",
        "pending_action": None,
    }


def _build_researcher_instruction(
    pending: str, data_status: Dict[str, Any], ticker: str, years: list
) -> str:
    """Build specific instruction for the researcher based on current state."""

    if pending:
        if "fetch_general_news" in pending and data_status["can_fetch_news"]:
            return f'\nCRITICAL:  Call fetch_general_news(ticker="{ticker}") NOW.'
        elif "fetch_price_history" in pending and data_status["can_fetch_price"]:
            return f'\nCRITICAL:  Call fetch_price_history(ticker="{ticker}", years={years}) NOW.'
        elif "fetch_financials" in pending and data_status["can_fetch_financials"]:
            return f'\nCRITICAL: Call fetch_financials(ticker_symbol="{ticker}") NOW.'
        else:
            return "\nRequested tool is exhausted.  Proceed with available data."

    # No pending action - check for missing data
    missing_calls = []

    if not data_status["has_financials"] and data_status["can_fetch_financials"]:
        missing_calls.append(f"fetch_financials(ticker_symbol='{ticker}')")
    if not data_status["has_price"] and data_status["can_fetch_price"]:
        missing_calls.append(f"fetch_price_history(ticker='{ticker}', years={years})")
    if not data_status["has_news"] and data_status["can_fetch_news"]:
        missing_calls.append(f"fetch_general_news(ticker='{ticker}')")

    if missing_calls:
        print(f"  Missing data, requesting:  {len(missing_calls)} tools")
        return "\nCall these tools to gather required data:\n" + "\n".join(
            [f"- {call}" for call in missing_calls]
        )

    return "\nAll available data has been collected."


###############################################################################
# Research Tools Node
###############################################################################


def research_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute research tools with rate limiting and error handling.
    """
    print("\n  RESEARCH TOOLS EXECUTION")
    print("  " + "-" * 40)

    messages = state.get("messages", [])

    if not messages or not isinstance(messages[-1], AIMessage):
        print("    No AI message found")
        return {"messages": []}

    last_message = messages[-1]
    if not last_message.tool_calls:
        print("    No tool calls in message")
        return {"messages": []}

    tool_call_counts = dict(state.get("tool_call_counts", {}))
    new_messages = []
    executed_count = 0
    blocked_count = 0

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_id = tc["id"]
        current_count = tool_call_counts.get(tool_name, 0)

        if current_count >= MAX_CALLS_PER_TOOL:
            print(
                f"    ✗ BLOCKED: {tool_name} (exhausted:  {current_count}/{MAX_CALLS_PER_TOOL})"
            )
            blocked_count += 1

            new_messages.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "success": False,
                            "error": f"TOOL EXHAUSTED: {tool_name} has been called {MAX_CALLS_PER_TOOL} times.",
                        }
                    ),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )
        else:
            print(f"    ✓ EXECUTING: {tool_name}")
            executed_count += 1

            try:
                selected_tool = next(
                    (t for t in research_tools if t.name == tool_name), None
                )

                if selected_tool:
                    tool_output = selected_tool.invoke(tc["args"])
                    new_messages.append(
                        ToolMessage(
                            content=safe_json_serialize(tool_output),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                else:
                    new_messages.append(
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "success": False,
                                    "error": f"Tool '{tool_name}' not found",
                                }
                            ),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
            except Exception as e:
                logger.error(f"Error executing {tool_name}: {e}")
                new_messages.append(
                    ToolMessage(
                        content=json.dumps({"success": False, "error": str(e)}),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

    print(f"    Summary: {executed_count} executed, {blocked_count} blocked")
    print("  " + "-" * 40)

    extracted = extract_data_from_messages(new_messages, state)
    new_total = state.get("total_tool_calls", 0) + len(last_message.tool_calls)

    return {
        "messages": new_messages,
        "total_tool_calls": new_total,
        "financial_data": extracted["financial_data"],
        "price_data": extracted["price_data"],
        "news_data": extracted["news_data"],
        "analysis_results": extracted["analysis_results"],
        "tool_call_counts": extracted["tool_call_counts"],
    }


###############################################################################
# Analyst Node
###############################################################################


def analyst_node(state: AgentState) -> Dict[str, Any]:
    """
    Analyst node that performs financial analysis using analysis tools.
    """
    _print_node_header("ANALYST")

    limits = check_limits(state)
    tool_status = get_available_tools_info(state, ANALYSIS_TOOL_NAMES)

    print(f"  Iteration: {limits['iteration_count']}/{MAX_ITERATIONS}")

    financial_data = state.get("financial_data", {})
    fin_data_str = ""

    if financial_data.get("data"):
        fin_data_str = f"\n\nUse this financial data for analysis:\n{safe_json_serialize(financial_data['data'])}"
        print(f"  Financial data available: {len(financial_data['data'])} years")
    else:
        print("  ⚠ No financial data available for analysis")

    prompt = ANALYST_PROMPT.format(tool_status=tool_status)

    response = analyst_llm.invoke(
        [
            SystemMessage(content=prompt),
            HumanMessage(content=f"{build_context(state)}{fin_data_str}"),
        ]
    )

    _log_tool_calls(response)
    _print_node_footer()

    return {
        "messages": [response],
        "current_phase": "analyzing",
        "current_node": "analyst",
    }


###############################################################################
# Analysis Tools Node
###############################################################################


def analysis_tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute analysis tools with rate limiting.
    """
    print("\n  ANALYSIS TOOLS EXECUTION")
    print("  " + "-" * 40)

    messages = state.get("messages", [])

    if not messages or not isinstance(messages[-1], AIMessage):
        print("    No AI message found")
        return {"messages": []}

    last_message = messages[-1]
    if not last_message.tool_calls:
        print("    No tool calls")
        return {"messages": []}

    tool_call_counts = dict(state.get("tool_call_counts", {}))
    new_messages = []
    executed_count = 0
    blocked_count = 0

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_id = tc["id"]
        current_count = tool_call_counts.get(tool_name, 0)

        if current_count >= MAX_CALLS_PER_TOOL:
            print(f"    ✗ BLOCKED:  {tool_name} (exhausted)")
            blocked_count += 1

            new_messages.append(
                ToolMessage(
                    content=json.dumps(
                        {"success": False, "error": f"TOOL EXHAUSTED: {tool_name}"}
                    ),
                    tool_call_id=tool_id,
                    name=tool_name,
                )
            )
        else:
            print(f"    ✓ EXECUTING: {tool_name}")
            executed_count += 1

            try:
                selected_tool = next(
                    (t for t in analysis_tools if t.name == tool_name), None
                )

                if selected_tool:
                    tool_output = selected_tool.invoke(tc["args"])
                    new_messages.append(
                        ToolMessage(
                            content=safe_json_serialize(tool_output),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
                else:
                    new_messages.append(
                        ToolMessage(
                            content=json.dumps(
                                {
                                    "success": False,
                                    "error": f"Tool '{tool_name}' not found",
                                }
                            ),
                            tool_call_id=tool_id,
                            name=tool_name,
                        )
                    )
            except Exception as e:
                logger.error(f"Error executing {tool_name}:  {e}")
                new_messages.append(
                    ToolMessage(
                        content=json.dumps({"success": False, "error": str(e)}),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

    print(f"    Summary: {executed_count} executed, {blocked_count} blocked")
    print("  " + "-" * 40)

    extracted = extract_data_from_messages(new_messages, state)
    new_total = state.get("total_tool_calls", 0) + len(last_message.tool_calls)

    return {
        "messages": new_messages,
        "total_tool_calls": new_total,
        "analysis_results": extracted["analysis_results"],
        "tool_call_counts": extracted["tool_call_counts"],
    }


###############################################################################
# Reviewer Node
###############################################################################


def reviewer_node(state: AgentState) -> Dict[str, Any]:
    """
    Reviewer node using STRUCTURED OUTPUT for deterministic routing.
    """
    iteration = state.get("iteration_count", 0) + 1
    data_status = check_data_status(state)

    plan = state.get("plan", {})
    ticker = plan.get("ticker", "NVDA")
    years = plan.get("years", [2023, 2024, 2025])

    _print_node_header("REVIEWER (Structured Output)")
    print(f"  Iteration: {iteration}/{MAX_ITERATIONS}")
    print("")
    print("  Data Status:")
    print(f"    - Financials: {'✓ YES' if data_status['has_financials'] else '✗ NO'}")
    print(f"    - Price Data: {'✓ YES' if data_status['has_price'] else '✗ NO'}")
    print(f"    - News Data:  {'✓ YES' if data_status['has_news'] else '✗ NO'}")
    print(f"    - Analysis Count: {data_status['analysis_count']}")
    print("")
    print("  Tools Exhausted:")
    print(
        f"    - All Research:  {'YES' if data_status['all_research_exhausted'] else 'NO'}"
    )
    print(
        f"    - All Analysis: {'YES' if data_status['all_analysis_exhausted'] else 'NO'}"
    )
    print(f"    - All Tools: {'YES' if data_status['all_tools_exhausted'] else 'NO'}")

    # HARD LIMIT 1: Max iterations
    if iteration >= MAX_ITERATIONS:
        print("")
        print("  ⚠ DECISION: Max iterations reached -> FORCE APPROVE")
        _print_node_footer()
        return _create_review_result(
            status="approved",
            feedback="Max iterations reached - approving with available data",
            iteration=iteration,
        )

    # HARD LIMIT 2: All tools exhausted
    if data_status["all_tools_exhausted"]:
        print("")
        print("  ⚠ DECISION:  All tools exhausted -> FORCE APPROVE")
        _print_node_footer()
        return _create_review_result(
            status="approved",
            feedback="All tools exhausted - approving with available data",
            iteration=iteration,
        )

    # Use structured output for LLM decision
    decision = _get_structured_review_decision(state, data_status)

    print("")
    print(f"  ✓ Structured Decision:")
    print(f"    Status: {decision. status. upper()}")
    print(f"    Reason: {decision.reason}")
    if decision.suggested_tool:
        print(f"    Suggested Tool: {decision. suggested_tool}")

    # Map decision to state updates
    status, pending_action, feedback = _map_review_decision(
        decision=decision, data_status=data_status, ticker=ticker, years=years
    )

    print(f"  DECISION:  {status.upper()}")
    _print_node_footer()

    return _create_review_result(
        status=status,
        pending_action=pending_action,
        feedback=feedback,
        iteration=iteration,
        include_message=True,
    )


def _get_structured_review_decision(
    state: AgentState, data_status: Dict[str, Any]
) -> ReviewDecision:
    """Get structured review decision from LLM."""
    tool_status = get_tool_status_for_reviewer(state)
    data_summary = get_data_summary(state)

    user_message = f"""
USER REQUEST:  {state['user_request']}

CURRENT DATA STATUS:
- Financial Data: {"Collected for years " + str(state. get('financial_data', {}).get('years', [])) if data_status['has_financials'] else "MISSING"}
- Price Data: {"Collected" if data_status['has_price'] else "MISSING"}
- News Data: {"Collected" if data_status['has_news'] else "MISSING"}
- Analysis Results: {data_status['analysis_count']} completed

TOOL AVAILABILITY:
{tool_status}

COLLECTED DATA SUMMARY:
{data_summary}

CONTEXT:
{build_context(state)}

Make your decision based on the above information. 
Only suggest tools that are AVAILABLE. 
Approve if we have sufficient data to write a comprehensive report. 
"""

    try:
        decision: ReviewDecision = reviewer_llm_structured.invoke(
            [SystemMessage(content=REVIEWER_PROMPT), HumanMessage(content=user_message)]
        )
        return decision

    except (ValidationError, AttributeError, Exception) as e:
        logger.warning(f"Structured output failed:  {e}")
        print(f"  ⚠ Structured output failed:  {e}")

        if data_status["has_financials"] and data_status["analysis_count"] >= 1:
            return create_fallback_review_decision(
                status="approved", reason="Fallback:  minimum data collected"
            )
        else:
            return create_fallback_review_decision(
                status="needs_research", reason="Fallback: need more data"
            )


def _map_review_decision(
    decision: ReviewDecision, data_status: Dict[str, Any], ticker: str, years: list
) -> tuple:
    """Map structured decision to status, pending_action, and feedback."""

    status = decision.status
    pending_action = None
    feedback = decision.reason

    if status == "needs_research" and decision.suggested_tool:
        tool = decision.suggested_tool

        tool_available = (
            (tool == "fetch_financials" and data_status["can_fetch_financials"])
            or (tool == "fetch_price_history" and data_status["can_fetch_price"])
            or (
                tool in ["fetch_general_news", "fetch_news_for_date"]
                and data_status["can_fetch_news"]
            )
        )

        if tool_available:
            if tool == "fetch_financials":
                pending_action = f"CALL fetch_financials(ticker_symbol='{ticker}')"
            elif tool == "fetch_price_history":
                pending_action = (
                    f"CALL fetch_price_history(ticker='{ticker}', years={years})"
                )
            elif tool in ["fetch_general_news", "fetch_news_for_date"]:
                pending_action = f"CALL fetch_general_news(ticker='{ticker}')"

            status = "needs_more_research"
        else:
            logger.warning(f"Suggested tool {tool} is exhausted")
            status = "approved"
            feedback = (
                f"Suggested tool {tool} is exhausted - approving with available data"
            )

    elif status == "needs_analysis" and decision.suggested_tool:
        if can_call_tool({"tool_call_counts": {}}, decision.suggested_tool):
            pending_action = f"Run {decision.suggested_tool} on collected data"
            status = "needs_more_analysis"
        else:
            status = "approved"
            feedback = f"Analysis tool exhausted - approving with available data"

    return status, pending_action, feedback


def _create_review_result(
    status: str,
    feedback: str,
    iteration: int,
    pending_action: str = None,
    include_message: bool = False,
) -> Dict[str, Any]:
    """Create a standardized review result dictionary."""
    result = {
        "review_status": status,
        "pending_action": pending_action,
        "review_feedback": feedback,
        "iteration_count": iteration,
        "current_phase": "reviewing",
        "current_node": "reviewer",
    }

    if include_message:
        result["messages"] = [
            AIMessage(content=f"Review Decision: {status} - {feedback}")
        ]
    else:
        result["messages"] = []

    return result


###############################################################################
# Writer Node
###############################################################################


def writer_node(state: AgentState) -> Dict[str, Any]:
    """
    Writer node that generates the final financial analysis report.
    """
    _print_node_header("WRITER")

    print(f"  Total iterations: {state. get('iteration_count', 0)}")
    print(f"  Total tool calls: {state. get('total_tool_calls', 0)}")
    print("  Tool usage summary:")

    tool_counts = state.get("tool_call_counts", {})
    for name in ALL_TOOL_NAMES:
        count = tool_counts.get(name, 0)
        if count > 0:
            print(f"    - {name}: {count}")

    print("  Generating report...")

    context = build_context(state)
    user_request = state["user_request"]

    response = writer_llm.invoke(
        [
            SystemMessage(content=WRITER_PROMPT),
            HumanMessage(content=f"{context}\n\nOriginal Request: {user_request}"),
        ]
    )

    print("  ✓ Report generated!")
    _print_node_footer()

    return {
        "messages": [response],
        "final_report": response.content,
        "current_phase": "complete",
        "current_node": "writer",
    }


###############################################################################
# Routing Functions
###############################################################################


def route_after_research_tools(state: AgentState) -> str:
    """
    Route after research tools execution.

    Decision logic:
    - If this is a research-only loop (came from reviewer for more research),
      go directly back to reviewer
    - If this is the initial flow (analysis not done yet), go to analyst

    Returns:
        Next node name:  'analyst' or 'reviewer'
    """
    current_phase = state.get("current_phase", "researching")
    review_status = state.get("review_status")
    analysis_count = len(state.get("analysis_results", {}))

    print(f"  Routing after research_tools:")
    print(f"    - Current phase: {current_phase}")
    print(f"    - Review status: {review_status}")
    print(f"    - Analysis count:  {analysis_count}")

    # If we came from reviewer requesting more research, go back to reviewer
    if review_status == "needs_more_research":
        print("  -> reviewer (research-only loop)")
        return "reviewer"

    # If analysis has been done at least once, we're in a research loop
    if analysis_count > 0 and current_phase == "researching":
        print("  -> reviewer (already have analysis)")
        return "reviewer"

    # First pass - go to analyst
    print("  -> analyst (initial flow)")
    return "analyst"


def route_after_review(state: AgentState) -> str:
    """
    Route to the next node based on review status.

    Returns:
        Next node name: 'writer', 'researcher', or 'analyst'
    """
    iteration = state.get("iteration_count", 0)
    status = state.get("review_status", "approved")

    print(f"  Routing:  iteration={iteration}, status={status}")

    # Safety check:  force completion if at limit
    if iteration >= MAX_ITERATIONS:
        print("  -> writer (iteration limit)")
        return "writer"

    # Route based on status
    if status == "approved":
        print("  -> writer")
        return "writer"
    elif status == "needs_more_research":
        print("  -> researcher")
        return "researcher"
    elif status == "needs_more_analysis":
        print("  -> analyst")
        return "analyst"

    # Default to writer
    print("  -> writer (default)")
    return "writer"
