"""
Utility functions for the Financial Analysis Agent.
Contains SHARED helper functions used across multiple modules.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from Constants import (ALL_TOOL_NAMES, ANALYSIS_TOOL_NAMES, MAX_CALLS_PER_TOOL,
                       MAX_ITERATIONS, RESEARCH_TOOL_NAMES)
from Schemas import AgentState, PlanOutput, ReviewDecision

logger = logging.getLogger(__name__)


###############################################################################
# Data Conversion Utilities (SHARED - used by multiple modules)
###############################################################################


def to_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert a list of dictionaries to a pandas DataFrame.

    Used by:  research_tools. py, analysis_tools.py
    """
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "year" in df.columns:
        df = df.set_index("year")
    return df


def df_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to list of record dictionaries.

    Used by:  research_tools.py, analysis_tools. py

    Args:
        df: DataFrame to convert

    Returns:
        List of dictionaries with year and metric values
    """
    records = []
    for idx in df.index:
        row = {"year": int(idx)} if isinstance(idx, (int, float)) else {}
        for col in df.columns:
            val = df.loc[idx, col]
            if pd.isna(val):
                row[col] = None
            elif isinstance(val, (int, float)):
                row[col] = float(val)
            else:
                row[col] = val
        records.append(row)
    return records


def safe_json_serialize(obj: Any, indent: int = 2) -> str:
    """
    Safely serialize an object to JSON, handling pandas types and special values.

    Used by: multiple modules for state serialization
    """

    def serializer(o: Any) -> Any:
        if pd.isna(o):
            return None
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        if isinstance(o, (set, frozenset)):
            return list(o)
        return str(o)

    return json.dumps(obj, default=serializer, indent=indent)


###############################################################################
# Tool Status Functions (SHARED - used by nodes)
###############################################################################


def get_tool_call_count(state: AgentState, tool_name: str) -> int:
    """Get the current call count for a specific tool."""
    return state.get("tool_call_counts", {}).get(tool_name, 0)


def can_call_tool(
    state: AgentState, tool_name: str, max_calls: int = MAX_CALLS_PER_TOOL
) -> bool:
    """Check if a tool can still be called."""
    current_count = state.get("tool_call_counts", {}).get(tool_name, 0)
    return current_count < max_calls


def get_available_tools_info(
    state: AgentState, tool_names: List[str], max_calls: int = MAX_CALLS_PER_TOOL
) -> str:
    """Generate status info for tools to include in LLM prompts."""
    tool_counts = state.get("tool_call_counts", {})
    lines = [f"TOOL CALL LIMITS (each tool max {max_calls} calls):"]

    for name in tool_names:
        count = tool_counts.get(name, 0)
        remaining = max_calls - count

        if remaining > 0:
            status = f"✓ AVAILABLE - {count}/{max_calls} used, {remaining} remaining"
        else:
            status = "✗ EXHAUSTED - DO NOT CALL THIS TOOL"

        lines.append(f"  • {name}: {status}")

    lines.append("")
    lines.append("IMPORTANT: Do NOT attempt to call any EXHAUSTED tool.")

    return "\n".join(lines)


def get_tool_status_for_reviewer(
    state: AgentState, max_calls: int = MAX_CALLS_PER_TOOL
) -> str:
    """Generate tool status summary for the reviewer node."""
    tool_counts = state.get("tool_call_counts", {})
    lines = []

    lines.append("RESEARCH TOOLS:")
    for name in RESEARCH_TOOL_NAMES:
        count = tool_counts.get(name, 0)
        remaining = max_calls - count
        status = f"AVAILABLE ({remaining} remaining)" if remaining > 0 else "EXHAUSTED"
        lines.append(f"  - {name}: {status}")

    lines.append("")
    lines.append("ANALYSIS TOOLS:")
    for name in ANALYSIS_TOOL_NAMES:
        count = tool_counts.get(name, 0)
        remaining = max_calls - count
        status = f"AVAILABLE ({remaining} remaining)" if remaining > 0 else "EXHAUSTED"
        lines.append(f"  - {name}: {status}")

    return "\n".join(lines)


###############################################################################
# Limit Checking Functions (SHARED - used by nodes)
###############################################################################


def check_limits(
    state: AgentState, max_iterations: int = MAX_ITERATIONS
) -> Dict[str, Any]:
    """Check current iteration and tool call limits."""
    iteration_count = state.get("iteration_count", 0)

    return {
        "iteration_limit_reached": iteration_count >= max_iterations,
        "iteration_count": iteration_count,
        "max_iterations": max_iterations,
        "total_tool_calls": state.get("total_tool_calls", 0),
        "tool_counts": state.get("tool_call_counts", {}),
    }


def check_data_status(state: AgentState) -> Dict[str, Any]:
    """Check what data has been collected and what's missing."""
    has_financials = state.get("financial_data") is not None
    has_price = state.get("price_data") is not None

    news_data = state.get("news_data")
    has_news = news_data is not None and len(news_data) > 10

    analysis_results = state.get("analysis_results", {})
    analysis_count = len(analysis_results)

    can_fetch_financials = can_call_tool(state, "fetch_financials")
    can_fetch_price = can_call_tool(state, "fetch_price_history")
    can_fetch_news = can_call_tool(state, "fetch_general_news")

    all_research_exhausted = not (
        can_fetch_financials or can_fetch_price or can_fetch_news
    )
    all_analysis_exhausted = all(
        not can_call_tool(state, t) for t in ANALYSIS_TOOL_NAMES
    )

    return {
        "has_financials": has_financials,
        "has_price": has_price,
        "has_news": has_news,
        "analysis_count": analysis_count,
        "can_fetch_financials": can_fetch_financials,
        "can_fetch_price": can_fetch_price,
        "can_fetch_news": can_fetch_news,
        "all_research_exhausted": all_research_exhausted,
        "all_analysis_exhausted": all_analysis_exhausted,
        "all_tools_exhausted": all_research_exhausted and all_analysis_exhausted,
    }


###############################################################################
# Message Extraction Functions (SHARED - used by tool nodes)
###############################################################################


def extract_data_from_messages(
    new_messages: Sequence[BaseMessage], state: AgentState
) -> Dict[str, Any]:
    """Extract tool results from messages with safe JSON parsing."""
    financial_data = state.get("financial_data")
    price_data = state.get("price_data")
    news_items: List[Dict[str, Any]] = []

    existing_news = state.get("news_data")
    if existing_news:
        try:
            parsed = json.loads(existing_news)
            news_items = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            logger.warning("Failed to parse existing news data")

    analysis_results = dict(state.get("analysis_results", {}))
    tool_call_counts = dict(state.get("tool_call_counts", {}))

    for msg in new_messages:
        if not isinstance(msg, ToolMessage):
            continue

        data: Optional[Dict[str, Any]] = None
        try:
            if isinstance(msg.content, str):
                data = json.loads(msg.content)
            elif isinstance(msg.content, dict):
                data = msg.content
            else:
                continue
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error:  {e}")
            data = {"success": False, "error": str(msg.content)}

        tool_name = getattr(msg, "name", "unknown_tool")
        tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

        if data and data.get("success"):
            if tool_name == "fetch_financials":
                financial_data = data
            elif tool_name == "fetch_price_history":
                price_data = data
            elif tool_name in ["fetch_news_for_date", "fetch_general_news"]:
                news_items.append(data)
            elif (
                tool_name.startswith("compute_")
                or tool_name == "compare_metrics_across_years"
            ):
                analysis_results[tool_name] = data

    return {
        "financial_data": financial_data,
        "price_data": price_data,
        "news_data": safe_json_serialize(news_items) if news_items else None,
        "analysis_results": analysis_results,
        "tool_call_counts": tool_call_counts,
    }


###############################################################################
# Context Building Functions (SHARED - used by nodes)
###############################################################################


def build_context(state: AgentState, max_news_length: int = 3000) -> str:
    """Build a context string from the current state."""
    parts = []

    if state.get("plan"):
        parts.append(f"## Plan:\n{safe_json_serialize(state['plan'])}")
    if state.get("financial_data"):
        parts.append(
            f"## Financial Data:\n{safe_json_serialize(state['financial_data'])}"
        )
    if state.get("price_data"):
        parts.append(f"## Price Data:\n{safe_json_serialize(state['price_data'])}")
    if state.get("news_data"):
        news = state["news_data"][:max_news_length]
        if len(state["news_data"]) > max_news_length:
            news += "\n... (truncated)"
        parts.append(f"## News:\n{news}")
    if state.get("analysis_results"):
        parts.append(f"## Analysis:\n{safe_json_serialize(state['analysis_results'])}")
    if state.get("pending_action"):
        parts.append(f"## PENDING ACTION:\n{state['pending_action']}")

    return "\n\n".join(parts)


def get_data_summary(state: AgentState) -> str:
    """Get a summary of collected data for the reviewer."""
    lines = []

    fin_data = state.get("financial_data")
    if fin_data and fin_data.get("success"):
        lines.append(
            f"FINANCIAL DATA:  Collected for years {fin_data. get('years', [])}"
        )
    else:
        lines.append("FINANCIAL DATA: Not collected")

    price_data = state.get("price_data")
    if price_data and price_data.get("success"):
        periods = price_data.get("periods", [])
        years = [p.get("year") for p in periods if p.get("year")]
        lines.append(f"PRICE DATA: Collected for years {years}")
    else:
        lines.append("PRICE DATA: Not collected")

    news_data = state.get("news_data")
    if news_data and len(news_data) > 10:
        try:
            news_list = json.loads(news_data)
            total = sum(
                len(i.get("articles", [])) for i in news_list if isinstance(i, dict)
            )
            lines.append(f"NEWS DATA: {total} articles collected")
        except json.JSONDecodeError:
            lines.append("NEWS DATA:  Collected (parse error)")
    else:
        lines.append("NEWS DATA:  Not collected")

    analysis = state.get("analysis_results", {})
    if analysis:
        lines.append(f"ANALYSIS COMPLETED: {list(analysis.keys())}")
    else:
        lines.append("ANALYSIS:  None completed")

    return "\n".join(lines)


###############################################################################
# Fallback Creators (SHARED - used by nodes)
###############################################################################


def create_default_plan(
    ticker: str = "NVDA", years: Optional[List[int]] = None
) -> PlanOutput:
    """Create a default plan when LLM parsing fails."""
    return PlanOutput(ticker=ticker, years=years or [2023, 2024, 2025])


def create_fallback_review_decision(
    status: str = "approved", reason: str = "Fallback approval due to parsing error"
) -> ReviewDecision:
    """Create a fallback review decision when structured output fails."""
    return ReviewDecision(status=status, reason=reason)


###############################################################################
# State Initialization (SHARED - used by main)
###############################################################################


def init_state(user_request: str) -> AgentState:
    """Initialize a new agent state."""
    return AgentState(
        messages=[HumanMessage(content=user_request)],
        user_request=user_request,
        plan=None,
        financial_data=None,
        price_data=None,
        news_data=None,
        analysis_results={},
        current_phase="planning",
        current_node="planner",
        review_status=None,
        review_feedback=None,
        pending_action=None,
        iteration_count=0,
        tool_call_counts={},
        total_tool_calls=0,
        final_report=None,
    )
