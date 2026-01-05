"""
Constants and type definitions for the Financial Analysis Agent.
Single source of truth for tool names, limits, and type aliases.
"""

from typing import List, Literal, get_args

###############################################################################
# Configuration Constants
###############################################################################

MAX_CALLS_PER_TOOL: int = 5
MAX_ITERATIONS: int = 40


###############################################################################
# Tool Name Literals (for Pydantic type validation)
###############################################################################

ResearchToolName = Literal[
    "fetch_financials",
    "fetch_price_history",
    "fetch_news_for_date",
    "fetch_general_news",
]

AnalysisToolName = Literal[
    "compute_yoy_changes",
    "compute_financial_ratios",
    "compute_growth_metrics",
    "compute_summary_statistics",
    "compare_metrics_across_years",
]

# Combined type for any tool
AllToolName = Literal[
    "fetch_financials",
    "fetch_price_history",
    "fetch_news_for_date",
    "fetch_general_news",
    "compute_yoy_changes",
    "compute_financial_ratios",
    "compute_growth_metrics",
    "compute_summary_statistics",
    "compare_metrics_across_years",
]

ReviewStatus = Literal["approved", "needs_research", "needs_analysis"]


###############################################################################
# Tool Name Lists (derived from Literals for runtime use)
###############################################################################

# Convert Literal types to List[str] for runtime iteration
RESEARCH_TOOL_NAMES: List[str] = list(get_args(ResearchToolName))
ANALYSIS_TOOL_NAMES: List[str] = list(get_args(AnalysisToolName))
ALL_TOOL_NAMES: List[str] = list(get_args(AllToolName))


###############################################################################
# Validation Helpers
###############################################################################


def is_valid_tool_name(name: str) -> bool:
    """Check if a string is a valid tool name."""
    return name in ALL_TOOL_NAMES


def is_research_tool(name: str) -> bool:
    """Check if a tool is a research tool."""
    return name in RESEARCH_TOOL_NAMES


def is_analysis_tool(name: str) -> bool:
    """Check if a tool is an analysis tool."""
    return name in ANALYSIS_TOOL_NAMES
