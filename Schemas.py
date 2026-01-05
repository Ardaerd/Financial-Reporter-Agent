"""
Schema definitions for the Financial Analysis Agent.
Contains Pydantic models for structured LLM output and TypedDict for agent state.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from pydantic import BaseModel, Field

# Import types from constants (single source of truth)
from Constants import (AllToolName, AnalysisToolName, ResearchToolName,
                       ReviewStatus)

###############################################################################
# Pydantic Models for Structured LLM Output
###############################################################################


class PlanOutput(BaseModel):
    """Structured output for the planner node."""

    ticker: str = Field(description="Stock ticker symbol (e.g., NVDA, AMD, AAPL)")
    years: List[int] = Field(
        description="List of years to analyze (e.g., [2023, 2024, 2025])"
    )
    data_requirements: List[str] = Field(
        default_factory=lambda: ["financials", "price_history", "news"],
        description="Types of data needed for analysis",
    )
    analyses_required: List[str] = Field(
        default_factory=lambda: ["yoy_changes", "ratios", "growth_metrics"],
        description="Types of analysis to perform on the data",
    )
    report_sections: List[str] = Field(
        default_factory=lambda: [
            "executive_summary",
            "financial_performance",
            "outlook",
        ],
        description="Sections to include in the final report",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "ticker": "NVDA",
                "years": [2023, 2024, 2025],
                "data_requirements": ["financials", "price_history", "news"],
                "analyses_required": ["yoy_changes", "ratios", "growth_metrics"],
                "report_sections": [
                    "executive_summary",
                    "financial_performance",
                    "outlook",
                ],
            }
        }
    }


class ReviewDecision(BaseModel):
    """Structured output for the reviewer node."""

    status: ReviewStatus = Field(
        description="Decision status:  'approved', 'needs_research', or 'needs_analysis'"
    )
    reason: str = Field(
        description="Detailed explanation for the decision", min_length=5
    )
    missing_info: Optional[str] = Field(
        default=None, description="What specific data or analysis is missing (if any)"
    )
    suggested_tool: Optional[AllToolName] = Field(
        default=None, description="The exact tool name to call if more work is needed"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "approved",
                    "reason": "All required data collected and analysis complete.",
                    "missing_info": None,
                    "suggested_tool": None,
                },
                {
                    "status": "needs_research",
                    "reason": "Financial data has not been collected yet.",
                    "missing_info": "Income statement, balance sheet data",
                    "suggested_tool": "fetch_financials",
                },
            ]
        }
    }


class ResearcherDecision(BaseModel):
    """Structured output when researcher cannot call tools."""

    can_proceed: bool = Field(description="Whether there is enough data to proceed")
    explanation: str = Field(description="Explanation of current status")
    tools_attempted: List[str] = Field(
        default_factory=list, description="Tools that were attempted but exhausted"
    )
    data_collected: List[str] = Field(
        default_factory=list, description="Data types successfully collected"
    )


class AnalystDecision(BaseModel):
    """Structured output for analyst node decisions."""

    analyses_completed: List[str] = Field(
        default_factory=list, description="Analyses that have been completed"
    )
    analyses_needed: List[str] = Field(
        default_factory=list, description="Analyses still needed"
    )
    can_proceed: bool = Field(description="Whether enough analysis has been done")
    explanation: str = Field(description="Explanation of analysis status")


###############################################################################
# Data Models for Tool Outputs
###############################################################################


class FinancialDataRecord(BaseModel):
    """Schema for a single year's financial data."""

    year: int
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    ebitda: Optional[float] = None
    cfo: Optional[float] = None
    capex: Optional[float] = None
    fcf: Optional[float] = None
    total_assets: Optional[float] = None
    equity: Optional[float] = None
    total_debt: Optional[float] = None
    cash: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None


class FinancialDataOutput(BaseModel):
    """Schema for fetch_financials tool output."""

    ticker: str
    success: bool
    data: List[FinancialDataRecord] = Field(default_factory=list)
    years: List[int] = Field(default_factory=list)
    error: Optional[str] = None


class QuarterData(BaseModel):
    """Schema for quarterly price data."""

    quarter: str
    year: int
    period: str
    start_date: str
    end_date: str
    open: float
    close: float
    high: float
    low: float
    quarter_return_pct: float
    total_volume: int
    avg_daily_volume: int
    volatility_pct: float
    trading_days: int


class YearSummary(BaseModel):
    """Schema for yearly price summary."""

    year_open: float
    year_close: float
    year_high: float
    year_low: float
    ytd_return_pct: float
    total_volume: int
    quarters_available: int


class PriceDataOutput(BaseModel):
    """Schema for fetch_price_history tool output."""

    ticker: str
    success: bool
    periods: List[Dict[str, Any]] = Field(default_factory=list)
    quarterly_data: List[QuarterData] = Field(default_factory=list)
    significant_quarters: List[Dict[str, Any]] = Field(default_factory=list)
    qoq_changes: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


class NewsArticle(BaseModel):
    """Schema for a news article."""

    url: str
    title: str
    content: str


class NewsDataOutput(BaseModel):
    """Schema for news tool output."""

    ticker: str
    success: bool
    articles: List[NewsArticle] = Field(default_factory=list)
    event_date: Optional[str] = None
    error: Optional[str] = None


###############################################################################
# TypedDict for Agent State (used by LangGraph)
###############################################################################


class AgentState(TypedDict):
    """State definition for the financial analysis agent graph."""

    messages: Annotated[Sequence[Any], operator.add]
    user_request: str
    plan: Optional[Dict[str, Any]]
    financial_data: Optional[Dict[str, Any]]
    price_data: Optional[Dict[str, Any]]
    news_data: Optional[str]
    analysis_results: Dict[str, Any]
    current_phase: str
    current_node: str
    review_status: Optional[str]
    review_feedback: Optional[str]
    pending_action: Optional[str]
    iteration_count: int
    tool_call_counts: Dict[str, int]
    total_tool_calls: int
    final_report: Optional[str]
