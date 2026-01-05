"""
Financial Analysis Agent - Streamlit UI

A modern, interactive web interface for the Financial Analysis Agent.
Provides real-time analysis, visualizations, and export capabilities.
"""

import json
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from Constants import ALL_TOOL_NAMES, MAX_CALLS_PER_TOOL, MAX_ITERATIONS
# Import from your existing modules
from main import FinancialAnalysisAgent, compile_graph
from Schemas import AgentState
from Utilities import init_state, safe_json_serialize

###############################################################################
# Streamlit Configuration
###############################################################################

st.set_page_config(
    page_title="Financial Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight:  bold;
        background: linear-gradient(90deg, #1f77b4, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color:  transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    . sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Status boxes */
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.5rem;
        margin:  1rem 0;
    }
    
    .info-box {
        background-color: #e7f3ff;
        border-left:  4px solid #1f77b4;
        border-radius:  0 8px 8px 0;
        padding:  1rem 1.5rem;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    . stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
    }
    
    . stTabs [data-baseweb="tab"] {
        height: 45px;
        padding: 0 20px;
        border-radius: 8px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    . stButton > button[kind="primary"] {
        background:  linear-gradient(90deg, #1f77b4, #2ecc71);
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1. 1rem;
        font-weight: 600;
        border-radius: 10px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(31, 119, 180, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background:  linear-gradient(90deg, #1f77b4, #2ecc71);
    }
    
    /* Text area styling */
    . stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }
    
    . stTextArea textarea:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)


###############################################################################
# Session State Initialization
###############################################################################


def init_session_state():
    """Initialize all session state variables."""
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "ticker" not in st.session_state:
        st.session_state.ticker = ""
    if "run_complete" not in st.session_state:
        st.session_state.run_complete = False
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""


###############################################################################
# Helper Functions
###############################################################################


def safe_get(data: Any, *keys, default=None) -> Any:
    """Safely get nested dictionary values."""
    try:
        result = data
        for key in keys:
            if isinstance(result, dict):
                result = result.get(key, default)
            else:
                return default
        return result if result is not None else default
    except Exception:
        return default


def parse_financial_data(financial_data: Dict[str, Any]) -> pd.DataFrame:
    """Parse financial data into a DataFrame."""
    try:
        if not financial_data or not financial_data.get("data"):
            return pd.DataFrame()

        df = pd.DataFrame(financial_data["data"])
        if "year" in df.columns:
            df = df.set_index("year")
            df = df.sort_index()

        return df
    except Exception as e:
        st.error(f"Error parsing financial data: {e}")
        return pd.DataFrame()


def format_large_number(num: Any) -> str:
    """Format large numbers for display."""
    try:
        if num is None or pd.isna(num):
            return "N/A"
        num = float(num)
        if abs(num) >= 1e12:
            return f"${num/1e12:.2f}T"
        elif abs(num) >= 1e9:
            return f"${num/1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"${num/1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"${num/1e3:.2f}K"
        else:
            return f"${num:.2f}"
    except Exception:
        return "N/A"


def format_percentage(num: Any) -> str:
    """Format percentage values."""
    try:
        if num is None or pd.isna(num):
            return "N/A"
        num = float(num)
        if abs(num) < 1:
            return f"{num * 100:.2f}%"
        else:
            return f"{num:. 2f}%"
    except Exception:
        return "N/A"


def format_ratio(num: Any) -> str:
    """Format ratio values."""
    try:
        if num is None or pd.isna(num):
            return "N/A"
        return f"{float(num):.2f}"
    except Exception:
        return "N/A"


def format_change(num: Any) -> str:
    """Format change values with color indicators."""
    try:
        if num is None or pd.isna(num):
            return "N/A"
        num = float(num)
        if num > 0:
            return f"🟢 +{num:.2f}%"
        elif num < 0:
            return f"🔴 {num:.2f}%"
        else:
            return f"⚪ {num:.2f}%"
    except Exception:
        return "N/A"


def get_status_icon(has_data: bool) -> str:
    """Get status icon based on data availability."""
    return "✅" if has_data else "❌"


def get_trend_icon(trend: str) -> str:
    """Get trend icon based on trend direction."""
    if trend == "increasing":
        return "📈"
    elif trend == "decreasing":
        return "📉"
    else:
        return "➡️"


def extract_ticker_from_query(query: str) -> str:
    """Extract ticker symbol from user query."""
    import re

    # Known tickers to prioritize
    known_tickers = [
        "NVDA",
        "AAPL",
        "MSFT",
        "GOOGL",
        "GOOG",
        "AMZN",
        "META",
        "TSLA",
        "AMD",
        "INTC",
        "NFLX",
        "ADBE",
        "CRM",
        "ORCL",
        "IBM",
        "CSCO",
        "JPM",
        "BAC",
        "WFC",
        "GS",
        "MS",
        "V",
        "MA",
        "PYPL",
        "JNJ",
        "PFE",
        "UNH",
        "MRK",
        "ABBV",
        "LLY",
        "TMO",
        "BMY",
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "PXD",
        "VLO",
        "MPC",
        "DIS",
        "CMCSA",
        "T",
        "VZ",
        "TMUS",
        "CHTR",
        "NFLX",
        "HD",
        "LOW",
        "TGT",
        "WMT",
        "COST",
        "AMZN",
        "EBAY",
        "BA",
        "LMT",
        "RTX",
        "GE",
        "CAT",
        "DE",
        "MMM",
        "HON",
        "KO",
        "PEP",
        "MCD",
        "SBUX",
        "NKE",
        "LULU",
        "TJX",
    ]

    # Check for known tickers first
    query_upper = query.upper()
    for ticker in known_tickers:
        if ticker in query_upper:
            return ticker

    # Fallback to pattern matching
    ticker_pattern = r"\b([A-Z]{1,5})\b"
    matches = re.findall(ticker_pattern, query.upper())

    # Filter out common words that might match
    common_words = {
        "FOR",
        "THE",
        "AND",
        "WITH",
        "FROM",
        "THIS",
        "THAT",
        "WHAT",
        "WHY",
        "HOW",
        "YOY",
        "CEO",
        "CFO",
        "IPO",
        "ETF",
        "USD",
        "EUR",
    }

    for match in matches:
        if match not in common_words and len(match) >= 2:
            return match

    return "UNKNOWN"


###############################################################################
# Example Queries
###############################################################################

EXAMPLE_QUERIES = [
    "Analyze NVDA for 2023-2025:  profitability, balance sheet, cash flow, stock performance, and recent news.  Include YoY changes.",
    "Give me a comprehensive investment analysis of AAPL including financial health, growth metrics, and risk assessment.",
    "What happened to TSLA stock in 2024? Analyze the price movements and explain the major changes.",
    "Compare MSFT's margins and profitability trends over the last 3 years.  Why did operating margin change?",
    "Provide a risk assessment for META including debt levels, liquidity ratios, and recent negative news.",
    "Analyze GOOGL's revenue growth and explain what's driving the changes.  Include news context.",
    "Quick overview of AMZN's financial health for 2024.",
    "Deep dive into AMD's gross margin trends and competitive position based on recent news.",
]


###############################################################################
# Display Components
###############################################################################


def display_header():
    """Display the main header."""
    st.markdown(
        '<p class="main-header">📊 Financial Analysis Agent</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">AI-Powered Stock Analysis with Real-Time Data Collection & Insights</p>',
        unsafe_allow_html=True,
    )


def display_sidebar():
    """Display sidebar with example queries and settings."""
    with st.sidebar:
        st.header("⚙️ Settings")

        st.divider()

        # Agent info
        st.subheader("🤖 Agent Configuration")
        with st.expander("View Limits", expanded=False):
            st.info(f"**Max Calls Per Tool:** {MAX_CALLS_PER_TOOL}")
            st.info(f"**Max Iterations:** {MAX_ITERATIONS}")
            st.caption("These limits prevent infinite loops and control API usage.")

        st.divider()

        # Example queries section
        st.subheader("💡 Example Queries")
        st.caption("Click to paste into search box:")

        # Display first 5 examples
        for i, query in enumerate(EXAMPLE_QUERIES[:5]):
            # Truncate for button display
            display_text = query[:40] + "..." if len(query) > 40 else query
            if st.button(
                f"📝 {display_text}",
                key=f"example_{i}",
                use_container_width=True,
                help=query,  # Show full query on hover
            ):
                # Update session state with selected query
                st.session_state.user_query = query
                st.rerun()

        # More examples in expander
        with st.expander("More Examples", expanded=False):
            for i, query in enumerate(EXAMPLE_QUERIES[5:], start=5):
                display_text = query[:40] + "..." if len(query) > 40 else query
                if st.button(
                    f"📝 {display_text}",
                    key=f"example_{i}",
                    use_container_width=True,
                    help=query,
                ):
                    st.session_state.user_query = query
                    st.rerun()

        st.divider()

        # Actions
        st.subheader("🔧 Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "🗑�� Clear All",
                use_container_width=True,
                help="Clear results and query",
            ):
                st.session_state.user_query = ""
                st.session_state.analysis_result = None
                st.session_state.ticker = ""
                st.session_state.run_complete = False
                st.session_state.last_query = ""
                st.rerun()

        with col2:
            if st.button("🔄 Reset", use_container_width=True, help="Reset query only"):
                st.session_state.user_query = ""
                st.rerun()


def display_query_input() -> str:
    """Display the query input section and return the user's query."""

    st.markdown("### 💬 What would you like to analyze?")

    # Placeholder text without problematic characters
    placeholder = "Enter your analysis request here.. .\n\n"

    # Text area - use session state value directly
    user_query = st.text_area(
        label="Analysis Request",
        value=st.session_state.user_query,
        height=120,
        placeholder=placeholder,
        help="Type your financial analysis question.  Be specific about the ticker, time period, and what aspects you want to analyze.",
        label_visibility="collapsed",
    )

    # Sync back to session state if user types something new
    if user_query != st.session_state.user_query:
        st.session_state.user_query = user_query

    # Tips expander
    with st.expander("💡 Tips for better results", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **Include in your query:**
            - **Ticker symbol**:  NVDA, AAPL, MSFT, etc.
            - **Time period**: "2023-2025", "last 3 years"
            - **Analysis type**: profitability, growth, risk
            - **Specific questions**: "Why did margins decline?"
            """
            )

        with col2:
            st.markdown(
                """
            **Query complexity levels:**
            - **Quick**:  "Quick summary of AAPL"
            - **Standard**: "Analyze NVDA profitability 2023-2025"
            - **Deep**: "Comprehensive analysis with news context"
            
            **Add**:  "Include YoY changes" or "Explain with news"
            """
            )

    return st.session_state.user_query


def display_data_status(result: Dict[str, Any]):
    """Display data collection status cards."""
    col1, col2, col3, col4, col5 = st.columns(5)

    has_financials = result.get("financial_data") is not None
    has_price = result.get("price_data") is not None
    has_news = (
        result.get("news_data") is not None
        and len(str(result.get("news_data", ""))) > 10
    )
    analysis_count = len(result.get("analysis_results", {}))
    has_report = bool(result.get("final_report"))

    with col1:
        st.metric(
            label="📊 Financials",
            value=get_status_icon(has_financials),
            help="Income statement, balance sheet, cash flow data",
        )

    with col2:
        st.metric(
            label="📈 Price Data",
            value=get_status_icon(has_price),
            help="Historical stock price data",
        )

    with col3:
        st.metric(
            label="📰 News",
            value=get_status_icon(has_news),
            help="Recent news articles",
        )

    with col4:
        st.metric(
            label="🔍 Analyses",
            value=str(analysis_count),
            help="Number of analyses completed",
        )

    with col5:
        st.metric(
            label="📋 Report",
            value=get_status_icon(has_report),
            help="Final analysis report",
        )


def display_execution_stats(result: Dict[str, Any]):
    """Display execution statistics."""
    with st.expander("Execution Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Iterations", result.get("iteration_count", 0))

        with col2:
            st.metric("Total Tool Calls", result.get("total_tool_calls", 0))

        with col3:
            st.metric("Final Status", result.get("review_status", "N/A"))

        # Tool usage breakdown
        tool_counts = result.get("tool_call_counts", {})
        if tool_counts:
            st.subheader("Tool Usage Breakdown")

            tool_data = []
            for name in ALL_TOOL_NAMES:
                count = tool_counts.get(name, 0)
                if count > 0:
                    tool_data.append(
                        {
                            "Tool": name,
                            "Calls": count,
                            "Remaining": MAX_CALLS_PER_TOOL - count,
                        }
                    )

            if tool_data:
                st.dataframe(
                    pd.DataFrame(tool_data), use_container_width=True, hide_index=True
                )


def display_financial_data(financial_data: Dict[str, Any]):
    """Display financial data in a structured format."""
    if not financial_data or not financial_data.get("data"):
        st.warning("No financial data available")
        return

    df = parse_financial_data(financial_data)

    if df.empty:
        st.warning("Could not parse financial data")
        return

    # Key Financial Metrics
    st.markdown("#### 💰 Key Financial Metrics")

    key_cols = ["revenue", "net_income", "gross_profit", "operating_income", "ebitda"]
    available_cols = [col for col in key_cols if col in df.columns]

    if available_cols:
        display_df = df[available_cols].copy()
        for col in display_df.columns:
            display_df[col] = display_df[col].apply(format_large_number)
        display_df.columns = [
            col.replace("_", " ").title() for col in display_df.columns
        ]
        st.dataframe(display_df, use_container_width=True)

    # Per Share Metrics
    st.markdown("#### 📊 Per Share & Cash Flow Metrics")

    share_cols = ["eps", "fcf", "cfo", "capex"]
    available_share = [col for col in share_cols if col in df.columns]

    if available_share:
        share_df = df[available_share].copy()
        for col in share_df.columns:
            if col == "eps":
                share_df[col] = share_df[col].apply(
                    lambda x: f"${x:.2f}" if pd.notna(x) else "N/A"
                )
            else:
                share_df[col] = share_df[col].apply(format_large_number)
        share_df.columns = [
            (
                col.upper()
                if col in ["eps", "fcf", "cfo"]
                else col.replace("_", " ").title()
            )
            for col in share_df.columns
        ]
        st.dataframe(share_df, use_container_width=True)

    # Profitability Margins
    st.markdown("#### 📈 Profitability Margins")

    margin_cols = ["gross_margin", "operating_margin", "net_margin"]
    available_margins = [col for col in margin_cols if col in df.columns]

    if available_margins:
        margin_df = df[available_margins].copy()
        for col in margin_df.columns:
            margin_df[col] = margin_df[col].apply(format_percentage)
        margin_df.columns = [col.replace("_", " ").title() for col in margin_df.columns]
        st.dataframe(margin_df, use_container_width=True)

    # Balance Sheet
    st.markdown("#### 🏦 Balance Sheet Highlights")

    balance_cols = [
        "total_assets",
        "equity",
        "total_debt",
        "cash",
        "current_assets",
        "current_liabilities",
    ]
    available_balance = [col for col in balance_cols if col in df.columns]

    if available_balance:
        balance_df = df[available_balance].copy()
        for col in balance_df.columns:
            balance_df[col] = balance_df[col].apply(format_large_number)
        balance_df.columns = [
            col.replace("_", " ").title() for col in balance_df.columns
        ]
        st.dataframe(balance_df, use_container_width=True)

    # Financial Ratios
    st.markdown("#### 📐 Financial Ratios")

    ratio_cols = ["current_ratio", "debt_to_equity"]
    available_ratios = [col for col in ratio_cols if col in df.columns]

    if available_ratios:
        ratio_df = df[available_ratios].copy()
        for col in ratio_df.columns:
            ratio_df[col] = ratio_df[col].apply(format_ratio)
        ratio_df.columns = [col.replace("_", " ").title() for col in ratio_df.columns]
        st.dataframe(ratio_df, use_container_width=True)

    # Full data in expander
    with st.expander("📄 View All Raw Data"):
        st.dataframe(df, use_container_width=True)


def display_price_data(price_data: Dict[str, Any]):
    """Display price analysis data."""
    if not price_data:
        st.warning("No price data available")
        return

    # Yearly Performance
    periods = price_data.get("periods", [])
    if periods:
        st.markdown("#### 📅 Yearly Performance Summary")

        perf_data = []
        for period in periods:
            year_summary = period.get("year_summary", {})
            if year_summary:
                perf_data.append(
                    {
                        "Year": period.get("year", "N/A"),
                        "Open": f"${year_summary.get('year_open', 0):.2f}",
                        "Close": f"${year_summary.get('year_close', 0):.2f}",
                        "High": f"${year_summary. get('year_high', 0):.2f}",
                        "Low": f"${year_summary.get('year_low', 0):.2f}",
                        "YTD Return": format_change(
                            year_summary.get("ytd_return_pct", 0)
                        ),
                    }
                )

        if perf_data:
            st.dataframe(
                pd.DataFrame(perf_data), use_container_width=True, hide_index=True
            )

    # Quarterly Data
    quarterly_data = price_data.get("quarterly_data", [])
    if quarterly_data:
        st.markdown("#### 📊 Quarterly Performance")

        quarterly_df = pd.DataFrame(quarterly_data)
        if not quarterly_df.empty:
            display_cols = [
                "period",
                "open",
                "close",
                "quarter_return_pct",
                "volatility_pct",
                "trading_days",
            ]
            available_cols = [
                col for col in display_cols if col in quarterly_df.columns
            ]

            if available_cols:
                q_display = quarterly_df[available_cols].copy()

                if "open" in q_display.columns:
                    q_display["open"] = q_display["open"].apply(lambda x: f"${x:.2f}")
                if "close" in q_display.columns:
                    q_display["close"] = q_display["close"].apply(lambda x: f"${x:.2f}")
                if "quarter_return_pct" in q_display.columns:
                    q_display["quarter_return_pct"] = q_display[
                        "quarter_return_pct"
                    ].apply(format_change)
                if "volatility_pct" in q_display.columns:
                    q_display["volatility_pct"] = q_display["volatility_pct"].apply(
                        lambda x: f"{x:.2f}%"
                    )

                q_display.columns = [
                    col.replace("_", " ").title() for col in q_display.columns
                ]
                st.dataframe(q_display, use_container_width=True, hide_index=True)

    # Significant Quarters
    significant = price_data.get("significant_quarters", [])
    if significant:
        st.markdown("#### 🎯 Significant Price Movements")

        sig_df = pd.DataFrame(significant)
        if not sig_df.empty:
            sig_df["return_pct"] = sig_df["return_pct"].apply(format_change)
            sig_df["direction"] = sig_df["direction"].apply(
                lambda x: "🟢 Up" if x == "up" else "🔴 Down"
            )

            display_cols = [
                "period",
                "direction",
                "return_pct",
                "open",
                "close",
                "volatility",
            ]
            available = [col for col in display_cols if col in sig_df.columns]

            if available:
                sig_display = sig_df[available].copy()
                if "open" in sig_display.columns:
                    sig_display["open"] = sig_display["open"].apply(
                        lambda x: f"${x:.2f}"
                    )
                if "close" in sig_display.columns:
                    sig_display["close"] = sig_display["close"].apply(
                        lambda x: f"${x:.2f}"
                    )

                sig_display.columns = [
                    col.replace("_", " ").title() for col in sig_display.columns
                ]
                st.dataframe(sig_display, use_container_width=True, hide_index=True)


def display_analysis_results(analysis_results: Dict[str, Any]):
    """Display analysis results in a structured format."""
    if not analysis_results:
        st.warning("No analysis results available")
        return

    # YoY Changes
    if "compute_yoy_changes" in analysis_results:
        yoy_result = analysis_results["compute_yoy_changes"]
        if isinstance(yoy_result, dict) and yoy_result.get("success"):
            st.markdown("#### 📈 Year-over-Year Changes")

            yoy_data = yoy_result.get("results", {})
            for metric, data in yoy_data.items():
                if isinstance(data, dict):
                    with st.expander(
                        f"📊 {metric. replace('_', ' ').title()}", expanded=False
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Values by Year**")
                            values = data.get("values", {})
                            if values:
                                values_df = pd.DataFrame(
                                    [
                                        {
                                            "Year": year,
                                            "Value": (
                                                format_large_number(val)
                                                if abs(val) > 1000
                                                else f"{val:.2f}"
                                            ),
                                        }
                                        for year, val in sorted(values.items())
                                    ]
                                )
                                st.dataframe(
                                    values_df, use_container_width=True, hide_index=True
                                )

                        with col2:
                            st.markdown("**YoY Changes**")
                            changes = data.get("yoy_changes", {})
                            if changes:
                                changes_df = pd.DataFrame(
                                    [
                                        {"Year": year, "Change": format_change(change)}
                                        for year, change in sorted(changes.items())
                                    ]
                                )
                                st.dataframe(
                                    changes_df,
                                    use_container_width=True,
                                    hide_index=True,
                                )

    # Financial Ratios
    if "compute_financial_ratios" in analysis_results:
        ratios_result = analysis_results["compute_financial_ratios"]
        if isinstance(ratios_result, dict) and ratios_result.get("success"):
            st.markdown("#### 📐 Financial Ratios")

            ratios_data = ratios_result.get("ratios", {})
            if ratios_data:
                for ratio_name, ratio_values in ratios_data.items():
                    if isinstance(ratio_values, dict):
                        values = ratio_values.get("values", ratio_values)
                        avg = ratio_values.get("avg")
                        trend = ratio_values.get("trend", "stable")

                        with st.expander(
                            f"{get_trend_icon(trend)} {ratio_name}", expanded=False
                        ):
                            if avg is not None:
                                st.metric("Average", f"{avg:.4f}")

                            if isinstance(values, dict):
                                ratio_df = pd.DataFrame(
                                    [
                                        {
                                            "Year": year,
                                            "Value": f"{val:.4f}" if val else "N/A",
                                        }
                                        for year, val in sorted(values.items())
                                    ]
                                )
                                st.dataframe(
                                    ratio_df, use_container_width=True, hide_index=True
                                )

    # Growth Metrics
    if "compute_growth_metrics" in analysis_results:
        growth_result = analysis_results["compute_growth_metrics"]
        if isinstance(growth_result, dict) and growth_result.get("success"):
            st.markdown("#### 🚀 Growth Metrics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Metric",
                    growth_result.get("column", "N/A").replace("_", " ").title(),
                )

            with col2:
                st.metric(
                    "Total Growth", f"{growth_result. get('total_growth_pct', 0):.2f}%"
                )

            with col3:
                st.metric("CAGR", f"{growth_result.get('cagr_pct', 0):.2f}%")

            with col4:
                st.metric(
                    "Period",
                    f"{growth_result. get('start_year', 'N/A')} - {growth_result.get('end_year', 'N/A')}",
                )

    # Summary Statistics
    if "compute_summary_statistics" in analysis_results:
        stats_result = analysis_results["compute_summary_statistics"]
        if isinstance(stats_result, dict) and stats_result.get("success"):
            st.markdown("#### 📊 Summary Statistics")

            stats_data = stats_result.get("statistics", {})
            if stats_data:
                stats_rows = []
                for metric, stats in stats_data.items():
                    if isinstance(stats, dict):
                        trend = stats.get("trend", "stable")
                        stats_rows.append(
                            {
                                "Metric": metric.replace("_", " ").title(),
                                "Mean": f"{stats. get('mean', 0):.4f}",
                                "Min": f"{stats. get('min', 0):.4f}",
                                "Max": f"{stats.get('max', 0):.4f}",
                                "Trend": f"{get_trend_icon(trend)} {trend. title()}",
                            }
                        )

                if stats_rows:
                    st.dataframe(
                        pd.DataFrame(stats_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

    # Comparison
    if "compare_metrics_across_years" in analysis_results:
        comparison_result = analysis_results["compare_metrics_across_years"]
        if isinstance(comparison_result, dict) and comparison_result.get("success"):
            st.markdown("#### 🔄 Year-over-Year Comparison")

            comparison_data = comparison_result.get("comparison", {})
            if comparison_data:
                try:
                    comp_df = pd.DataFrame(comparison_data).T
                    comp_df.index.name = "Metric"
                    st.dataframe(comp_df, use_container_width=True)
                except Exception:
                    st.json(comparison_data)


def display_news(news_data: str):
    """Display news articles."""
    if not news_data:
        st.warning("No news data available")
        return

    try:
        news_list = json.loads(news_data) if isinstance(news_data, str) else news_data

        if not isinstance(news_list, list) or not news_list:
            st.warning("No news articles found")
            return

        article_count = 0
        for item in news_list:
            if isinstance(item, dict):
                articles = item.get("articles", [])
                for article in articles:
                    if isinstance(article, dict):
                        title = article.get("title", "News Article")
                        if title:
                            article_count += 1
                            display_title = (
                                title[:100] + "..." if len(title) > 100 else title
                            )

                            with st.expander(f"📄 {display_title}", expanded=False):
                                content = article.get("content", "No content available")
                                st.write(
                                    content[:1000] + "..."
                                    if len(content) > 1000
                                    else content
                                )

                                url = article.get("url")
                                if url:
                                    st.markdown(f"[🔗 Read Full Article]({url})")

        if article_count == 0:
            st.info("No news articles available for display")

    except json.JSONDecodeError:
        st.warning("Could not parse news data")
    except Exception as e:
        st.warning(f"Error displaying news: {e}")


def display_report(report: str):
    """Display the final analysis report."""
    if not report:
        st.warning("No report generated")
        return

    # Display the markdown report
    st.markdown(report)

    # Raw markdown view
    with st.expander("📄 View Raw Markdown", expanded=False):
        st.code(report, language="markdown")


def display_export_options(result: Dict[str, Any], ticker: str):
    """Display export options for the analysis results."""
    st.markdown("### 📥 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        report = result.get("final_report")
        if report:
            st.download_button(
                label="📄 Download Report (MD)",
                data=report,
                file_name=f"{ticker}_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.button("📄 Report N/A", disabled=True, use_container_width=True)

    with col2:
        financial_data = result.get("financial_data")
        if financial_data:
            try:
                json_data = safe_json_serialize(financial_data)
                st.download_button(
                    label="📊 Download Data (JSON)",
                    data=json_data,
                    file_name=f"{ticker}_financial_data.json",
                    mime="application/json",
                    use_container_width=True,
                )
            except Exception:
                st.button("📊 JSON N/A", disabled=True, use_container_width=True)
        else:
            st.button("📊 JSON N/A", disabled=True, use_container_width=True)

    with col3:
        fin_data = result.get("financial_data", {})
        if isinstance(fin_data, dict) and fin_data.get("data"):
            try:
                df = pd.DataFrame(fin_data["data"])
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📈 Download Data (CSV)",
                    data=csv_data,
                    file_name=f"{ticker}_financial_data.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception:
                st.button("📈 CSV N/A", disabled=True, use_container_width=True)
        else:
            st.button("📈 CSV N/A", disabled=True, use_container_width=True)


def display_debug_info(result: Dict[str, Any]):
    """Display debug information."""
    with st.expander("🔧 Debug Information", expanded=False):
        debug_data = {
            "has_financial_data": result.get("financial_data") is not None,
            "has_price_data": result.get("price_data") is not None,
            "has_news_data": result.get("news_data") is not None,
            "has_analysis_results": bool(result.get("analysis_results")),
            "has_final_report": bool(result.get("final_report")),
            "iteration_count": result.get("iteration_count", 0),
            "total_tool_calls": result.get("total_tool_calls", 0),
            "tool_call_counts": result.get("tool_call_counts", {}),
            "review_status": result.get("review_status"),
            "current_phase": result.get("current_phase"),
            "current_node": result.get("current_node"),
        }
        st.json(debug_data)


###############################################################################
# Main Application
###############################################################################


def run_analysis(
    user_request: str, progress_bar, status_text
) -> Optional[Dict[str, Any]]:
    """Run the analysis and return results."""
    try:
        status_text.text("🔄 Initializing agent...")
        progress_bar.progress(10)

        # Compile the graph
        app = compile_graph(save_visualization=False)

        status_text.text("📋 Creating analysis plan...")
        progress_bar.progress(20)

        # Initialize state
        initial_state = init_state(user_request)

        status_text.text("🔍 Executing analysis pipeline...")
        progress_bar.progress(30)

        # Run the agent
        result = app.invoke(initial_state, config={"recursion_limit": 200})

        progress_bar.progress(90)
        status_text.text("📝 Finalizing report...")

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        return result

    except Exception as e:
        st.error(f"❌ Analysis failed:  {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None


def main():
    """Main application entry point."""

    # Initialize session state first
    init_session_state()

    # Display header
    display_header()

    # Display sidebar (this handles example query clicks)
    display_sidebar()

    # Main content area - Query input section
    user_query = display_query_input()

    # Validate query
    if not user_query or len(user_query.strip()) < 10:
        st.info(
            "👆 Enter your analysis request above to get started.  Be specific about the ticker, time period, and what you want to analyze."
        )

        # Show some guidance
        st.markdown("---")
        st.markdown("### 🚀 Getting Started")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                """
            **What can this agent do?**
            - 📊 Fetch and analyze financial statements
            - 📈 Track stock price performance
            - 📰 Gather relevant news and context
            - 🔍 Compute financial ratios and metrics
            - 📝 Generate professional investment reports
            """
            )

        with col2:
            st.markdown(
                """
            **Supported analysis types:**
            - Profitability analysis (margins, returns)
            - Growth metrics (revenue, EPS, CAGR)
            - Balance sheet health (leverage, liquidity)
            - Cash flow analysis
            - Risk assessment
            - News-driven context for anomalies
            """
            )

        return

    # Extract ticker for display purposes
    detected_ticker = extract_ticker_from_query(user_query)

    # Display detected ticker
    st.markdown("---")
    if detected_ticker != "UNKNOWN":
        st.success(f"🎯 **Detected Ticker:** `{detected_ticker}`")
    else:
        st.warning(
            "⚠️ Could not detect a stock ticker in your query.  Please include a ticker symbol (e.g., NVDA, AAPL, MSFT)."
        )

    # Run Analysis Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_button = st.button(
            "🚀 Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=(detected_ticker == "UNKNOWN"),
        )

    if run_button:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Run analysis
        result = run_analysis(user_query, progress_bar, status_text)

        if result:
            # Store results in session state
            st.session_state.analysis_result = result
            st.session_state.ticker = detected_ticker
            st.session_state.run_complete = True
            st.session_state.last_query = user_query

            # Clear progress
            progress_bar.empty()
            status_text.empty()

            st.success(f"✅ Analysis completed successfully for **{detected_ticker}**!")
            st.rerun()

    # Display Results if available
    if st.session_state.analysis_result is not None:
        result = st.session_state.analysis_result
        display_ticker = st.session_state.ticker or "Unknown"
        last_query = st.session_state.last_query

        st.divider()

        # Show the query that was analyzed
        if last_query:
            st.markdown("**📝 Analyzed Query:**")
            st.info(last_query)

        # Data status
        st.markdown(f"## 📈 Results for {display_ticker}")
        display_data_status(result)
        display_execution_stats(result)

        st.divider()

        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📋 Report",
                "📊 Financial Data",
                "📈 Price Analysis",
                "🔍 Analysis Details",
                "📰 News",
            ]
        )

        with tab1:
            st.markdown("### 📋 Analysis Report")
            display_report(result.get("final_report", ""))

        with tab2:
            st.markdown("### 📊 Financial Data")
            display_financial_data(result.get("financial_data", {}))

        with tab3:
            st.markdown("### 📈 Price Analysis")
            display_price_data(result.get("price_data", {}))

        with tab4:
            st.markdown("### 🔍 Detailed Analysis")
            display_analysis_results(result.get("analysis_results", {}))

        with tab5:
            st.markdown("### 📰 News & Events")
            display_news(result.get("news_data"))

        st.divider()

        # Export options
        display_export_options(result, display_ticker)

        # Debug info
        display_debug_info(result)


if __name__ == "__main__":
    main()
