"""
Research tools for the Financial Analysis Agent.
Contains tools for fetching financial data, price history, and news.
"""

import logging
import re
import traceback
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool

from Utilities import df_to_records

logger = logging.getLogger(__name__)


###############################################################################
# Module-Private Constants (only used in this file)
###############################################################################

_INCOME_STATEMENT_MAPPINGS: Dict[str, str] = {
    "Total Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "Operating Income": "operating_income",
    "Net Income": "net_income",
    "Diluted EPS": "eps",
    "EBITDA": "ebitda",
}

_CASHFLOW_MAPPINGS: Dict[str, str] = {
    "Operating Cash Flow": "cfo",
    "Capital Expenditure": "capex",
    "Free Cash Flow": "fcf",
}

_BALANCE_SHEET_MAPPINGS: Dict[str, str] = {
    "Total Assets": "total_assets",
    "Stockholders Equity": "equity",
    "Total Equity Gross Minority Interest": "equity",
    "Long Term Debt": "long_term_debt",
    "Current Debt": "short_term_debt",
    "Cash And Cash Equivalents": "cash",
    "Current Assets": "current_assets",
    "Current Liabilities": "current_liabilities",
}

_QUARTERS: List[tuple] = [("Q1", 1, 3), ("Q2", 4, 6), ("Q3", 7, 9), ("Q4", 10, 12)]


###############################################################################
# Module-Private Helper Functions (only used in this file)
###############################################################################


def _normalize_financial_df(df_input: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize a financial DataFrame by transposing and setting year as index."""
    if df_input is None or df_input.empty:
        return pd.DataFrame()

    df = df_input.T.copy()
    df.index = pd.to_datetime(df.index).year
    df = df[~df.index.duplicated(keep="first")]
    return df


def _apply_column_mappings(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    mappings: Dict[str, str],
    skip_existing: bool = False,
) -> pd.DataFrame:
    """Apply column mappings from source DataFrame to target DataFrame."""
    for orig_col, new_col in mappings.items():
        if orig_col in source_df.columns:
            if skip_existing and new_col in target_df.columns:
                continue
            target_df[new_col] = pd.to_numeric(source_df[orig_col], errors="coerce")
    return target_df


def _calculate_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate financial ratios and add them to the DataFrame."""
    # Total debt
    long_term = df.get("long_term_debt", pd.Series(0, index=df.index)).fillna(0)
    short_term = df.get("short_term_debt", pd.Series(0, index=df.index)).fillna(0)
    df["total_debt"] = long_term + short_term

    # Margin ratios
    if "revenue" in df.columns:
        revenue = df["revenue"].replace(0, pd.NA)
        if "gross_profit" in df.columns:
            df["gross_margin"] = (df["gross_profit"] / revenue).astype(float)
        if "operating_income" in df.columns:
            df["operating_margin"] = (df["operating_income"] / revenue).astype(float)
        if "net_income" in df.columns:
            df["net_margin"] = (df["net_income"] / revenue).astype(float)

    # Liquidity ratio
    if "current_assets" in df.columns and "current_liabilities" in df.columns:
        current_liab = df["current_liabilities"].replace(0, pd.NA)
        df["current_ratio"] = (df["current_assets"] / current_liab).astype(float)

    # Leverage ratio
    if "total_debt" in df.columns and "equity" in df.columns:
        equity = df["equity"].replace(0, pd.NA)
        df["debt_to_equity"] = (df["total_debt"] / equity).astype(float)

    return df


def _parse_search_results(results: str) -> Dict[str, List[str]]:
    """Parse DuckDuckGo search results to extract URLs, titles, and snippets."""
    return {
        "urls": re.findall(r"link:\s*(https?://[^\s,\]]+)", results),
        "titles": re.findall(r"title:\s*([^,\]]+)", results),
        "snippets": re.findall(r"snippet:\s*([^,\]]+)", results),
    }


def _load_articles_from_urls(
    urls: List[str],
    titles: List[str],
    snippets: List[str],
    max_urls: int = 2,
    max_content_length: int = 1500,
) -> List[Dict[str, str]]:
    """Load article content from URLs with fallback to snippets."""
    articles = []
    urls_to_load = urls[:max_urls]

    if not urls_to_load:
        return articles

    try:
        docs = WebBaseLoader(urls_to_load).load()
        for i, doc in enumerate(docs):
            articles.append(
                {
                    "url": urls[i],
                    "title": titles[i] if i < len(titles) else "Unknown",
                    "content": " ".join(doc.page_content.split())[:max_content_length],
                }
            )
    except Exception as e:
        logger.warning(f"Failed to load articles from URLs:  {e}")
        for i in range(min(len(urls_to_load), len(titles), len(snippets))):
            articles.append(
                {"url": urls[i], "title": titles[i], "content": snippets[i]}
            )

    return articles


###############################################################################
# Research Tools
###############################################################################


@tool
def fetch_financials(ticker_symbol: str) -> Dict[str, Any]:
    """
    Fetch annual financials including income statement, balance sheet, and cash flow.

    Args:
        ticker_symbol: Stock ticker symbol (e. g., 'NVDA', 'AAPL')

    Returns:
        Dictionary with ticker, success status, data records, years, and error if any
    """
    try:
        ticker = yf.Ticker(ticker_symbol)

        income_stmt = getattr(ticker, "income_stmt", None)
        balance_sheet = getattr(ticker, "balance_sheet", None)
        cashflow = getattr(ticker, "cashflow", None)

        if income_stmt is None or income_stmt.empty:
            logger.warning(f"No financial data available for {ticker_symbol}")
            return {
                "ticker": ticker_symbol,
                "success": False,
                "error": "No financial data available",
                "data": [],
                "years": [],
            }

        # Normalize DataFrames
        inc = _normalize_financial_df(income_stmt)
        bs = _normalize_financial_df(balance_sheet)
        cf = _normalize_financial_df(cashflow)

        # Build combined DataFrame
        df = pd.DataFrame(index=inc.index)
        df = _apply_column_mappings(df, inc, _INCOME_STATEMENT_MAPPINGS)
        df = _apply_column_mappings(df, cf, _CASHFLOW_MAPPINGS)
        df = _apply_column_mappings(df, bs, _BALANCE_SHEET_MAPPINGS, skip_existing=True)

        # Calculate ratios
        df = _calculate_financial_ratios(df)

        # Clean up
        df = df.sort_index()
        df = df.replace([float("inf"), float("-inf")], None)

        # Convert to records (using shared utility)
        records = df_to_records(df)

        logger.info(
            f"Successfully fetched financials for {ticker_symbol}:  {len(records)} years"
        )

        return {
            "ticker": ticker_symbol,
            "success": True,
            "data": records,
            "years": [int(y) for y in df.index.tolist()],
        }

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error fetching financials for {ticker_symbol}:  {error_msg}")
        return {
            "ticker": ticker_symbol,
            "success": False,
            "error": error_msg,
            "data": [],
            "years": [],
        }


@tool
def fetch_price_history(
    ticker: str, years: List[int], price_change_threshold: float = 10.0
) -> Dict[str, Any]:
    """
    Fetch quarterly stock price data and detect significant price movements.

    Args:
        ticker: Stock ticker symbol
        years: List of years to fetch data for
        price_change_threshold: Minimum percentage change to flag as significant

    Returns:
        Dictionary with price data, quarterly breakdowns, and significant movements
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        today = pd.Timestamp.now()
        periods_data = []
        all_quarterly_data = []

        # Normalize years input
        if isinstance(years, int):
            years = [years]
        years = sorted(set(years))

        for year in years:
            year_quarters = []

            for q_name, start_month, end_month in _QUARTERS:
                try:
                    start = pd.Timestamp(year=year, month=start_month, day=1)

                    if end_month == 12:
                        end = pd.Timestamp(year=year, month=12, day=31)
                    else:
                        end = pd.Timestamp(
                            year=year, month=end_month + 1, day=1
                        ) - pd.Timedelta(days=1)

                    if start > today:
                        continue

                    end = min(end, today)

                    hist = ticker_obj.history(
                        start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d")
                    )

                    if hist.empty:
                        continue

                    hist = hist.reset_index()
                    hist["Date"] = pd.to_datetime(hist["Date"])

                    if hist["Date"].dt.tz is not None:
                        hist["Date"] = hist["Date"].dt.tz_localize(None)

                    q_open = float(hist.iloc[0]["Open"])
                    q_close = float(hist.iloc[-1]["Close"])
                    q_high = float(hist["High"].max())
                    q_low = float(hist["Low"].min())
                    q_volume = int(hist["Volume"].sum())
                    avg_volume = int(hist["Volume"].mean())
                    q_return = (
                        round(((q_close - q_open) / q_open * 100), 2)
                        if q_open != 0
                        else 0.0
                    )
                    volatility = round(float(hist["Close"].pct_change().std() * 100), 2)

                    quarter_data = {
                        "quarter": q_name,
                        "year": year,
                        "period": f"{year}-{q_name}",
                        "start_date": start.strftime("%Y-%m-%d"),
                        "end_date": end.strftime("%Y-%m-%d"),
                        "open": round(q_open, 2),
                        "close": round(q_close, 2),
                        "high": round(q_high, 2),
                        "low": round(q_low, 2),
                        "quarter_return_pct": q_return,
                        "total_volume": q_volume,
                        "avg_daily_volume": avg_volume,
                        "volatility_pct": volatility,
                        "trading_days": len(hist),
                    }

                    year_quarters.append(quarter_data)
                    all_quarterly_data.append(quarter_data)

                except Exception as e:
                    logger.warning(f"Error fetching {year} {q_name} for {ticker}: {e}")
                    continue

            if year_quarters:
                year_open = year_quarters[0]["open"]
                year_close = year_quarters[-1]["close"]
                year_high = max(q["high"] for q in year_quarters)
                year_low = min(q["low"] for q in year_quarters)
                year_return = (
                    round(((year_close - year_open) / year_open * 100), 2)
                    if year_open != 0
                    else 0.0
                )
                total_volume = sum(q["total_volume"] for q in year_quarters)

                periods_data.append(
                    {
                        "year": year,
                        "quarters": year_quarters,
                        "year_summary": {
                            "year_open": round(year_open, 2),
                            "year_close": round(year_close, 2),
                            "year_high": round(year_high, 2),
                            "year_low": round(year_low, 2),
                            "ytd_return_pct": year_return,
                            "total_volume": total_volume,
                            "quarters_available": len(year_quarters),
                        },
                    }
                )

        # Significant movements
        significant_quarters = [
            {
                "period": q["period"],
                "return_pct": q["quarter_return_pct"],
                "direction": "up" if q["quarter_return_pct"] > 0 else "down",
                "open": q["open"],
                "close": q["close"],
                "volatility": q["volatility_pct"],
            }
            for q in all_quarterly_data
            if abs(q["quarter_return_pct"]) >= price_change_threshold
        ]
        significant_quarters.sort(key=lambda x: abs(x["return_pct"]), reverse=True)

        # QoQ changes
        qoq_changes = []
        for i in range(1, len(all_quarterly_data)):
            prev = all_quarterly_data[i - 1]
            curr = all_quarterly_data[i]
            if prev["close"] != 0:
                qoq_change = round(
                    ((curr["close"] - prev["close"]) / prev["close"] * 100), 2
                )
                qoq_changes.append(
                    {
                        "from": prev["period"],
                        "to": curr["period"],
                        "change_pct": qoq_change,
                        "direction": "up" if qoq_change > 0 else "down",
                    }
                )

        if not periods_data:
            logger.warning(f"No price data found for {ticker} in years {years}")
            return {
                "ticker": ticker,
                "success": False,
                "error": "No price data found for specified years",
                "periods": [],
                "quarterly_data": [],
                "significant_quarters": [],
                "qoq_changes": [],
            }

        logger.info(
            f"Successfully fetched price history for {ticker}: {len(all_quarterly_data)} quarters"
        )

        return {
            "ticker": ticker,
            "success": True,
            "periods": periods_data,
            "quarterly_data": all_quarterly_data,
            "significant_quarters": significant_quarters[:10],
            "qoq_changes": qoq_changes,
        }

    except Exception as e:
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        logger.error(f"Error fetching price history for {ticker}: {error_msg}")
        return {
            "ticker": ticker,
            "success": False,
            "error": error_msg,
            "periods": [],
            "quarterly_data": [],
            "significant_quarters": [],
            "qoq_changes": [],
        }


@tool
def fetch_news_for_date(
    ticker: str, event_date: str, direction: str = "any"
) -> Dict[str, Any]:
    """
    Fetch news for a specific date to explain significant anomalies in price or financial metrics.

    Use when analysis reveals unexplained movements:
        - Stock price: >5% or <5% single-day movement
        - Stock price: >40% or <-40% in one quarter
        - Margins (gross/operating/net): >300bps YoY change
        - Revenue: >20% YoY change outside trend
        - Debt/Capex/Cash flow: Major unexpected shifts

    Args:
        ticker: Stock symbol (e.g., 'AAPL')
        event_date: Target date (YYYY-MM-DD). Use fiscal period end for metric changes.
        direction: 'up' (positive), 'down' (negative), or 'any' (default)

    Returns:
        dict: {ticker, event_date, success, articles}

    Examples:
        >>> fetch_news_for_date('AAPL', '2024-08-05', 'down')  # Price crash
        >>> fetch_news_for_date('NVDA', '2024-12-31', 'up')    # Margin expansion
    """
    try:
        direction_words = {"up": "surge", "down": "drop", "any": "news"}
        direction_word = direction_words.get(direction.lower(), "news")

        query = f"{ticker} stock {direction_word} {event_date}"
        search = DuckDuckGoSearchResults(backend="news", num_results=3)
        results = search.invoke(query)

        parsed = _parse_search_results(results)
        articles = _load_articles_from_urls(
            urls=parsed["urls"],
            titles=parsed["titles"],
            snippets=parsed["snippets"],
            max_urls=2,
            max_content_length=1500,
        )

        logger.info(
            f"Fetched {len(articles)} news articles for {ticker} on {event_date}"
        )

        return {
            "ticker": ticker,
            "event_date": event_date,
            "success": True,
            "articles": articles,
        }

    except Exception as e:
        logger.error(f"Error fetching news for {ticker} on {event_date}: {e}")
        return {
            "ticker": ticker,
            "event_date": event_date,
            "success": False,
            "error": str(e),
            "articles": [],
        }


@tool
def fetch_general_news(ticker: str, topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch recent general news about a company.

    Args:
        ticker: Stock ticker symbol
        topic: Optional specific topic to search for

    Returns:
        Dictionary with ticker, success status, and articles
    """
    try:
        query = f"{ticker} stock {topic}" if topic else f"{ticker} stock"

        search = DuckDuckGoSearchResults(backend="news", num_results=5)
        results = search.invoke(query)

        parsed = _parse_search_results(results)

        if not parsed["urls"]:
            logger.warning(f"No news found for {ticker}")
            return {
                "ticker": ticker,
                "success": False,
                "error": "No news articles found",
                "articles": [],
            }

        articles = []
        urls_to_load = parsed["urls"][:3]

        try:
            docs = WebBaseLoader(urls_to_load).load()
            for i, doc in enumerate(docs):
                articles.append(
                    {
                        "url": parsed["urls"][i],
                        "title": doc.metadata.get("title", "Unknown"),
                        "content": " ".join(doc.page_content.split())[:2000],
                    }
                )
        except Exception as e:
            logger.error(f"Error loading news articles: {e}")
            return {"ticker": ticker, "success": False, "error": str(e), "articles": []}

        logger.info(f"Fetched {len(articles)} general news articles for {ticker}")

        return {"ticker": ticker, "success": True, "articles": articles}

    except Exception as e:
        logger.error(f"Error fetching general news for {ticker}: {e}")
        return {"ticker": ticker, "success": False, "error": str(e), "articles": []}


###############################################################################
# Tool Collection Export
###############################################################################

research_tools = [
    fetch_financials,
    fetch_price_history,
    fetch_news_for_date,
    fetch_general_news,
]
