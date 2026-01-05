"""
Analysis tools for the Financial Analysis Agent.
Contains tools for computing financial metrics, ratios, and statistical analysis.
"""

import logging
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.tools import tool

# Import shared utilities
from Utilities import to_dataframe

# Setup logging
logger = logging.getLogger(__name__)


###############################################################################
# Module-Private Helper Functions
###############################################################################


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Safely convert a DataFrame column to numeric, dropping NaN values.

    Args:
        df: Source DataFrame
        column:  Column name to convert

    Returns:
        Numeric Series with NaN values dropped
    """
    return pd.to_numeric(df[column], errors="coerce").dropna()


def _calculate_trend(series: pd.Series, threshold: float = 0.05) -> str:
    """
    Determine the trend direction of a series.

    Args:
        series:  Numeric series to analyze
        threshold: Percentage threshold for determining trend (default 5%)

    Returns:
        Trend direction: 'increasing', 'decreasing', or 'stable'
    """
    if len(series) < 2:
        return "insufficient_data"

    first_val = series.iloc[0]
    last_val = series.iloc[-1]

    if first_val == 0:
        return "stable" if last_val == 0 else "increasing"

    change_ratio = (last_val - first_val) / abs(first_val)

    if change_ratio > threshold:
        return "increasing"
    elif change_ratio < -threshold:
        return "decreasing"
    else:
        return "stable"


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Safely divide two series, handling division by zero and infinity.

    Args:
        numerator: Numerator series
        denominator: Denominator series

    Returns:
        Result series with inf values replaced by NA
    """
    result = numerator / denominator
    return result.replace([float("inf"), float("-inf")], pd.NA)


def _series_to_year_dict(
    series: pd.Series, decimals: int = 2, include_none: bool = True
) -> Dict[int, Optional[float]]:
    """
    Convert a pandas Series with year index to a dictionary.

    Args:
        series:  Series with year index
        decimals: Number of decimal places for rounding
        include_none: Whether to include None for NA values

    Returns:
        Dictionary mapping years to values
    """
    result = {}
    for year, value in series.items():
        year_int = int(year)
        if pd.notna(value):
            result[year_int] = round(float(value), decimals)
        elif include_none:
            result[year_int] = None
    return result


###############################################################################
# Analysis Tools
###############################################################################


@tool
def compute_yoy_changes(
    data: List[Dict[str, Any]], columns: List[str]
) -> Dict[str, Any]:
    """
    Compute year-over-year percentage changes for specified columns.

    Args:
        data:  List of yearly financial records (must include 'year' field)
        columns: List of column names to compute YoY changes for

    Returns:
        Dictionary containing:
        - success: Whether computation was successful
        - results: Dict mapping column names to their YoY changes and values
        - error: Error message if failed
    """
    try:
        df = to_dataframe(data)

        if df.empty:
            logger.warning("compute_yoy_changes: Empty DataFrame provided")
            return {"success": False, "error": "No data provided"}

        if not columns:
            logger.warning("compute_yoy_changes:  No columns specified")
            return {"success": False, "error": "No columns specified"}

        results = {}
        columns_processed = 0

        for col in columns:
            if col not in df.columns:
                logger.debug(f"Column '{col}' not found in data, skipping")
                continue

            series = _safe_numeric_series(df, col)

            if series.empty:
                logger.debug(f"Column '{col}' has no valid numeric data, skipping")
                continue

            # Calculate percentage change
            pct_change = series.pct_change() * 100

            results[col] = {
                "yoy_changes": _series_to_year_dict(pct_change.dropna(), decimals=2),
                "values": _series_to_year_dict(series, decimals=2),
                "avg_yoy_change": (
                    round(float(pct_change.mean()), 2)
                    if not pct_change.dropna().empty
                    else None
                ),
            }
            columns_processed += 1

        if columns_processed == 0:
            return {"success": False, "error": "No valid columns found in data"}

        logger.info(f"Computed YoY changes for {columns_processed} columns")

        return {
            "success": True,
            "results": results,
            "columns_processed": columns_processed,
        }

    except Exception as e:
        logger.error(f"Error in compute_yoy_changes: {e}")
        return {"success": False, "error": str(e)}


@tool
def compute_financial_ratios(
    data: List[Dict[str, Any]], ratio_configs: List[Dict[str, str]]
) -> Dict[str, Any]:
    """
    Compute custom financial ratios from the provided data.

    Args:
        data: List of yearly financial records
        ratio_configs: List of ratio configurations, each containing:
            - name: Name for the ratio (e.g., 'ROE', 'ROA')
            - numerator: Column name for numerator
            - denominator:  Column name for denominator

    Returns:
        Dictionary containing:
        - success: Whether computation was successful
        - ratios: Dict mapping ratio names to yearly values
        - error: Error message if failed

    Example ratio_configs:
        [
            {"name":  "ROE", "numerator": "net_income", "denominator": "equity"},
            {"name": "ROA", "numerator": "net_income", "denominator": "total_assets"}
        ]
    """
    try:
        df = to_dataframe(data)

        if df.empty:
            logger.warning("compute_financial_ratios:  Empty DataFrame provided")
            return {"success": False, "error": "No data provided"}

        if not ratio_configs:
            logger.warning("compute_financial_ratios: No ratio configs provided")
            return {"success": False, "error": "No ratio configurations provided"}

        results = {}
        ratios_computed = 0
        skipped_ratios = []

        for cfg in ratio_configs:
            name = cfg.get("name")
            numerator_col = cfg.get("numerator")
            denominator_col = cfg.get("denominator")

            # Validate config
            if not all([name, numerator_col, denominator_col]):
                logger.warning(f"Invalid ratio config: {cfg}")
                skipped_ratios.append(
                    {"config": cfg, "reason": "missing required fields"}
                )
                continue

            # Check columns exist
            if numerator_col not in df.columns:
                skipped_ratios.append(
                    {"name": name, "reason": f"numerator '{numerator_col}' not found"}
                )
                continue

            if denominator_col not in df.columns:
                skipped_ratios.append(
                    {
                        "name": name,
                        "reason": f"denominator '{denominator_col}' not found",
                    }
                )
                continue

            # Compute ratio
            numerator = pd.to_numeric(df[numerator_col], errors="coerce")
            denominator = pd.to_numeric(df[denominator_col], errors="coerce")

            ratio = _safe_divide(numerator, denominator)

            results[name] = {
                "values": _series_to_year_dict(ratio, decimals=4),
                "avg": (
                    round(float(ratio.mean()), 4) if not ratio.dropna().empty else None
                ),
                "trend": _calculate_trend(ratio.dropna()),
            }
            ratios_computed += 1

        if ratios_computed == 0:
            return {
                "success": False,
                "error": "No ratios could be computed",
                "skipped": skipped_ratios,
            }

        logger.info(f"Computed {ratios_computed} financial ratios")

        return {
            "success": True,
            "ratios": results,
            "ratios_computed": ratios_computed,
            "skipped": skipped_ratios if skipped_ratios else None,
        }

    except Exception as e:
        logger.error(f"Error in compute_financial_ratios: {e}")
        return {"success": False, "error": str(e)}


@tool
def compute_growth_metrics(
    data: List[Dict[str, Any]],
    column: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute growth metrics including CAGR and total growth for a column.

    Args:
        data: List of yearly financial records
        column:  Column name to compute growth for
        start_year: Starting year (defaults to earliest year in data)
        end_year:  Ending year (defaults to latest year in data)

    Returns:
        Dictionary containing:
        - success: Whether computation was successful
        - column: The analyzed column name
        - start_year, end_year: The year range used
        - start_value, end_value:  Values at start and end
        - total_growth_pct: Total percentage growth
        - cagr_pct: Compound Annual Growth Rate
        - periods: Number of years in the range
        - error: Error message if failed
    """
    try:
        df = to_dataframe(data)

        if df.empty:
            logger.warning("compute_growth_metrics: Empty DataFrame provided")
            return {"success": False, "error": "No data provided"}

        if column not in df.columns:
            logger.warning(f"compute_growth_metrics:  Column '{column}' not found")
            return {"success": False, "error": f"Column '{column}' not found in data"}

        # Get available years
        available_years = sorted(df.index.tolist())

        if len(available_years) < 2:
            return {
                "success": False,
                "error": "Need at least 2 years of data for growth calculation",
            }

        # Determine year range
        calc_start_year = (
            start_year if start_year is not None else int(available_years[0])
        )
        calc_end_year = end_year if end_year is not None else int(available_years[-1])

        # Validate years exist in data
        if calc_start_year not in df.index:
            return {
                "success": False,
                "error": f"Start year {calc_start_year} not in data.  Available:  {available_years}",
            }

        if calc_end_year not in df.index:
            return {
                "success": False,
                "error": f"End year {calc_end_year} not in data. Available: {available_years}",
            }

        if calc_start_year >= calc_end_year:
            return {"success": False, "error": "Start year must be before end year"}

        # Get values
        start_val = pd.to_numeric(df.loc[calc_start_year, column], errors="coerce")
        end_val = pd.to_numeric(df.loc[calc_end_year, column], errors="coerce")

        # Validate values
        if pd.isna(start_val) or pd.isna(end_val):
            return {
                "success": False,
                "error": "Start or end value is not a valid number",
            }

        if start_val <= 0:
            return {
                "success": False,
                "error": f"Start value ({start_val}) must be positive for growth calculation",
            }

        # Calculate metrics
        periods = calc_end_year - calc_start_year
        total_growth = ((end_val - start_val) / start_val) * 100
        cagr = ((end_val / start_val) ** (1 / periods) - 1) * 100

        # Calculate average annual growth
        series = _safe_numeric_series(df.loc[calc_start_year:calc_end_year], column)
        avg_annual_growth = (
            series.pct_change().mean() * 100 if len(series) > 1 else None
        )

        logger.info(f"Computed growth metrics for {column}:  CAGR={cagr:.2f}%")

        return {
            "success": True,
            "column": column,
            "start_year": calc_start_year,
            "end_year": calc_end_year,
            "start_value": round(float(start_val), 2),
            "end_value": round(float(end_val), 2),
            "periods": periods,
            "total_growth_pct": round(float(total_growth), 2),
            "cagr_pct": round(float(cagr), 2),
            "avg_annual_growth_pct": (
                round(float(avg_annual_growth), 2) if avg_annual_growth else None
            ),
        }

    except Exception as e:
        logger.error(f"Error in compute_growth_metrics: {e}")
        return {"success": False, "error": str(e)}


@tool
def compute_summary_statistics(
    data: List[Dict[str, Any]], columns: List[str]
) -> Dict[str, Any]:
    """
    Compute summary statistics (mean, min, max, std, trend) for specified columns.

    Args:
        data: List of yearly financial records
        columns: List of column names to compute statistics for

    Returns:
        Dictionary containing:
        - success: Whether computation was successful
        - statistics: Dict mapping column names to their statistics
        - error:  Error message if failed
    """
    try:
        df = to_dataframe(data)

        if df.empty:
            logger.warning("compute_summary_statistics: Empty DataFrame provided")
            return {"success": False, "error": "No data provided"}

        if not columns:
            logger.warning("compute_summary_statistics: No columns specified")
            return {"success": False, "error": "No columns specified"}

        results = {}
        columns_processed = 0

        for col in columns:
            if col not in df.columns:
                logger.debug(f"Column '{col}' not found in data, skipping")
                continue

            series = _safe_numeric_series(df, col)

            if series.empty:
                logger.debug(f"Column '{col}' has no valid numeric data, skipping")
                continue

            # Calculate statistics
            stats = {
                "count": int(len(series)),
                "mean": round(float(series.mean()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "std": round(float(series.std()), 4) if len(series) > 1 else 0.0,
                "median": round(float(series.median()), 4),
                "trend": _calculate_trend(series),
                "first_year": int(series.index.min()),
                "last_year": int(series.index.max()),
                "first_value": round(float(series.iloc[0]), 4),
                "last_value": round(float(series.iloc[-1]), 4),
            }

            results[col] = stats
            columns_processed += 1

        if columns_processed == 0:
            return {"success": False, "error": "No valid columns found in data"}

        logger.info(f"Computed summary statistics for {columns_processed} columns")

        return {
            "success": True,
            "statistics": results,
            "columns_processed": columns_processed,
        }

    except Exception as e:
        logger.error(f"Error in compute_summary_statistics: {e}")
        return {"success": False, "error": str(e)}


@tool
def compare_metrics_across_years(
    data: List[Dict[str, Any]], columns: List[str], years: List[int]
) -> Dict[str, Any]:
    """
    Create a comparison table of metrics across specified years.

    Args:
        data: List of yearly financial records
        columns: List of column names to compare
        years: List of years to include in comparison

    Returns:
        Dictionary containing:
        - success: Whether computation was successful
        - comparison: Dict mapping column names to year-value dicts
        - year_over_year: YoY changes between consecutive years
        - error: Error message if failed
    """
    try:
        df = to_dataframe(data)

        if df.empty:
            logger.warning("compare_metrics_across_years: Empty DataFrame provided")
            return {"success": False, "error": "No data provided"}

        if not columns:
            logger.warning("compare_metrics_across_years: No columns specified")
            return {"success": False, "error": "No columns specified"}

        if not years:
            logger.warning("compare_metrics_across_years: No years specified")
            return {"success": False, "error": "No years specified"}

        # Sort years for consistent output
        sorted_years = sorted(years)

        comparison = {}
        yoy_changes = {}
        columns_processed = 0

        for col in columns:
            if col not in df.columns:
                logger.debug(f"Column '{col}' not found in data, skipping")
                continue

            col_data = {}
            col_yoy = {}
            prev_value = None
            prev_year = None

            for year in sorted_years:
                if year in df.index:
                    value = df.loc[year, col]
                    if pd.notna(value):
                        current_value = round(float(value), 4)
                        col_data[year] = current_value

                        # Calculate YoY change
                        if prev_value is not None and prev_value != 0:
                            yoy_pct = (
                                (current_value - prev_value) / abs(prev_value)
                            ) * 100
                            col_yoy[f"{prev_year}_to_{year}"] = round(yoy_pct, 2)

                        prev_value = current_value
                        prev_year = year
                    else:
                        col_data[year] = None
                else:
                    col_data[year] = None

            comparison[col] = col_data
            if col_yoy:
                yoy_changes[col] = col_yoy
            columns_processed += 1

        if columns_processed == 0:
            return {"success": False, "error": "No valid columns found in data"}

        # Check which years were found
        available_years = [y for y in sorted_years if y in df.index]
        missing_years = [y for y in sorted_years if y not in df.index]

        logger.info(
            f"Compared {columns_processed} metrics across {len(available_years)} years"
        )

        return {
            "success": True,
            "comparison": comparison,
            "year_over_year_changes": yoy_changes if yoy_changes else None,
            "years_requested": sorted_years,
            "years_available": available_years,
            "years_missing": missing_years if missing_years else None,
            "columns_processed": columns_processed,
        }

    except Exception as e:
        logger.error(f"Error in compare_metrics_across_years: {e}")
        return {"success": False, "error": str(e)}


###############################################################################
# Tool Collection Export
###############################################################################

analysis_tools = [
    compute_yoy_changes,
    compute_financial_ratios,
    compute_growth_metrics,
    compute_summary_statistics,
    compare_metrics_across_years,
]
