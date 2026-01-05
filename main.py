"""
Financial Analysis Agent - Main Entry Point

A LangGraph-based multi-agent system for comprehensive financial analysis.
Uses structured output for deterministic routing and tool-calling agents for data collection and analysis.
"""

import logging
import os

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

# Import from local modules
from Constants import ALL_TOOL_NAMES, MAX_CALLS_PER_TOOL, MAX_ITERATIONS
from Nodes import (analysis_tools_node, analyst_node, planner_node,
                   research_tools_node, researcher_node, reviewer_node,
                   route_after_research_tools, route_after_review, writer_node)
from Schemas import AgentState
from Utilities import init_state

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


###############################################################################
# Configuration
###############################################################################

AUTH_CODE = os.getenv("AUTH_CODE")
BASE_URL = os.getenv("BASE_URL")


def validate_config() -> bool:
    """Validate required configuration is present."""
    if not AUTH_CODE:
        logger.warning("AUTH_CODE not set in environment")
        return False
    if not BASE_URL:
        logger.warning("BASE_URL not set in environment")
        return False
    return True


###############################################################################
# Graph Builder
###############################################################################


def build_graph() -> StateGraph:
    """
    Build the LangGraph state graph for the financial analysis agent.

    Key improvement: When reviewer requests more research (e.g., news),
    the flow goes: reviewer → researcher → research_tools → reviewer
    WITHOUT going through analyst unnecessarily.

    Returns:
        Configured StateGraph instance
    """
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("research_tools", research_tools_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("analysis_tools", analysis_tools_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("writer", writer_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Planner → Researcher (initial flow)
    graph.add_edge("planner", "researcher")

    # Researcher → Research Tools
    graph.add_edge("researcher", "research_tools")

    # Research Tools → Conditional routing
    # If this is the first pass (no analysis yet), go to analyst
    # If this is a research-only loop from reviewer, go back to reviewer
    graph.add_conditional_edges(
        "research_tools",
        route_after_research_tools,
        {"analyst": "analyst", "reviewer": "reviewer"},
    )

    # Analyst → Analysis Tools
    graph.add_edge("analyst", "analysis_tools")

    # Analysis Tools → Reviewer
    graph.add_edge("analysis_tools", "reviewer")

    # Reviewer → Conditional routing
    graph.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"writer": "writer", "researcher": "researcher", "analyst": "analyst"},
    )

    # Writer → END
    graph.add_edge("writer", END)

    return graph


def compile_graph(save_visualization: bool = True):
    """
    Compile the graph and optionally save visualization.

    Args:
        save_visualization: Whether to save graph as PNG

    Returns:
        Compiled graph application
    """
    graph = build_graph()
    app = graph.compile()

    print(f"save: {save_visualization}")

    try:
        app.get_graph().draw_mermaid_png(output_file_path="graph.png")
        logger.info("Graph visualization saved to graph.png")
    except Exception as e:
        logger.warning(f"Could not save graph visualization:  {e}")

    return app


###############################################################################
# Analysis Runner
###############################################################################


class FinancialAnalysisAgent:
    """
    Main class for running financial analysis.

    Attributes:
        app:  Compiled LangGraph application
        recursion_limit: Maximum recursion depth for graph execution
    """

    def __init__(self, recursion_limit: int = 200, save_graph: bool = True):
        """
        Initialize the Financial Analysis Agent.

        Args:
            recursion_limit: Maximum recursion depth
            save_graph:  Whether to save graph visualization
        """
        self.recursion_limit = recursion_limit
        self.app = compile_graph(save_visualization=save_graph)
        logger.info("Financial Analysis Agent initialized")

    def run(self, user_request: str) -> str:
        """
        Run financial analysis for the given request.

        Args:
            user_request: Natural language analysis request

        Returns:
            Generated financial analysis report
        """
        self._print_header(user_request)

        try:
            # Initialize state and run graph
            initial_state = init_state(user_request)
            config = {"recursion_limit": self.recursion_limit}

            result = self.app.invoke(initial_state, config=config)

            report = result.get("final_report", "No report generated")

            self._print_summary(result)

            return report

        except Exception as e:
            import traceback

            error_msg = f"Error:  {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return error_msg

    def _print_header(self, user_request: str) -> None:
        """Print execution header."""
        print("")
        print("=" * 70)
        print("FINANCIAL ANALYSIS AGENT")
        print("=" * 70)
        print("Config:")
        print(f"  - Max calls per tool: {MAX_CALLS_PER_TOOL}")
        print(f"  - Max iterations: {MAX_ITERATIONS}")
        print(f"  - Recursion limit: {self.recursion_limit}")
        print("")
        print(f"Request: {user_request}")
        print("=" * 70)

    def _print_summary(self, result: dict) -> None:
        """Print execution summary."""
        print("")
        print("=" * 70)
        print("EXECUTION SUMMARY")
        print("=" * 70)
        print(f"  Total iterations: {result.get('iteration_count', 0)}")
        print(f"  Total tool calls: {result. get('total_tool_calls', 0)}")
        print("  Tool usage:")

        tool_counts = result.get("tool_call_counts", {})
        for name in ALL_TOOL_NAMES:
            count = tool_counts.get(name, 0)
            if count > 0:
                print(f"    - {name}: {count}")

        print("=" * 70)


def run_analysis(user_request: str) -> str:
    """
    Convenience function to run analysis without instantiating class.

    Args:
        user_request: Natural language analysis request

    Returns:
        Generated financial analysis report
    """
    agent = FinancialAnalysisAgent(save_graph=True)
    return agent.run(user_request)


###############################################################################
# Main Entry Point
###############################################################################


def main() -> None:
    """Main entry point for the Financial Analysis Agent."""

    # Validate configuration
    if not validate_config():
        logger.warning("Configuration validation failed.  Some features may not work.")

    # Default request
    request = "Analyze NVDA for 2023-2025:  profitability, balance sheet, cash flow, and recent news.  Include YoY changes."

    # Run analysis
    agent = FinancialAnalysisAgent(save_graph=True)
    report = agent.run(request)

    # Print report
    print("")
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(report)


if __name__ == "__main__":
    main()
