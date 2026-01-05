# Self-Correcting Financial Analysis Agent 📈

A multi-agent AI system built with **LangGraph** and **Streamlit** for automated, reliable financial analysis. This agent moves beyond simple "chat" by orchestrating a team of specialized AI workers to plan research, gather live market data, verify metrics, and write comprehensive investment reports.

![Project Demo](project_demo.gif)

## 🚀 Key Features

*   **Plan-Execute-Verify-Correct Architecture:** Unlike standard LLM chains, this agent self-corrects. If data is missing or a tool fails, it replans and retries automatically.
*   **Multi-Agent Orchestration:** Specialized roles (Planner, Researcher, Analyst, Reviewer, Writer) handle different aspects of the pipeline.
*   **Live Market Data:** Integrates with `yfinance` for real-time stock data and `DuckDuckGo` for current market news.
*   **Structured Outputs:** Uses Pydantic models to ensure strictly formatted data routing between nodes.
*   **Interactive UI:** A clean, real-time dashboard built with Streamlit to visualize the agent's thought process and final report.

## 🏗️ Architecture

This project implements a state machine using **LangGraph**. The workflow consists of the following nodes:

1.  **Planner:** Decomposes the user request into a step-by-step research plan.
2.  **Researcher:** Executes specific tools (Financials, Price History, News) to gather data.
3.  **Analyst:** Computes strict metrics (YoY Growth, Ratios) based on the gathered data.
4.  **Reviewer:** Validates the data against the plan. If data is missing, it routes back to the *Researcher*. If analysis is weak, it routes back to the *Analyst*.
5.  **Writer:** Synthesizes all verified data into a final markdown report.

## 🛠️ Tech Stack

*   **Framework:** [LangChain](https://www.langchain.com/) / [LangGraph](https://python.langchain.com/docs/langgraph)
*   **Frontend:** [Streamlit](https://streamlit.io/)
*   **LLM:** OpenAI GPT-OSS-120b (Reasoning & Generation)
*   **Tools:** `yfinance` (Market Data), `DuckDuckGo` (Search/News)
*   **Package Manager:** [uv](https://github.com/astral-sh/uv)

## 📦 Installation

This project uses `uv` for fast dependency management.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/financial-analysis-agent.git
    cd financial-analysis-agent
    ```

2.  **Install `uv` (if not already installed)**
    ```bash
    pip install uv
    # Or on macOS: brew install uv
    ```

3.  **Sync dependencies**
    This creates the virtual environment and installs all packages from `pyproject.toml`.
    ```bash
    uv sync
    ```

4.  **Set up environment variables**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    # Required: API Key or Auth Code depending on your setup
    # OPENAI_API_KEY=your_key_here
    # OR if using a custom endpoint/proxy:
    AUTH_CODE=your_auth_code
    BASE_URL=your_base_url
    
    # Required for Tracing with LangSmith (Optional but recommended)
    LANGSMITH_TRACING=true
    LANGSMITH_ENDPOINT=https://api.smith.langchain.com
    LANGSMITH_API_KEY=your_langsmith_key
    LANGSMITH_PROJECT=financial-agent
    ```

## 🏃‍♂️ Usage

Run the Streamlit application using `uv run` to automatically handle the environment:

```bash
uv run streamlit run streamlit_app.py --server.enableCORS=false