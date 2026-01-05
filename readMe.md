# Self-Correcting Financial Analysis Agent 📈

A multi-agent AI system built with **LangGraph** and **Streamlit** for automated, reliable financial analysis. This agent moves beyond simple "chat" by orchestrating a team of specialized AI workers to plan research, gather live market data, verify metrics, and write comprehensive investment reports.

![Project Demo](project_demo.gif)

## 🚀 Key Features

*   **Plan-Execute-Verify-Correct Architecture:** Unlike standard LLM chains, this agent self-corrects. If data is missing or a tool fails, it replans and retries automatically.
*   **Multi-Agent Orchestration:** Specialized roles (Planner, Researcher, Analyst, Reviewer, Writer) handle different aspects of the pipeline.
*   **Live Market Data:** Integrates with `yfinance` for real-time stock data and `Tavily` for current market news.
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
*   **LLM:** OpenAI GPT-4o (Reasoning & Generation)
*   **Tools:** `yfinance` (Market Data), `Tavily` (Search/News)

## 📦 Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/financial-analysis-agent.git
    cd financial-analysis-agent
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables**
    Create a `.env` file in the root directory and add your API keys:
    ```env
    # Required for LLM
    OPENAI_API_KEY=your_openai_api_key
    
    # Required for News Search
    TAVILY_API_KEY=your_tavily_api_key

    # Optional: If using a custom endpoint/proxy
    # BASE_URL=your_base_url
    # AUTH_CODE=your_auth_code
    ```

## 🏃‍♂️ Usage

To start the application, run the Streamlit command:

```bash
streamlit run main.py