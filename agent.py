# agent.py – Agent using langchain_classic (works with Ollama, no conflicts)
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import numexpr as ne

def get_agent(llm, verbose=False):
    """Build a classic zero‑shot ReAct agent with web search, calculator, and Wikipedia."""

    # Web search
    search = DuckDuckGoSearchRun()
    web_search = Tool(
        name="Web Search",
        func=search.run,
        description="useful for finding current information from the internet"
    )

    # Calculator
    def calculator(expression: str) -> str:
        try:
            return str(ne.evaluate(expression))
        except Exception as e:
            return f"Error: {e}"

    calculator_tool = Tool(
        name="Calculator",
        func=calculator,
        description="useful for performing mathematical calculations. Input should be a mathematical expression."
    )

    # Wikipedia
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wikipedia_search = Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="useful for looking up facts and summaries from Wikipedia. Input should be a search term."
    )

    tools = [web_search, calculator, wikipedia_search]

    # Create agent with the classic initializer (always works)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        handle_parsing_errors=True
    )
    return agent