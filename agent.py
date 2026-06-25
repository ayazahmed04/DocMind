# agent.py – Agentic reasoning with local LLM and free tools
# Uses langgraph's prebuilt ReAct agent (modern replacement for AgentExecutor)
from langchain_ollama import OllamaLLM
from langgraph.prebuilt import create_react_agent          # ✅ modern API
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
import numexpr as ne


def get_agent(llm, verbose=True):
    """
    Build a ReAct agent using langgraph (modern LangChain stack).
    Returns a compiled graph that behaves like AgentExecutor.
    """

    # Web search tool (DuckDuckGo – free, no key)
    search = DuckDuckGoSearchRun()

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information. Input should be a search query."""
        return search.run(query)

    # Calculator tool
    @tool
    def calculator(expression: str) -> str:
        """Perform mathematical calculations. Input should be a math expression like '2 + 2' or 'sqrt(16)'."""
        try:
            result = ne.evaluate(expression)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    # Wikipedia tool
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    @tool
    def wikipedia_search(query: str) -> str:
        """Look up factual information on Wikipedia. Input should be a search term."""
        return wikipedia.run(query)

    tools = [web_search, calculator, wikipedia_search]

    # create_react_agent from langgraph returns a compiled graph
    agent = create_react_agent(llm, tools)
    return agent


def run_agent(agent, input_text: str) -> str:
    """
    Helper to invoke the langgraph agent and extract the final text response.
    Usage: run_agent(agent, "What is the capital of France?")
    """
    result = agent.invoke({"messages": [("user", input_text)]})
    # Last message in the list is the final answer
    return result["messages"][-1].content