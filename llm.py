# llm.py
from langchain_ollama import OllamaLLM

def get_llm(model: str = "mistral:7b", temperature: float = 0.0):
    """Return an Ollama LLM instance."""
    return OllamaLLM(model=model, temperature=temperature)