from langchain_ollama import OllamaLLM

def get_llm(model: str = "mistral:7b", temperature: float = 0.0):
    import os
    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return OllamaLLM(model=model, temperature=temperature, base_url=base_url)