from dataclasses import dataclass
from datetime import datetime


@dataclass
class ConfigAssistant:
    crewai_verbose: bool = False
    
    stellantis_llm: str = ""
    
    # custom LLM configuration (optional)
    ollama_host: str = "localhost"
    ollama_port: int = 11434
    ollama_llm: str = "ollama/qwen2.5:3b-instruct"
    ollama_timeout: int = 1200
