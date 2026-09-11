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

    data_websocket_url: str = "ws://localhost:4545/socket"
    data_replay_folder: str = ""

    knowledge_update_interval: float = 1.0
    knowledge_numeric_change_ratio: float = 0.05

    #influxDB
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "my-super-secret-token"
    influxdb_org: str = "stellantis"
    influxdb_bucket: str = "assistant-bucket"
