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

    max_tokens: int = 1024
    context_window_size: int = 4096

    data_websocket_url: str = "ws://localhost:4545/socket"
    data_replay_folder: str = ""

    # MQTT broker configuration
    mqtt_enabled: bool = False
    mqtt_embedded_broker: bool = False
    mqtt_host: str = "127.0.0.1"
    mqtt_port: int = 1883
    mqtt_topic: str = "telemetry/#"
    mqtt_username: str | None = None
    mqtt_password: str | None = None

    knowledge_update_interval: float = 1.0
    knowledge_numeric_change_ratio: float = 0.05

    #influxDB
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "my-super-secret-token"
    influxdb_org: str = "stellantis"
    influxdb_bucket: str = "assistant-bucket"
