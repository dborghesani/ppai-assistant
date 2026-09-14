# ppai-assistant

Automotive AI Assistant powered by **CrewAI** and **Qt**.

This project demonstrates an in-vehicle intelligent assistant capable of handling user requests, vehicle telemetry, and proactive suggestions using a multi-agent system.

## 🚗 Features

- **Multimodal Interaction**: Processes both user inputs (text/voice) and internal vehicle events (e.g., battery level, driver fatigue).
- **CrewAI Integration**: Utilizes CrewAI for agent orchestration, task management, and LLM inference.
- **Skill-Based Architecture**: Modular skills including:
  - **Conversation**: Natural language interaction.
  - **Driver Health**: Monitoring driver wellbeing.
  - **Navigation**: Handling route and location data.
  - **Proactive Suggestions**: Anticipating driver needs.
  - **Vehicle**: Managing vehicle states and services.
- **Real-Time UI**: Built with PySide6 and QML for a responsive dashboard interface.

## 🛠 Tech Stack

- **Python**: Core logic.
- **CrewAI**: Agent framework.
- **PySide6**: UI framework (Qt for Python).
- **QML**: Declarative UI language (with automatic OS Dark/Light theme switching).
- **Ollama**: Local LLM inference (default: `qwen2.5:3b-instruct`).
- **OmegaConf**: Configuration management.
- **structlog**: Structured logging.
- **InfluxDB 2**: Time-series storage for vehicle/driver state history.
- **MQTT (aiomqtt & amqtt)**: Real-time telemetry broker and messaging support (with embedded broker option).
- **WebSockets**: Ingress and replay streaming for sensor telemetry.

## 📂 Project Structure

```text
.
├── main.py              # Application entry point and async lifecycle
├── automotive_agent.py  # CrewAI Agent, Task and notification logic
├── config.py            # Configuration dataclasses (OmegaConf)
├── skill_manager.py     # Manager for loading skill contexts
├── knowledge_manager.py # Statistical knowledge extraction from telemetry
├── action_manager.py    # Vehicle action dispatching
├── events.py            # Event models (CarEvent)
├── skills/              # Markdown files defining agent skills
├── data/                # Data ingest, dataclasses and InfluxDB persistence layer
│   ├── assistant_dataclasses.py # Vehicle, driver, and environment models
│   ├── database_manager.py      # InfluxDB persistence and query layer
│   ├── skill_map.py             # Event-to-skill routing map
│   └── source_manager.py        # WebSocket/MQTT source and replay manager
├── ui/                  # QML interface and Qt Bridge
│   ├── main.qml         # Dashboard UI with dark/light mode and agent response display
│   └── bridge.py        # PySide6 VehicleBridge communicating with async agent/broker
└── tools/               # Agent tools directory
```

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone ...
   cd ppai-assistant
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Run a Local LLM**:
   Ensure you have **Ollama** installed and running. Pull the default model:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

4. **Run InfluxDB** (used by `data/database_manager.py` to persist vehicle/driver state):

   Create a dedicated Docker network with a subnet that doesn't overlap with your VPN's range
   (the default `172.17.0.0/16` used by Docker's bridge network conflicts with some VPN configs):
   ```bash
   docker network create --subnet=172.30.0.0/16 influx-net
   ```

   Then start the InfluxDB 2 container on that network:
   ```bash
   docker run -d \
     --name influxdb \
     --network influx-net \
     --restart unless-stopped \
     -p 8086:8086 \
     -v influxdb-data:/var/lib/influxdb2 \
     -e DOCKER_INFLUXDB_INIT_MODE=setup \
     -e DOCKER_INFLUXDB_INIT_USERNAME=admin \
     -e DOCKER_INFLUXDB_INIT_PASSWORD=admin123456 \
     -e DOCKER_INFLUXDB_INIT_ORG=stellantis \
     -e DOCKER_INFLUXDB_INIT_BUCKET=assistant-bucket \
     -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-token \
     influxdb:2
   ```

   The default values above match the ones in `config.py` (`influxdb_url`, `influxdb_token`, `influxdb_org`, `influxdb_bucket`).
   If you change any of them, override the corresponding `influxdb_*` setting when running `main.py` (see [Configuration](#configuration)).

## 🚀 Usage

### Basic Run
Run the application with default settings:
```bash
python main.py
```

### Configuration
The application supports command-line configuration via **OmegaConf**. You can override settings defined in `config.py`.

| Argument | Description | Default |
| :--- | :--- | :--- |
| `crewai_verbose` | Enable verbose CrewAI logging | `False` |
| `ollama_host` | Ollama server host | `localhost` |
| `ollama_port` | Ollama server port | `11434` |
| `ollama_llm` | LLM model to use | `ollama/qwen2.5:3b-instruct` |
| `ollama_timeout` | LLM timeout in seconds | `1200` |
| `data_websocket_url` | WebSocket URL for telemetry source | `ws://localhost:4545/socket` |
| `data_replay_folder` | Path to JSON dataset folder for telemetry replay | `""` |
| `mqtt_enabled` | Enable MQTT telemetry broker and client integration | `False` |
| `mqtt_embedded_broker` | Run built-in embedded MQTT broker (amqtt) when MQTT is enabled | `True` |
| `mqtt_host` | MQTT broker host | `127.0.0.1` |
| `mqtt_port` | MQTT broker port | `1883` |
| `mqtt_topic` | MQTT subscribe topic | `telemetry/#` |
| `knowledge_update_interval` | Interval (seconds) for batching telemetry updates | `1.0` |
| `influxdb_url` | InfluxDB server URL | `http://localhost:8086` |
| `influxdb_token` | InfluxDB auth token | `my-super-secret-token` |
| `influxdb_org` | InfluxDB organization | `stellantis` |
| `influxdb_bucket` | InfluxDB bucket | `assistant-bucket` |

**Examples:**
```bash
# Run with custom model and replay dataset
python main.py ollama_llm="ollama/qwen2.5:7b-instruct" data_replay_folder="../Dataset/assistant/json_output"

# Run with embedded MQTT broker enabled
python main.py mqtt_enabled=True
```

## 🧠 How It Works

1. **Initialization**: The app starts a PySide6 event loop alongside an async loop using `qasync`.
2. **Ingress & Source Management**:
   - Telemetry from WebSockets or MQTT is received, deserialized into strongly-typed dataclasses, and routed into `data_event_queue`.
   - UI adjustments (e.g. `DriverEmotionState`, `DriverDrivingStyle`) are routed via MQTT or direct database writes.
3. **Storage & Knowledge Extraction**:
   - `DatabaseManager` writes telemetry time-series into InfluxDB and notifies `KnowledgeManager`.
   - `KnowledgeManager` computes statistical trends, windowed aggregations, and detects significant changes.
4. **Agent Orchestration**:
   - `AutomotiveAgent` evaluates significant events and user inputs using CrewAI against domain skills (`skills/*.md`).
   - Applies deduplication and urgency-based cooldowns to prevent alert fatigue.
5. **UI & Speech Output**:
   - Spoken responses and actions are published back to `VehicleBridge` and displayed live in `responseField` in [ui/main.qml](ui/main.qml).
   - The UI automatically synchronizes its color palette with the operating system's Dark/Light mode theme.
