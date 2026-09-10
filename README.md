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
- **QML**: Declarative UI language.
- **Ollama**: Local LLM inference (default: `qwen2.5:3b-instruct`).
- **OmegaConf**: Configuration management.
- **structlog**: Structured logging.
- **InfluxDB 2**: Time-series storage for vehicle/driver state history.

## 📂 Project Structure

```text
.
├── main.py          # Application entry point
├── agents.py        # CrewAI Agent and Task definitions
├── config.py        # Configuration dataclasses
├── skill_manager.py # Manager for loading skill contexts
├── knowledge_manager.py
├── action_manager.py
├── events.py
├── skills/          # Markdown files defining agent skills
├── data/            # Dataclasses and InfluxDB persistence layer
│   ├── assistant_dataclasses.py
│   ├── database_manager.py
│   └── skill_map.py
├── ui/              # QML interface and bridge to Python
│   ├── main.qml
│   └── bridge.py
└── tools/           # Agent tools directory
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
| `influxdb_url` | InfluxDB server URL | `http://localhost:8086` |
| `influxdb_token` | InfluxDB auth token | `my-super-secret-token` |
| `influxdb_org` | InfluxDB organization | `stellantis` |
| `influxdb_bucket` | InfluxDB bucket | `assistant-bucket` |

**Example:**
```bash
python main.py ollama_host="192.168.1.50" ollama_llm="mistral"
```

## 🧠 How It Works

1. **Initialization**: The app starts a PySide6 event loop and an async loop using `qasync`.
2. **Agent Creation**: An `AutomotiveAgent` is initialized with a CrewAI setup (Agent, Task, Crew).
3. **Event Loop**: The agent continuously listens for:
   - **User Events**: Inputs from the QML UI via the `VehicleBridge`.
   - **System Events**: Simulated or connected vehicle telemetry (e.g., "low_battery", "driver_fatigue").
4. **Processing**: The CrewAI processes the event against the relevant `Skill` context and returns a response.
5. **UI Update**: The response is pushed back to the QML interface.
