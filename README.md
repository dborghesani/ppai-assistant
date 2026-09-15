# ppai-assistant

Automotive AI Assistant powered by **CrewAI**, **Qt**, and **Kyutai** voice models.

This project demonstrates an in-vehicle intelligent assistant capable of handling user requests, vehicle telemetry, and proactive suggestions using a multi-agent system, with optional spoken (TTS/STT) and hands-free continuous-conversation interaction.

## 🚗 Features

- **Multimodal Interaction**: Processes both user inputs (text/voice) and internal vehicle events (e.g., battery level, driver fatigue).
- **CrewAI Integration**: Utilizes CrewAI for agent orchestration, task management, and LLM inference.
- **Skill-Based Architecture**: Modular skills including:
  - **Conversation**: Natural language interaction.
  - **Driver Health**: Monitoring driver wellbeing.
  - **Navigation**: Handling route and location data.
  - **Proactive Suggestions**: Anticipating driver needs.
  - **Vehicle**: Managing vehicle states and services.
- **Voice (optional, GPU required)**: Kyutai Delayed Streams Modeling TTS/STT, with three usage modes:
  - **Text-only**: type in the UI, read the spoken response as text (default, no GPU needed).
  - **Spoken responses (TTS)**: agent replies are synthesized and played out loud.
  - **Push-to-talk (STT)**: hold a mic button to record a question, released to transcribe and send it.
  - **Continuous conversation**: always-listening mode with automatic turn-taking (semantic VAD, or a silence-based fallback), barge-in (interrupt the assistant mid-sentence), and an echo-suppression heuristic (raw-energy gate + cross-correlation against the TTS's own recent output) to avoid the assistant re-transcribing itself.
- **Real-Time UI**: Built with PySide6 and QML for a responsive dashboard interface.

## 🛠 Tech Stack

- **Python**: Core logic.
- **CrewAI**: Agent framework.
- **PySide6**: UI framework (Qt for Python).
- **QML**: Declarative UI language (with automatic OS Dark/Light theme switching).
- **Ollama**: Local LLM inference (default: `qwen2.5:3b-instruct`).
- **Kyutai `moshi` (Delayed Streams Modeling)**: PyTorch TTS/STT models for voice input/output, run locally on GPU.
- **sounddevice**: Microphone capture and speaker playback.
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
├── voice/               # Kyutai TTS/STT (Delayed Streams Modeling, PyTorch backend)
│   ├── tts_manager.py   # Text-to-speech: warmup, playback, barge-in stop(), echo reference buffer
│   ├── stt_manager.py   # Push-to-talk transcription + continuous conversation loop (turn-taking, barge-in)
│   └── gpu_lock.py      # Shared lock serializing STT/TTS GPU access (avoids CUDA graph capture races)
├── ui/                  # QML interface and Qt Bridge
│   ├── main.qml         # Dashboard UI with dark/light mode, text input, push-to-talk and conversation toggle
│   └── bridge.py        # PySide6 VehicleBridge communicating with async agent/broker/voice managers
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
   This also installs the voice stack (`moshi`, `torch`, `sounddevice`, `sphn`), which is a sizeable download
   (torch alone is ~1GB). Voice features are opt-in at runtime (`tts_enabled`/`stt_enabled`, both `False` by
   default) so you don't need a GPU just to run the app in text-only mode.

3. **Run a Local LLM**:
   Ensure you have **Ollama** installed and running. Pull the default model:
   ```bash
   ollama pull qwen2.5:3b-instruct
   ```

4. **(Optional) Enable voice**: TTS/STT require an NVIDIA GPU (CUDA). The default models are:
   - TTS: `kyutai/tts-1.6b-en_fr` (or the smaller, English-only `kyutai/tts-0.75b-en-public`)
   - STT: `kyutai/stt-1b-en_fr`

   Weights are downloaded automatically from Hugging Face on first use. Both models are warmed up
   (a dummy inference) at startup so the first real request isn't slowed down by CUDA kernel
   compilation. See [Configuration](#configuration) for all `tts_*`/`stt_*`/`conversation_*` options.

5. **Run InfluxDB** (used by `data/database_manager.py` to persist vehicle/driver state):

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

   or on powershell:

   ```powershell
   docker run -d `
   --name influxdb `
   --network influx-net `
   --restart unless-stopped `
   -p 8086:8086 `
   -v influxdb-data:/var/lib/influxdb2 `
   -e DOCKER_INFLUXDB_INIT_MODE=setup `
   -e DOCKER_INFLUXDB_INIT_USERNAME=admin `
   -e DOCKER_INFLUXDB_INIT_PASSWORD=admin123456 `
   -e DOCKER_INFLUXDB_INIT_ORG=stellantis `
   -e DOCKER_INFLUXDB_INIT_BUCKET=assistant-bucket `
   -e DOCKER_INFLUXDB_INIT_ADMIN_TOKEN=my-super-secret-token `
   influxdb:2

The default values above match the ones in `config.py` (`influxdb_url`, `influxdb_token`, `influxdb_org`, `influxdb_bucket`).
If you change any of them, override the corresponding `influxdb_*` setting when running `main.py` (see [Configuration](#configuration)).

## 🚀 Usage

### Basic Run
Run the application with default settings (text-only, no voice, no MQTT):
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
| `max_tokens` | Max tokens generated per LLM response | `1024` |
| `context_window_size` | LLM context window size | `4096` |
| `data_websocket_url` | WebSocket URL for telemetry source | `ws://localhost:4545/socket` |
| `data_replay_folder` | Path to JSON dataset folder for telemetry replay | `""` |
| `mqtt_enabled` | Enable MQTT telemetry broker and client integration | `False` |
| `mqtt_embedded_broker` | Run built-in embedded MQTT broker (amqtt) when MQTT is enabled | `False` |
| `mqtt_host` | MQTT broker host | `127.0.0.1` |
| `mqtt_port` | MQTT broker port | `1883` |
| `mqtt_topic` | MQTT subscribe topic | `telemetry/#` |
| `knowledge_update_interval` | Interval (seconds) for batching telemetry updates | `1.0` |
| `knowledge_numeric_change_ratio` | Relative change ratio that counts as "significant" for numeric telemetry | `0.05` |
| `tts_enabled` | Enable spoken responses (Kyutai TTS, requires GPU) | `False` |
| `tts_hf_repo` | Hugging Face repo of the TTS checkpoint | `kyutai/tts-1.6b-en_fr` |
| `tts_voice` | Voice reference clip (from `kyutai/tts-voices`) | `expresso/ex03-ex01_happy_001_channel1_334s.wav` |
| `tts_device` | Device for TTS inference | `cuda` |
| `tts_n_q` | Number of audio codebooks used by the TTS model | `16` |
| `tts_cfg_coef` | Classifier-free guidance coefficient (ignored if the checkpoint has no CFG distillation support) | `3.0` |
| `stt_enabled` | Enable push-to-talk / continuous conversation (Kyutai STT, requires GPU) | `False` |
| `stt_hf_repo` | Hugging Face repo of the STT checkpoint | `kyutai/stt-1b-en_fr` |
| `stt_device` | Device for STT inference | `cuda` |
| `conversation_vad_head_index` | Semantic-VAD pause-duration head to use, if the checkpoint provides one (0=0.5s, 1=1.0s, 2=2.0s, 3=3.0s) | `2` |
| `conversation_vad_threshold` | Pause probability above which a turn is considered finished | `0.5` |
| `conversation_pause_seconds` | Silence-token fallback turn-taking duration (used when the checkpoint has no VAD heads) | `1.0` |
| `conversation_session_silence_timeout` | Seconds of total silence before a conversation session auto-ends | `10.0` |
| `conversation_mute_mic_during_tts` | Don't transcribe mic audio while the assistant is speaking (avoids self-transcription feedback loops); set `False` only with headphones | `True` |
| `conversation_barge_in_energy_threshold` | Raw audio RMS above which muted audio is considered for barge-in | `0.035` |
| `conversation_barge_in_min_frames` | Consecutive loud audio callbacks required before barge-in triggers | `4` |
| `conversation_echo_correlation_threshold` | Cross-correlation against the TTS's own recent output above which loud audio is classified as echo (ignored) rather than barge-in | `0.6` |
| `influxdb_url` | InfluxDB server URL | `http://localhost:8086` |
| `influxdb_token` | InfluxDB auth token | `my-super-secret-token` |
| `influxdb_org` | InfluxDB organization | `stellantis` |
| `influxdb_bucket` | InfluxDB bucket | `assistant-bucket` |
| `is_socket_enabled` | Enable the WebSocket telemetry ingress source | `True` |
| `socket_url` | WebSocket URL for telemetry source (used by `SourceManager`) | `ws://localhost:4545/socket` |

**Examples:**
```bash
# Run with custom model and replay dataset
python main.py ollama_llm="ollama/qwen2.5:7b-instruct" data_replay_folder="../Dataset/assistant/json_output"

# Run with embedded MQTT broker enabled
python main.py mqtt_enabled=True mqtt_embedded_broker=True

# Enable spoken responses only (TTS), keep typing questions in the UI
python main.py tts_enabled=True

# Enable push-to-talk: hold the mic button to record, release to transcribe and send
python main.py tts_enabled=True stt_enabled=True

# Full hands-free continuous conversation (always-listening, barge-in, auto turn-taking)
# toggled from the "🎙️ Conversation" button in the UI once stt_enabled/tts_enabled are on
python main.py tts_enabled=True stt_enabled=True

# Continuous conversation with headphones: disable the echo-avoidance mute so
# barge-in reacts through the real STT model instead of the raw-energy heuristic
python main.py tts_enabled=True stt_enabled=True conversation_mute_mic_during_tts=False

# Use a smaller/English-only TTS checkpoint (lower VRAM, no CFG distillation)
python main.py tts_enabled=True tts_hf_repo="kyutai/tts-0.75b-en-public" tts_n_q=16

# Replay a recorded telemetry dataset with MQTT and voice all enabled
python main.py \
  data_replay_folder="../Dataset/assistant/json_output_20260310_113315" \
  mqtt_enabled=True mqtt_embedded_broker=True \
  tts_enabled=True stt_enabled=True
```

## 🧠 How It Works

1. **Initialization**: The app starts a PySide6 event loop alongside an async loop using `qasync`. If enabled,
   `TTSManager`/`STTManager` load their Kyutai models and run a warmup inference off the event loop, so the
   first real interaction isn't slowed down by CUDA kernel compilation.
2. **Ingress & Source Management**:
   - Telemetry from WebSockets or MQTT is received, deserialized into strongly-typed dataclasses, and routed into `data_event_queue`.
   - UI adjustments (e.g. `DriverEmotionState`, `DriverDrivingStyle`) are routed via MQTT or direct database writes.
3. **Storage & Knowledge Extraction**:
   - `DatabaseManager` writes telemetry time-series into InfluxDB and notifies `KnowledgeManager`.
   - `KnowledgeManager` computes statistical trends, windowed aggregations, and detects significant changes.
4. **Agent Orchestration**:
   - `AutomotiveAgent` evaluates significant events and user inputs using CrewAI against domain skills (`skills/*.md`).
   - Applies deduplication and urgency-based cooldowns to prevent alert fatigue, while direct user requests always get a real response.
5. **Voice Input/Output** (optional):
   - Push-to-talk records the mic while the button is held and transcribes on release.
   - Continuous conversation mode keeps the mic open, detects end-of-turn (semantic VAD or a silence-token fallback), and supports barge-in: loud audio while the assistant is speaking is checked against a raw-energy gate and, if a recent-playback reference is available, cross-correlated against what was just played to tell apart a genuine interruption from the assistant's own voice leaking into the mic.
   - `voice/gpu_lock.py` serializes GPU access between STT and TTS, since Kyutai's CUDA-graph-based inference isn't safe to run concurrently from multiple threads.
6. **UI & Speech Output**:
   - Spoken responses and actions are published back to `VehicleBridge` and displayed live in `responseField` in [ui/main.qml](ui/main.qml), and spoken aloud when `tts_enabled=True`.
   - The UI automatically synchronizes its color palette with the operating system's Dark/Light mode theme.

## ⚠️ Known Limitations

- The Kyutai models are run through the **PyTorch reference implementation**, which the upstream project
  documents as intended "for research and tinkering", not low latency. Expect a few seconds of TTS latency
  per utterance; STT is fast after warmup. Kyutai's own low-latency deployments (e.g. Unmute) use an
  optimized Rust server instead.
- There's no real acoustic echo cancellation (AEC); the barge-in echo-suppression heuristic (energy gate +
  cross-correlation) is a practical approximation, not a substitute for proper AEC or a headset.

