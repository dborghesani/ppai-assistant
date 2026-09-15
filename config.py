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

    # Text-to-speech (Kyutai TTS, Delayed Streams Modeling)
    tts_enabled: bool = False
    tts_hf_repo: str = "kyutai/tts-1.6b-en_fr"  # tts-0.75b-en-public English-only model
    tts_voice: str = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    tts_device: str = "cuda"
    tts_n_q: int = 16
    tts_cfg_coef: float = 3.0

    # Speech-to-text (Kyutai STT, push-to-talk, Delayed Streams Modeling)
    stt_enabled: bool = False
    stt_hf_repo: str = "kyutai/stt-1b-en_fr"
    stt_device: str = "cuda"

    # Continuous conversation mode (turn-taking via semantic VAD, barge-in)
    conversation_vad_head_index: int = 2  # pause-duration head: 0=0.5s, 1=1.0s, 2=2.0s, 3=3.0s
    conversation_vad_threshold: float = 0.5
    conversation_session_silence_timeout: float = 10.0
    conversation_pause_seconds: float = 1.0  # fallback turn-taking when checkpoint has no VAD heads
    # Without acoustic echo cancellation, the mic can pick up the assistant's own
    # speaker output and re-transcribe it as a new user request, causing a runaway
    # feedback loop. Muting transcription while TTS is speaking prevents that;
    # barge-in is still possible via a cheap raw-energy check below. Set to False
    # only when using headphones (no speaker leakage into the mic).
    conversation_mute_mic_during_tts: bool = True
    # Raised from initial defaults (0.02/2) because transient noise (bumps, coughs,
    # engine/road noise) was occasionally crossing the threshold and cutting off
    # playback with no real interruption. Tune per your noise environment.
    conversation_barge_in_energy_threshold: float = 0.035
    conversation_barge_in_min_frames: int = 4
    # Loud audio picked up while muted is cross-correlated against the audio
    # actually being played; above this threshold it's classified as TTS echo
    # (ignored) rather than a genuine user interruption.
    conversation_echo_correlation_threshold: float = 0.6

    #influxDB
    influxdb_url: str = "http://localhost:8086"
    influxdb_token: str = "my-super-secret-token"
    influxdb_org: str = "stellantis"
    influxdb_bucket: str = "assistant-bucket"

    # socket configuration
    socket_url: str = "ws://localhost:4545/socket"
    is_socket_enabled: bool = True
