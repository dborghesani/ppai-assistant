const CONTROL_GROUPS = {
  driver: [
    { title: "Driver name", type: "text", cls: "DriverPreferences", key: "driver_name", value: "David", maxLength: 80 },
    { title: "Activity", type: "activities", cls: "DriverPhysicalState", key: "activity", value: "", options: ["Talking", "Driving", "Eating", "On the phone", "Sleeping", "About to exit", "Children out of place"] },
    { title: "Emotion", type: "ranges", cls: "DriverEmotionState", controls: [
      ["Angry", "angry", 0, 1, 0, .01, "decimal"], ["Disgust", "disgust", 0, 1, 0, .01, "decimal"],
      ["Fear", "fear", 0, 1, 0, .01, "decimal"], ["Happy", "happy", 0, 1, 0, .01, "decimal"],
      ["Sad", "sad", 0, 1, 0, .01, "decimal"], ["Surprise", "surprise", 0, 1, 0, .01, "decimal"],
      ["Neutral", "neutral", 0, 1, 1, .01, "decimal"]
    ]},
    { title: "Readiness", type: "ranges", controls: [
      ["Fatigue", "fatigue_level", 0, 1, 0, .01, "decimal", "DriverPhysicalState"],
      ["Attention", "attention_level", 0, 1, 1, .01, "decimal", "DriverPhysicalState"],
      ["Driving tension", "driving_tension", 0, 1, 0, .01, "decimal", "DriverDrivingStyle"]
    ]},
    { title: "Comfort preference", type: "ranges", cls: "DriverPreferences", controls: [
      ["Preferred cabin temperature", "preferred_cabin_temperature", 16, 30, 22, .5, "temp"]
    ]},
    { title: "Music preference", type: "textarea", cls: "DriverPreferences", key: "preferred_music", value: "", placeholder: "e.g. classic 1970s rock, guitar-driven and upbeat" }
  ],
  road: [
    { title: "Weather", type: "options", cls: "EnvironmentState", key: "weather", value: "Sunny", options: ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"] },
    { title: "Forecast", type: "options", cls: "EnvironmentState", key: "forecast_weather", value: "Sunny", options: ["Sunny", "Cloudy", "Rainy", "Snowy", "Foggy"] },
    { title: "Traffic", type: "options", cls: "EnvironmentState", key: "traffic", value: "No traffic", options: ["No traffic", "Light", "Medium", "Heavy"] },
    { title: "Road type", type: "options", cls: "EnvironmentState", key: "road_type", value: "Urban", options: ["Rural", "Urban", "Highway", "Residential"] },
    { title: "Risk", type: "options", cls: "EnvironmentState", key: "risk_level", value: "None", options: ["None", "Low", "Medium", "High"] },
    { title: "Time of day", type: "options", cls: "EnvironmentState", key: "time_of_day", value: "Morning", options: ["Morning", "Afternoon", "Evening", "Night"] },
    { title: "Road condition", type: "options", cls: "EnvironmentState", key: "road_condition", value: "Dry", options: ["Dry", "Wet", "Icy", "Gravel", "Snowy", "Muddy"] },
    { title: "Conditions", type: "ranges", cls: "EnvironmentState", controls: [
      ["Visibility", "visibility", 0, 1, 1, .01, "percent"],
      ["Outside temp.", "external_temperature", 10, 40, 22, .5, "temp"]
    ]}
  ],
  vehicle: [
    { title: "Vehicle state", type: "toggles", cls: "VehicleState", controls: [
      ["Engine", "engine_on", false, "On", "Off"], ["Doors", "doors_locked", false, "Locked", "Unlocked"],
      ["Front left door", "door_open_front_left", false, "Open", "Closed"], ["Front right door", "door_open_front_right", false, "Open", "Closed"],
      ["Rear left door", "door_open_rear_left", false, "Open", "Closed"], ["Rear right door", "door_open_rear_right", false, "Open", "Closed"],
      ["Trunk", "trunk_open", false, "Open", "Closed"], ["Navigation", "navigation_active", false, "Active", "Inactive"],
      ["Privacy mode", "privacy_mode", false, "On", "Off"]
    ]},
    { title: "Windows & roof", type: "toggles", cls: "VehicleState", controls: [
      ["Front left window", "window_open_front_left", false, "Open", "Closed"], ["Front right window", "window_open_front_right", false, "Open", "Closed"],
      ["Rear left window", "window_open_rear_left", false, "Open", "Closed"], ["Rear right window", "window_open_rear_right", false, "Open", "Closed"],
      ["Sunroof", "sunroof_open", false, "Open", "Closed"]
    ]},
    { title: "Climate", type: "ranges", cls: "VehicleState", integer: true, controls: [
      ["Cabin temp.", "internal_temperature", 10, 40, 22, .5, "temp"],
      ["Fan speed", "fan_speed", 0, 7, 3, 1, "integer"]
    ]},
    { title: "Climate controls", type: "toggles", cls: "VehicleState", controls: [
      ["Air conditioning", "air_conditioning_on", false, "On", "Off"],
      ["Cabin heating", "heating_on", false, "On", "Off"],
      ["Air recirculation", "air_recirculation_on", false, "On", "Off"],
      ["Seat heating", "seat_heating_on", false, "On", "Off"]
    ]},
    { title: "Audio", type: "toggles", cls: "VehicleState", controls: [
      ["Radio", "radio_on", false, "On", "Off"]
    ]},
    { title: "Audio volume", type: "ranges", cls: "VehicleState", integer: true, controls: [
      ["Volume", "audio_volume", 0, 100, 20, 5, "integer"]
    ]},
    { title: "Lighting", type: "toggles", cls: "VehicleState", controls: [
      ["Sidelights", "lights_on_sidelights", false, "On", "Off"], ["Low beams", "lights_on_low_beams", false, "On", "Off"],
      ["High beams", "lights_on_high_beams", false, "On", "Off"], ["Fog lights", "lights_on_fog_lights", false, "On", "Off"]
    ]},
    { title: "Turn signal", type: "options", cls: "VehicleState", key: "turn_signal", value: "Off", options: ["Off", "Right", "Left", "Both"] },
    { title: "Motion", type: "ranges", cls: "VehicleMotion", controls: [
      ["Speed", "speed", 0, 200, 0, 1, "speed"]
    ]},
    { title: "ADAS assistance", type: "toggles", cls: "VehicleState", controls: [
      ["Adaptive cruise control", "adaptive_cruise_control_on", false, "On", "Off"],
      ["Lane keeping assist", "lane_keep_assist_enabled", false, "On", "Off"],
      ["Blind spot monitor", "blind_spot_monitor", false, "On", "Off"]
    ]},
    { title: "ADAS settings", type: "ranges", cls: "VehicleState", integer: true, controls: [
      ["Target speed", "adas_target_speed", 0, 160, 90, 10, "speed"],
      ["Following distance", "following_distance_level", 1, 5, 3, 1, "integer"]
    ]},
    { title: "Detected objects", type: "ranges", cls: "DetectedObjects", integer: true, controls: [
      ["People around", "people_around", 0, 20, 0, 1, "integer"],
      ["Vehicles around", "vehicles_around", 0, 20, 0, 1, "integer"]
    ]},
    { title: "Object detection", type: "ranges", cls: "DetectedObjects", integer: true, controls: [
      ["Dangerous objects", "dangerous_objects_around", 0, 10, 0, 1, "integer"]
    ]},
    { title: "Interior object detection", type: "ranges", cls: "DetectedObjects", integer: true, controls: [
      ["People inside", "people_inside", 0, 5, 1, 1, "integer"],
      ["Children inside", "children_inside", 0, 4, 0, 1, "integer"]
    ]},
    { title: "Interior animal detection", type: "toggles", cls: "DetectedObjects", controls: [
      ["Dogs or cats inside", "animal_inside", false, "Detected", "Not detected"]
    ]}
  ],
  scenario: [
    { title: "Demo scenarios", type: "scenarios", hint: "Choose a scenario to add context or start a message simulation.", controls: [
      { label: "Meeting", id: "meeting", description: "A work meeting about an autonomous parking system, covering product scope, architecture, safety and driver experience.", groups: [
        { title: "Meeting phase", type: "meeting", cls: "DriverAgenda", keys: ["meeting_status", "meeting_timing"], controls: [
          { label: "Prepare", status: "prepare" },
          { label: "Late", timing: "late" },
          { label: "Started", status: "started" }
        ]}
      ]},
      { label: "Messages from Luca", id: "friend_messages", description: "Luca sends occasional messages about mutual friends. Privacy mode queues them until you allow the assistant to read them.", simulation: "friend-messages" },
      { label: "Driving simulation video", id: "driving_video", description: "Play a YouTube driving video in the assistant panel; you can keep chatting while it runs.", simulation: "driving-video" }
    ]}
  ]
};

class BridgeClient {
  constructor() {
    this.socket = null;
    this.requestId = 0;
    this.pending = new Map();
    this.reconnectDelay = 700;
    this.handlers = new Map();
  }

  connect() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    this.socket = new WebSocket(`${protocol}://${location.host}/ws`);
    setConnection("connecting");
    this.socket.addEventListener("open", () => {
      this.reconnectDelay = 700;
      setConnection("connected");
    });
    this.socket.addEventListener("message", event => this.receive(JSON.parse(event.data)));
    this.socket.addEventListener("close", () => {
      setConnection("offline");
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 1.6, 5000);
    });
  }

  receive(message) {
    if (message.event) {
      (this.handlers.get(message.event) || []).forEach(handler => handler(message.data));
      return;
    }
    const pending = this.pending.get(message.id);
    if (!pending) return;
    this.pending.delete(message.id);
    message.error ? pending.reject(new Error(message.error)) : pending.resolve(message.result);
  }

  on(event, handler) {
    this.handlers.set(event, [...(this.handlers.get(event) || []), handler]);
  }

  call(method, ...params) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
      showToast("Assistant connection is unavailable");
      return Promise.reject(new Error("WebSocket unavailable"));
    }
    const id = ++this.requestId;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
}

const bridge = new BridgeClient();
const controlContent = document.querySelector("#control-content");
const conversation = document.querySelector("#conversation");
const welcomeState = document.querySelector("#welcome-state");
const input = document.querySelector("#message-input");
const conversationButton = document.querySelector("#conversation-toggle");
const clearConversationButton = document.querySelector("#clear-conversation");
const duplicateSuppressionToggle = document.querySelector("#duplicate-suppression-toggle");
let activeTab = "driver";
let activeScenario = null;
let drivingSimulationActive = false;
let drivingVideoReady = false;
const DEFAULT_DRIVING_VIDEO_URL = "/recordings/rome.mp4";
let drivingVideoUrl = DEFAULT_DRIVING_VIDEO_URL;
let activityDetectionBusy = false;

function setActivityDetectionBusy(busy) {
  activityDetectionBusy = busy;
  const button = document.querySelector("#detect-activity");
  if (button) {
    button.disabled = busy;
    button.setAttribute("aria-pressed", String(busy));
    button.textContent = busy ? "Detecting..." : "Detect...";
  }
}

async function detectCabinActivity() {
  if (activityDetectionBusy) return;
  setActivityDetectionBusy(true);
  let stream;
  let video;
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: {width: 640, height: 480}, audio: false});
    video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    video.className = "cabin-camera-preview";
    video.setAttribute("aria-label", "Cabin camera");
    document.querySelector('[data-class="DriverPhysicalState"][data-key="activity"]')?.append(video);
    await video.play();
    if (!video.videoWidth || !video.videoHeight) throw new Error("Camera is not ready");
    const canvas = document.createElement("canvas");
    canvas.width = Math.min(640, video.videoWidth);
    canvas.height = Math.round(canvas.width * video.videoHeight / video.videoWidth);
    const context = canvas.getContext("2d");
    const frames = [];
    const started = performance.now();
    for (let index = 0; index < 6; index++) {
      if (index) await new Promise(resolve => setTimeout(resolve, 500));
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      frames.push({image: canvas.toDataURL("image/jpeg", 0.65), seconds: (performance.now() - started) / 1000});
    }
    stream.getTracks().forEach(track => track.stop());
    await bridge.call("detectActivity", frames);
  } catch (error) {
    showToast(error.message || "Camera unavailable");
    setActivityDetectionBusy(false);
  } finally {
    stream?.getTracks().forEach(track => track.stop());
    if (video) video.srcObject = null;
    video?.remove();
  }
}

bridge.on("activityDetectionChanged", data => {
  const state = data[0];
  setActivityDetectionBusy(state.busy);
  if (state.error) showToast(state.error);
  else if (state.activity === "Unknown") showToast("Activity could not be determined");
});
bridge.on("driverPhysicalStateChanged", data => {
  rememberControlState("DriverPhysicalState", data[0]);
  updateControlsInPlace("DriverPhysicalState", data[0]);
});

function loadDrivingVideo(rawUrl, autoplay = true) {
  const holder = document.querySelector("#driving-video-frame-holder");
  if (!holder) return false;
  drivingVideoUrl = rawUrl;
  drivingVideoReady = false;
  holder.replaceChildren();
  const video = document.createElement("video");
  video.title = "Driving simulation video";
  video.src = drivingVideoUrl;
  video.loop = true;
  video.autoplay = autoplay;
  video.muted = true;
  video.playsInline = true;
  video.addEventListener("canplay", () => { drivingVideoReady = true; });
  video.addEventListener("play", () => setDrivingVideoToggleIcon(false));
  video.addEventListener("pause", () => setDrivingVideoToggleIcon(true));
  video.addEventListener("loadedmetadata", updateDrivingVideoSlider);
  video.addEventListener("timeupdate", updateDrivingVideoSlider);
  video.addEventListener("error", () => showToast("Could not load driving recording"));
  holder.append(video);
  return true;
}

function drivingVideoElement() {
  return document.querySelector("#driving-video-frame-holder video");
}

function setDrivingVideoToggleIcon(paused) {
  const button = document.querySelector("#driving-video-toggle");
  if (!button) return;
  button.textContent = paused ? "\u25B6" : "\u275A\u275A";
  button.title = paused ? "Play video" : "Pause video";
}

function sendDrivingVideoCommand(command, value) {
  const video = drivingVideoElement();
  if (command === "togglePlay") {
    if (!video) return;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  } else if (command === "seekTo") {
    if (!video) return;
    video.currentTime = typeof value === "number" ? value : 0;
    if (video.paused) video.play().catch(() => {});
  }
}

let drivingVideoScrubbing = false;
let drivingVideoWasPlaying = false;

function startDrivingVideoScrub() {
  const video = drivingVideoElement();
  if (!video || drivingVideoScrubbing) return;
  drivingVideoScrubbing = true;
  drivingVideoWasPlaying = !video.paused && !video.ended;
  video.pause();
}

function scrubDrivingVideo(seconds) {
  const video = drivingVideoElement();
  if (!video || !Number.isFinite(seconds)) return;
  video.currentTime = seconds;
  updateDrivingVideoSlider(true);
}

function endDrivingVideoScrub() {
  if (!drivingVideoScrubbing) return;
  drivingVideoScrubbing = false;
  if (drivingVideoWasPlaying) drivingVideoElement()?.play().catch(() => {});
}

function formatDrivingVideoTime(seconds) {
  if (!Number.isFinite(seconds)) return "0:00";
  const total = Math.max(0, Math.floor(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function updateDrivingVideoSlider(scrubbing = false) {
  const video = drivingVideoElement();
  const slider = document.querySelector("#driving-video-seek");
  const label = document.querySelector("#driving-video-time");
  if (!slider || !label) return;
  if (!video || !Number.isFinite(video.duration) || video.duration <= 0) {
    slider.value = 0; slider.disabled = true;
    label.textContent = "0:00 / 0:00";
    return;
  }
  slider.disabled = false;
  slider.max = Math.floor(video.duration);
  if (!scrubbing && !drivingVideoScrubbing && !slider.matches(":active")) {
    slider.value = Math.floor(video.currentTime);
  }
  label.textContent = `${formatDrivingVideoTime(video.currentTime)} / ${formatDrivingVideoTime(video.duration)}`;
}

bridge.on("drivingVideoCaptureRequested", () => {
  const video = drivingVideoElement();
  const seconds = drivingSimulationActive && video && drivingVideoReady
    ? video.currentTime : null;
  bridge.call("reportDrivingVideoTime", null, seconds).catch(() => {});
});

function setDrivingSimulation(active) {
  drivingSimulationActive = active;
  if (!active) {
    drivingVideoElement()?.pause();
    drivingVideoReady = false;
    document.querySelector("#driving-video-frame-holder")?.replaceChildren();
  }
  document.querySelector('#driving-simulation-toggle[data-simulation="driving-video"]')
    ?.setAttribute("aria-pressed", String(active));
  const buttons = [...controlContent.querySelectorAll('.scenario-button[data-simulation="driving-video"]')];
  buttons.forEach(button => button.textContent = active ? "Stop simulation" : "Start simulation");
  const input = document.querySelector("#driving-video-url");
  if (input) {
    input.value = "";
    input.placeholder = drivingVideoUrl;
  }
  if (activeTab === "scenario") renderControls("scenario");
}

const DRIVING_VIDEO_BLOCK_HTML = `
  <h3>Driving simulation video</h3>
  <div class="driving-video-heading">
    <input id="driving-video-url" type="text" placeholder="e.g. paris (file in recordings folder)" aria-label="Recording name">
    <button id="driving-video-load" class="icon-button" type="button" title="Load video" aria-label="Load video">&#8601;</button>
  </div>
  <div id="driving-video-frame-holder" class="driving-video-frame-holder"></div>
  <div class="driving-video-seek-row">
    <input id="driving-video-seek" class="driving-video-seek" type="range" min="0" max="0" step="1" value="0" disabled aria-label="Video position">
    <span id="driving-video-time" class="driving-video-time">0:00 / 0:00</span>
  </div>
  <div class="driving-video-transport">
    <button id="driving-video-toggle" class="icon-button" type="button" title="Play/Pause video" aria-label="Play or pause video">\u25B6</button>
    <button id="driving-video-restart" class="icon-button" type="button" title="Restart video" aria-label="Restart video">&#8634;</button>
  </div>`;
let conversationActive = false;
let connectionState = "connecting";
let socketWasOpen = false;
let connectionTimer = null;
let assistantStatus = "idle";
let controlState = {};
let speakingTone = null;
let actPulseTimer = null;
let showActPulse = false;
let incomingMessageClassification = null;
let permissionMessageClassification = null;
let incomingMessageTimer = null;
let pendingAssistantMessage = null;
let assistantTurnPaused = false;
let friendSimulationActive = false;
const PERMISSION_BIT = 1 << 2; // AssistantStatus.ASK_PERMISSION_TO_TALK index
let permissionReminderSeconds = 10; // overridden from config via ready payload
let assistantFlags = 1; // idle
let permissionReminderTimer = null;
let permissionPulseTimer = null; // active only for the duration of a pulse
let toastTimer;
const musicAudio = document.querySelector("#music-audio");
let musicQueue = [];
let musicTrackIndex = 0;
let musicDuckingFactor = 0.25;
let musicPlaylistTitle = "Jamendo";

function updateMusicVolume() {
  const configured = Number(controlState["VehicleState.audio_volume"] ?? 20);
  const base = Math.max(0, Math.min(100, Number.isFinite(configured) ? configured : 20)) / 100;
  const ducked = assistantStatus === "talking" || speakingTone !== null;
  musicAudio.volume = base * (ducked ? musicDuckingFactor : 1);
}

function musicLink(selector, url) {
  const link = document.querySelector(selector);
  link.hidden = !url;
  if (url) link.href = url;
  else link.removeAttribute("href");
}

function displayMusicTrack(track) {
  document.querySelector("#music-track-title").textContent = track.title;
  document.querySelector("#music-artist").textContent = track.artist;
  const cover = document.querySelector("#music-cover");
  cover.hidden = !track.image;
  if (track.image) cover.src = track.image;
  else cover.removeAttribute("src");
  musicLink("#music-source", track.url);
  musicLink("#music-license", track.license);
  updateMusicVolume();
}

function syncMusicTransport() {
  const playing = !musicAudio.paused && Boolean(musicAudio.src);
  const button = document.querySelector("#music-play");
  button.classList.toggle("is-playing", playing);
  button.title = playing ? "Pause music" : "Play music";
  button.setAttribute("aria-label", button.title);
  document.querySelector("#music-next").disabled = musicTrackIndex + 1 >= musicQueue.length;
}

async function playMusicTrack(index) {
  if (!musicQueue[index]) return;
  musicTrackIndex = index;
  const track = musicQueue[index];
  musicAudio.src = track.audio;
  musicAudio.load();
  displayMusicTrack(track);
  try {
    await musicAudio.play();
    const muted = musicAudio.volume === 0;
    document.querySelector("#music-status").textContent = muted ? "Playing silently: volume is 0" : "Playing";
    if (muted) showToast("Increase Vehicle Audio volume to hear music");
  } catch (error) {
    const blocked = error?.name === "NotAllowedError";
    const media = musicAudio.error;
    const mediaError = media?.code;
    const volumeIsZero = musicAudio.volume === 0;
    const audioUrl = new URL(track.audio);
    const status = blocked ? "Tap Play to start"
      : volumeIsZero ? "Vehicle Audio volume is 0"
      : mediaError === 4 ? "Music stream unavailable or invalid"
      : "Audio source unavailable";
    document.querySelector("#music-status").textContent = status;
    showToast(blocked ? "Tap Play to start music" : status);
    console.warn("Music playback failed", {
      error: error?.name,
      errorMessage: error?.message,
      mediaError,
      mediaErrorMessage: media?.message,
      networkState: musicAudio.networkState,
      readyState: musicAudio.readyState,
      audioHost: audioUrl.host,
      audioPath: audioUrl.pathname,
      audioFormat: audioUrl.searchParams.get("format") || "unspecified",
      canPlayOgg: musicAudio.canPlayType('audio/ogg; codecs="vorbis"'),
      canPlayMp3: musicAudio.canPlayType("audio/mpeg"),
    });
  }
  syncMusicTransport();
}

async function controlMusic(command) {
  if (command === "stop") {
    musicAudio.pause();
    musicAudio.removeAttribute("src");
    musicAudio.load();
    musicQueue = [];
    document.querySelector("#music-transport").hidden = true;
    document.querySelector("#music-status").textContent = "Stopped";
    musicLink("#music-source", "");
    musicLink("#music-license", "");
  } else if (command === "pause") {
    musicAudio.pause();
    document.querySelector("#music-status").textContent = "Paused";
  } else if (command === "resume" && musicQueue.length) {
    await playMusicTrack(musicTrackIndex);
  } else if (command === "next") {
    await playMusicTrack(musicTrackIndex + 1);
  }
  syncMusicTransport();
}

function renderMusic(state) {
  if (!state) return;
  if (state.status === "control") {
    controlMusic(state.command);
    return;
  }
  document.querySelector("#music-panel").hidden = state.status === "disabled";
  const statuses = {not_configured: "Not configured", loading: "Finding a playlist", empty: "No matching music", error: "Music unavailable", idle: "", ready: "Ready", proposal: "Selection found"};
  document.querySelector("#music-status").textContent = statuses[state.status] ?? "";
  if (state.title) document.querySelector("#music-title").textContent = state.title;
  else document.querySelector("#music-title").textContent = musicPlaylistTitle;
  if ((state.status === "ready" || state.status === "proposal") && state.tracks?.length) {
    musicQueue = state.tracks;
    musicPlaylistTitle = state.title;
    musicTrackIndex = 0;
    displayMusicTrack(musicQueue[0]);
    document.querySelector("#music-transport").hidden = false;
    const controls = document.querySelectorAll("#music-transport button");
    controls.forEach(button => { button.disabled = state.status === "proposal"; });
    if (state.status === "ready" && state.autoplay) playMusicTrack(0);
    else if (state.status === "ready") {
      syncMusicTransport();
    }
  }
}

musicAudio.addEventListener("ended", () => {
  if (musicTrackIndex + 1 < musicQueue.length) playMusicTrack(musicTrackIndex + 1);
  else { document.querySelector("#music-status").textContent = "Finished"; syncMusicTransport(); }
});
musicAudio.addEventListener("error", () => {
  if (musicAudio.getAttribute("src")) {
    const status = musicAudio.error?.code === 4
      ? "Music stream unavailable or invalid"
      : "Audio source unavailable";
    document.querySelector("#music-status").textContent = status;
    showToast("This track could not be played");
  }
  syncMusicTransport();
});
musicAudio.addEventListener("play", syncMusicTransport);
musicAudio.addEventListener("pause", syncMusicTransport);
document.querySelector("#music-play").addEventListener("click", () => controlMusic(musicAudio.paused ? "resume" : "pause"));
document.querySelector("#music-next").addEventListener("click", () => controlMusic("next"));
document.querySelector("#music-stop").addEventListener("click", () => {
  controlMusic("stop");
  bridge.call("musicControl", "stop").catch(() => {});
});

function controlStateKey(className, key) {
  return `${className}.${key}`;
}

function rememberControlState(className, values) {
  let changed = false;
  Object.entries(values || {}).forEach(([key, value]) => {
    const stateKey = controlStateKey(className, key);
    if (!Object.is(controlState[stateKey], value)) changed = true;
    controlState[stateKey] = value;
  });
  return changed;
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = String(value);
  return element.innerHTML;
}

function setConnection(state) {
  clearTimeout(connectionTimer);
  if (state === "connected") {
    socketWasOpen = true;
    if (connectionState === "connected") return;
    connectionTimer = setTimeout(() => {
      connectionState = "connected";
      renderDriveStatus();
      maybeVibrateForPermission();
    }, 2500);
    return;
  }
  if (socketWasOpen) {
    randomizeOfflineShape(document.querySelector("#connection-dot"));
    socketWasOpen = false;
  }
  if (connectionState === state) return;
  connectionState = state;
  renderDriveStatus();
}

function randomizeOfflineShape(dot) {
  const random = (min, max) => min + Math.random() * (max - min);
  const horizontalRadii = Array.from({ length: 4 }, () => random(18, 86));
  const verticalRadii = Array.from({ length: 4 }, () => random(24, 78));
  dot.style.setProperty(
    "--offline-radius",
    `${horizontalRadii.map(value => `${value}%`).join(" ")} / ${verticalRadii.map(value => `${value}%`).join(" ")}`
  );
  dot.style.setProperty("--offline-tilt", "0deg");
  dot.style.setProperty("--offline-skew", "0deg");
  dot.style.setProperty("--offline-wide", random(1.55, 1.9));
  dot.style.setProperty("--offline-flat", random(.24, .33));
}

function permissionRestRadius() {
  return "50% 50% 50% 50% / 50% 50% 50% 50%";
}

function randomizePermissionShape(dot) {
  const pointCount = 12;
  const phases = Array.from({ length: pointCount }, () => Math.random() * Math.PI * 2);
  const amplitudes = Array.from({ length: pointCount }, () => 10 + Math.random() * 4);
  dot.style.setProperty("--permission-rest-radius", permissionRestRadius());
  dot.style.setProperty("--permission-frame-0", permissionRestRadius());

  for (let frame = 1; frame < pointCount; frame += 1) {
    const pointOffsets = phases.map((phase, index) => (
      amplitudes[index] * Math.sin((frame / pointCount) * Math.PI * 2 + phase)
    ));
    const cornerRadii = Array.from({ length: 8 }, (_, index) => {
      const pointPosition = (index * pointCount) / 8;
      const lowerIndex = Math.floor(pointPosition);
      const upperIndex = (lowerIndex + 1) % pointCount;
      const fraction = pointPosition - lowerIndex;
      return 50 + pointOffsets[lowerIndex] * (1 - fraction) + pointOffsets[upperIndex] * fraction;
    });
    const horizontal = cornerRadii.slice(0, 4).map(value => `${value.toFixed(2)}%`).join(" ");
    const vertical = cornerRadii.slice(4).map(value => `${value.toFixed(2)}%`).join(" ");
    dot.style.setProperty(
      `--permission-frame-${frame}`,
      `${horizontal} / ${vertical}`,
    );
  }
}

function renderDriveStatus() {
  const dot = document.querySelector("#connection-dot");
  const disconnected = connectionState !== "connected";
  const visualStatus = disconnected
    ? "disconnected"
    : showActPulse && assistantStatus === "idle"
      ? "act"
      : assistantStatus === "background_task_running"
        ? "background"
        : assistantStatus;
  const messageClassification = visualStatus === "idle"
    ? incomingMessageClassification || (permissionPulseTimer ? permissionMessageClassification : null)
    : null;
  const classifiedIncomingMessage =
    !disconnected && messageClassification !== null;
  const urgentIncomingMessage = classifiedIncomingMessage &&
    ["high", "critical"].includes(messageClassification.urgency);
  dot.className = `status-dot ${connectionState} assistant-${visualStatus}${permissionPulseTimer ? " assistant-ask_permission_to_talk" : ""}${classifiedIncomingMessage ? " incoming-message-classified" : ""}${urgentIncomingMessage && incomingMessageClassification ? " incoming-message-urgent" : ""}`;
  if (classifiedIncomingMessage) {
    dot.dataset.messageTone = messageClassification.tone;
    dot.dataset.messageUrgency = messageClassification.urgency;
  } else {
    delete dot.dataset.messageTone;
    delete dot.dataset.messageUrgency;
  }
  if (urgentIncomingMessage) {
    dot.setAttribute(
      "aria-label",
      `Incoming message, ${messageClassification.urgency} urgency, ${messageClassification.tone} tone`,
    );
  } else if (!disconnected && assistantStatus === "talking" && speakingTone) {
    dot.dataset.tone = speakingTone;
    dot.setAttribute("aria-label", `Assistant speaking, ${speakingTone} tone`);
  } else {
    delete dot.dataset.tone;
    dot.setAttribute("aria-label", disconnected
      ? connectionState === "offline" ? "Assistant reconnecting" : "Assistant connecting"
      : `Assistant ${visualStatus.replaceAll("_", " ")}`);
  }
}

function setSpeakingTone(tone) {
  speakingTone = tone;
  renderDriveStatus();
}

function setIncomingMessageClassification(classification) {
  clearTimeout(incomingMessageTimer);
  incomingMessageClassification = classification;
  permissionMessageClassification = classification;
  if (!classification) {
    renderDriveStatus();
    return;
  }

  if (["high", "critical"].includes(classification.urgency)) {
    const dot = document.querySelector("#connection-dot");
    randomizePermissionShape(dot);
    dot.classList.remove("incoming-message-urgent");
    void dot.offsetWidth;
  }
  renderDriveStatus();
  incomingMessageTimer = setTimeout(() => {
    incomingMessageTimer = null;
    incomingMessageClassification = null;
    if (!(assistantFlags & PERMISSION_BIT)) permissionMessageClassification = null;
    renderDriveStatus();
  }, 1960);
}

function restartPermissionPulse(dot) {
  randomizePermissionShape(dot);
  dot.classList.remove("assistant-ask_permission_to_talk");
  void dot.offsetWidth;
  dot.classList.add("assistant-ask_permission_to_talk");
  // The transparent-blob visual lives only for the pulse itself; afterwards
  // the regular dot rendering takes over again (permission-impatient runs
  // .28s x 7 iterations). Arm the pulse window before the render so
  // renderDriveStatus keeps the class.
  clearTimeout(permissionPulseTimer);
  permissionPulseTimer = setTimeout(() => {
    permissionPulseTimer = null;
    renderDriveStatus();
  }, 1960);
  renderDriveStatus();
}

// Episodic reminder: vibration only when the current visual is the idle
// fallback; any higher-priority active state keeps its own animation and
// suppresses the vibration until it ends.
function maybeVibrateForPermission() {
  if (!(assistantFlags & PERMISSION_BIT)) return;
  if (assistantStatus !== "idle" || connectionState !== "connected") return;
  restartPermissionPulse(document.querySelector("#connection-dot"));
}

function setAssistantFlags(flags) {
  const hadPermission = !!(assistantFlags & PERMISSION_BIT);
  assistantFlags = flags;
  const hasPermission = !!(flags & PERMISSION_BIT);
  if (hasPermission && !permissionReminderTimer) {
    permissionReminderTimer = setInterval(
      maybeVibrateForPermission, permissionReminderSeconds * 1000,
    );
    maybeVibrateForPermission();
  } else if (!hasPermission && hadPermission) {
    clearInterval(permissionReminderTimer);
    permissionReminderTimer = null;
    permissionMessageClassification = null;
    clearTimeout(permissionPulseTimer);
    permissionPulseTimer = null;
    renderDriveStatus();
  }
}

function setAssistantStatus(status, flags) {
  if (flags !== undefined) setAssistantFlags(flags);
  assistantStatus = status;
  // When the assistant pauses for low attention, close the current bubble
  // with an ellipsis; the resumed speech opens a fresh bubble.
  if (status === "paused_attention" && !assistantTurnPaused) {
    assistantTurnPaused = true;
    if (pendingAssistantMessage !== null) {
      const bubble = pendingAssistantMessage.querySelector(".bubble");
      const text = bubble.textContent.trim();
      if (text) bubble.textContent = `${text} …`;
      pendingAssistantMessage = null;
    }
  }
  if (status === "act") {
    showActPulse = true;
    clearTimeout(actPulseTimer);
    actPulseTimer = setTimeout(() => {
      showActPulse = false;
      renderDriveStatus();
    }, 760);
  } else if (status !== "idle") {
    showActPulse = false;
    clearTimeout(actPulseTimer);
    // A higher-priority state takes over the visual: end any permission pulse.
    clearTimeout(permissionPulseTimer);
    permissionPulseTimer = null;
  }
  renderDriveStatus();
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    maybeVibrateForPermission();
  }
});

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2600);
}

function setFriendSimulation(active) {
  friendSimulationActive = active;
  const button = document.querySelector('#scenario-button[data-simulation="friend-messages"], #friend-simulation-toggle[data-simulation="friend-messages"]');
  if (!button) return;
  button.textContent = active ? "Stop simulation" : "Start simulation";
  button.setAttribute("aria-pressed", String(active));
}

function setScenarioActive(scenarioId, active) {
  if (active) {
    activeScenario = scenarioId;
  } else if (activeScenario === scenarioId || scenarioId === "") {
    activeScenario = null;
  }
  controlContent.querySelectorAll(".scenario-button").forEach(btn => {
    if (!btn.dataset.simulation) {
      btn.setAttribute("aria-pressed", String(btn.dataset.scenario === activeScenario));
    }
  });
}

function formatValue(value, format) {
  const number = Number(value);
  if (format === "percent") return `${Math.round(number * 100)}%`;
  if (format === "temp") return `${number.toFixed(1)} C`;
  if (format === "speed") return `${number.toFixed(0)} km/h`;
  if (format === "integer") return String(Math.round(number));
  return number.toFixed(2);
}

function renderScenarioChoices(group) {
  const current = controlState[controlStateKey(group.cls, group.key)] ?? group.value;
  const options = group.options.map(option => `<button type="button" data-choice="${escapeHtml(String(option.value))}" aria-pressed="${option.value === current}">${escapeHtml(option.label)}</button>`).join("");
  return `<section class="control-section" data-type="choices" data-class="${group.cls}" data-key="${group.key}" data-value-kind="${typeof group.value}" data-default="${group.value}" data-clearable="${Boolean(group.clearable)}"><h3>${group.title}</h3><div class="segmented" role="group" aria-label="${group.title}">${options}</div></section>`;
}

function renderMeetingChoices(group) {
  const status = controlState[controlStateKey(group.cls, "meeting_status")];
  const timing = controlState[controlStateKey(group.cls, "meeting_timing")];
  const options = group.controls.map(control => {
    const current = control.timing ? timing === control.timing : status === control.status;
    const payload = control.timing ? { timing: control.timing } : { status: control.status };
    return `<button type="button" data-meeting='${escapeHtml(JSON.stringify(payload))}' aria-pressed="${current}">${escapeHtml(control.label)}</button>`;
  }).join("");
  return `<section class="control-section" data-type="meeting" data-class="${group.cls}"><h3>${group.title}</h3><div class="segmented" role="group" aria-label="${group.title}">${options}</div></section>`;
}

function renderControls(tabName) {
  controlContent.innerHTML = CONTROL_GROUPS[tabName].map((group, groupIndex) => {
    if (group.type === "text") {
      const current = controlState[controlStateKey(group.cls, group.key)] ?? group.value;
      return `<section class="control-section" data-type="text" data-class="${group.cls}"><label class="preference-field" for="${tabName}-${group.key}"><span>${group.title}</span><input id="${tabName}-${group.key}" type="text" data-key="${group.key}" maxlength="${group.maxLength}" value="${escapeHtml(current)}" autocomplete="given-name"></label></section>`;
    }
    if (group.type === "activities") {
      const value = controlState[controlStateKey(group.cls, group.key)] ?? group.value;
      const options = group.options.map(option => `<label><input type="checkbox" value="${escapeHtml(option)}" ${value === option ? "checked" : ""}><span>${escapeHtml(option)}</span></label>`).join("");
      return `<section class="control-section" data-type="activities" data-class="${group.cls}" data-key="${group.key}"><h3>${group.title}</h3><div class="segmented">${options}<button id="detect-activity" class="scenario-button" type="button" aria-pressed="${activityDetectionBusy}" ${activityDetectionBusy ? "disabled" : ""}>${activityDetectionBusy ? "Detecting..." : "Detect..."}</button></div></section>`;
    }
    if (group.type === "options") {
      const name = `${tabName}-${groupIndex}-${group.key}`;
      const selectedValue = controlState[controlStateKey(group.cls, group.key)] ?? group.value;
      const options = group.options.map(option => `<label><input type="radio" name="${name}" value="${escapeHtml(option)}" ${option === selectedValue ? "checked" : ""}><span>${escapeHtml(option)}</span></label>`).join("");
      return `<section class="control-section" data-type="options" data-class="${group.cls}" data-key="${group.key}"><h3>${group.title}</h3><div class="segmented">${options}</div></section>`;
    }
    if (group.type === "textarea") {
      const current = controlState[controlStateKey(group.cls, group.key)] ?? group.value ?? "";
      return `<section class="control-section" data-type="textarea" data-class="${group.cls}"><h3>${group.title}</h3><label class="preference-field" for="${tabName}-${group.key}"><span>Describe the music you enjoy</span><textarea id="${tabName}-${group.key}" data-key="${group.key}" maxlength="500" rows="4" placeholder="${escapeHtml(group.placeholder)}">${escapeHtml(current)}</textarea></label></section>`;
    }
    if (group.type === "toggles") {
      const controls = group.controls.map(([label, key, checked, on, off]) => {
        const current = controlState[controlStateKey(group.cls, key)] ?? checked;
        return `<label class="toggle-row"><span>${label}</span><span class="switch-control"><input type="checkbox" data-key="${key}" ${current ? "checked" : ""}><span class="switch-track"></span><span data-state>${current ? on : off}</span></span></label>`;
      }).join("");
      return `<section class="control-section" data-type="toggles" data-class="${group.cls}"><h3>${group.title}</h3>${controls}</section>`;
    }
    if (group.type === "scenarios") {
      const hint = group.hint ? `<p class="scenario-hint">${escapeHtml(group.hint)}</p>` : "";
      const scenarios = group.controls.map(scenario => {
        if (Array.isArray(scenario.groups)) {
          const rendered = scenario.groups.map(subGroup => subGroup.type === "meeting"
            ? renderMeetingChoices(subGroup)
            : renderScenarioChoices(subGroup)).join("");
          return `<div class="scenario-item" data-scenario="${scenario.id}"><div class="scenario-copy"><h4>${escapeHtml(scenario.label)}</h4><p>${escapeHtml(scenario.description)}</p></div><div class="scenario-controls">${rendered}</div></div>`;
        }
        const isSimulation = Boolean(scenario.simulation);
        const active = scenario.simulation === "driving-video" ? drivingSimulationActive
          : isSimulation ? friendSimulationActive : activeScenario === scenario.id;
        const buttonId = scenario.simulation === "driving-video"
          ? ' id="driving-simulation-toggle"'
          : isSimulation ? ' id="friend-simulation-toggle"' : "";
        const buttonLabel = isSimulation
          ? active ? "Stop simulation" : "Start simulation"
          : active ? "Stop scenario" : "Start scenario";
        const dataAttribute = isSimulation
          ? `data-simulation="${scenario.simulation}"`
          : `data-scenario="${scenario.id}"`;
        return `<div class="scenario-item"><div class="scenario-copy"><h4>${escapeHtml(scenario.label)}</h4><p>${escapeHtml(scenario.description)}</p></div><button type="button" class="scenario-button"${buttonId} ${dataAttribute} aria-pressed="${active}">${buttonLabel}</button></div>`;
      }).join("");
      return `<section class="control-section" data-type="scenarios"><h3>${group.title}</h3>${hint}${scenarios}${drivingSimulationActive ? `<section id="driving-video-block" class="driving-video-panel" aria-label="Driving simulation video">${DRIVING_VIDEO_BLOCK_HTML}</section>` : ""}</section>`;
    }
    const controls = group.controls.map(([label, key, min, max, value, step, format, ownClass]) => {
      const controlClass = ownClass || group.cls;
      const current = controlState[controlStateKey(controlClass, key)] ?? value;
      return `<div class="range-row"><label for="${tabName}-${key}">${label}</label><input id="${tabName}-${key}" type="range" min="${min}" max="${max}" value="${current}" step="${step}" data-key="${key}" data-class="${controlClass}" data-format="${format}" data-integer="${group.integer || format === "integer"}"><output>${formatValue(current, format)}</output></div>`;
    }).join("");
    const sectionClass = group.cls ? ` data-class="${group.cls}"` : "";
    return `<section class="control-section" data-type="ranges"${sectionClass}><h3>${group.title}</h3>${controls}</section>`;
  }).join("");
  if (tabName === "scenario") setFriendSimulation(friendSimulationActive);
}

function updateControlsInPlace(className, values) {
  const sections = [...controlContent.querySelectorAll(".control-section[data-class]")]
    .filter(section => section.dataset.class === className);

  Object.entries(values || {}).forEach(([key, value]) => {
    sections.forEach(section => {
      if (section.dataset.type === "meeting" && ["meeting_status", "meeting_timing"].includes(key)) {
        const status = controlState[controlStateKey(className, "meeting_status")];
        const timing = controlState[controlStateKey(className, "meeting_timing")];
        section.querySelectorAll("button[data-meeting]").forEach(button => {
          const payload = JSON.parse(button.dataset.meeting);
          const pressed = payload.timing ? timing === payload.timing : status === payload.status;
          button.setAttribute("aria-pressed", String(pressed));
        });
      } else if (section.dataset.type === "choices" && section.dataset.key === key) {
        section.querySelectorAll("button[data-choice]").forEach(button => {
          button.setAttribute("aria-pressed", String(button.dataset.choice === String(value)));
        });
      } else if (section.dataset.type === "ranges") {
        const inputElement = [...section.querySelectorAll('input[type="range"]')]
          .find(input => input.dataset.key === key);
        if (!inputElement) return;
        inputElement.value = value;
        inputElement.nextElementSibling.value = formatValue(value, inputElement.dataset.format);
      } else if (section.dataset.type === "toggles") {
        const inputElement = [...section.querySelectorAll('input[type="checkbox"]')]
          .find(input => input.dataset.key === key);
        if (!inputElement) return;
        inputElement.checked = Boolean(value);
        const control = CONTROL_GROUPS.vehicle
          .filter(group => group.type === "toggles" && group.cls === className)
          .flatMap(group => group.controls)
          .find(item => item[1] === key);
        if (control) {
          inputElement.closest(".switch-control").querySelector("[data-state]").textContent =
            value ? control[3] : control[4];
        }
      } else if (section.dataset.type === "activities" && section.dataset.key === key) {
        section.querySelectorAll('input[type="checkbox"]').forEach(input => {
          input.checked = value === input.value;
        });
      } else if (section.dataset.type === "options" && section.dataset.key === key) {
        section.querySelectorAll('input[type="radio"]').forEach(input => {
          input.checked = input.value === value;
        });
      } else if (section.dataset.type === "textarea") {
        const inputElement = section.querySelector("textarea");
        if (inputElement) inputElement.value = value ?? "";
      } else if (section.dataset.type === "text") {
        const inputElement = section.querySelector('input[type="text"]');
        if (inputElement) inputElement.value = value ?? "";
      }
    });
  });
}

function sendProgressive(inputElement, fromValue, toValue) {
  const steps = 5;
  const isInteger = inputElement.dataset.integer === "true";
  if (fromValue === toValue) return;
  for (let index = 1; index <= steps; index += 1) {
    setTimeout(() => {
      let value = fromValue + ((toValue - fromValue) * index / steps);
      value = isInteger ? Math.round(value) : value;
      const method = isInteger ? "intChanged" : "floatChanged";
      const request = inputElement.dataset.class === "DriverPreferences"
        ? bridge.call("setDriverPreference", inputElement.dataset.key, value)
        : bridge.call(method, inputElement.dataset.class, inputElement.dataset.key, value);
      request.catch(() => {});
    }, index * 150);
  }
}

controlContent.addEventListener("pointerdown", event => {
  if (event.target.matches('input[type="range"]')) event.target.dataset.start = event.target.value;
});
controlContent.addEventListener("click", event => {
  if (event.target.closest("#detect-activity")) detectCabinActivity();
});
controlContent.addEventListener("input", event => {
  if (event.target.matches('input[type="range"]')) {
    const value = Number(event.target.value);
    event.target.nextElementSibling.value = formatValue(value, event.target.dataset.format);
    controlState[controlStateKey(event.target.dataset.class, event.target.dataset.key)] = value;
    if (event.target.dataset.class === "VehicleState" && event.target.dataset.key === "audio_volume") updateMusicVolume();
  }
  if (event.target.matches("#driving-video-seek")) scrubDrivingVideo(Number(event.target.value));
});
controlContent.addEventListener("change", event => {
  const target = event.target;
  if (target.matches('input[type="radio"]')) {
    const section = target.closest(".control-section");
    controlState[controlStateKey(section.dataset.class, section.dataset.key)] = target.value;
    bridge.call("stringChanged", section.dataset.class, section.dataset.key, target.value).catch(() => {});
  } else if (target.matches('input[type="checkbox"]')) {
    const section = target.closest(".control-section");
    if (section.dataset.type === "activities") {
      const activity = target.checked ? target.value : "No activity detected";
      section.querySelectorAll('input[type="checkbox"]').forEach(input => {
        input.checked = input.value === activity;
      });
      controlState[controlStateKey(section.dataset.class, section.dataset.key)] = activity;
      bridge.call("stringChanged", section.dataset.class, section.dataset.key, activity).catch(() => {});
      return;
    }
    const control = CONTROL_GROUPS.vehicle
      .filter(item => item.type === "toggles" && item.cls === section.dataset.class)
      .flatMap(item => item.controls)
      .find(item => item[1] === target.dataset.key);
    controlState[controlStateKey(section.dataset.class, target.dataset.key)] = target.checked;
    target.closest(".switch-control").querySelector("[data-state]").textContent = target.checked ? control[3] : control[4];
    bridge.call("boolChanged", section.dataset.class, target.dataset.key, target.checked).catch(() => {});
  } else if (target.matches('textarea[data-key], input[type="text"][data-key]')) {
    const section = target.closest(".control-section");
    controlState[controlStateKey(section.dataset.class, target.dataset.key)] = target.value;
    bridge.call("setDriverPreference", target.dataset.key, target.value).catch(() => {});
  } else if (target.matches('input[type="range"]')) {
    sendProgressive(target, Number(target.dataset.start ?? target.defaultValue), Number(target.value));
  }
});

controlContent.addEventListener("pointerdown", event => {
  if (event.target.matches("#driving-video-seek")) startDrivingVideoScrub();
});
controlContent.addEventListener("pointerup", event => {
  if (event.target.matches("#driving-video-seek")) endDrivingVideoScrub();
});
controlContent.addEventListener("pointercancel", event => {
  if (event.target.matches("#driving-video-seek")) endDrivingVideoScrub();
});

controlContent.addEventListener("click", event => {
  const meetingButton = event.target.closest("button[data-meeting]");
  if (meetingButton) {
    const section = meetingButton.closest(".control-section");
    const wasPressed = meetingButton.getAttribute("aria-pressed") === "true";
    const payload = JSON.parse(meetingButton.dataset.meeting);
    const key = payload.timing ? "meeting_timing" : "meeting_status";
    const requestedValue = wasPressed ? "none" : payload.timing || payload.status;
    const request = payload.timing && !wasPressed
      ? bridge.call("setScenario", "late_for_meeting", true)
      : bridge.call("stringChanged", section.dataset.class, key, requestedValue);
    meetingButton.disabled = true;
    request.then(appliedValue => {
      const state = { [key]: appliedValue ?? requestedValue };
      rememberControlState(section.dataset.class, state);
      updateControlsInPlace(section.dataset.class, state);
    })
      .catch(() => showToast("Could not update meeting context"))
      .finally(() => { meetingButton.disabled = false; });
    return;
  }
  const choice = event.target.closest("button[data-choice]");
  if (choice) {
    const section = choice.closest(".control-section");
    const booleanValue = section.dataset.valueKind === "boolean";
    const value = booleanValue ? choice.dataset.choice === "true"
      : section.dataset.clearable === "true" && choice.getAttribute("aria-pressed") === "true"
        ? "none" : choice.dataset.choice;
    const buttons = section.querySelectorAll("button[data-choice]");
    buttons.forEach(button => { button.disabled = true; });
    bridge.call(booleanValue ? "boolChanged" : "stringChanged", section.dataset.class, section.dataset.key, value)
      .then(appliedValue => {
        const values = { [section.dataset.key]: appliedValue ?? value };
        rememberControlState(section.dataset.class, values);
        updateControlsInPlace(section.dataset.class, values);
      })
      .catch(() => showToast("Could not update meeting context"))
      .finally(() => buttons.forEach(button => { button.disabled = false; }));
    return;
  }
  if (event.target.closest("#driving-video-toggle")) { sendDrivingVideoCommand("togglePlay"); return; }
  if (event.target.closest("#driving-video-restart")) { sendDrivingVideoCommand("seekTo", 0); return; }
  if (event.target.closest("#driving-video-load")) {
    const input = document.querySelector("#driving-video-url");
    const requested = (input?.value || "").trim();
    bridge.call("startDrivingSimulation", requested || null)
      .then(name => {
        if (!name) { showToast("Recording not found in recordings"); return; }
        if (input) {
          input.value = "";
          input.placeholder = name;
        }
        showToast("Driving video loaded");
      })
      .catch(() => showToast("Could not load recording"));
    return;
  }
  const button = event.target.closest(".scenario-button");
  if (!button) return;
  if (button.dataset.simulation === "friend-messages") {
    const wasActive = friendSimulationActive;
    button.disabled = true;
    bridge.call(
      wasActive ? "stopMessageSimulation" : "startMessageSimulation"
    ).then(() => setFriendSimulation(!wasActive))
      .catch(() => showToast("Could not update friend message simulation"))
      .finally(() => { button.disabled = false; });
    return;
  }
  if (button.dataset.simulation === "driving-video") {
    button.disabled = true;
    const requestedUrl = (document.querySelector("#driving-video-url")?.value || "").trim() || drivingVideoUrl;
    const request = drivingSimulationActive
      ? bridge.call("stopDrivingSimulation")
      : bridge.call("startDrivingSimulation", requestedUrl);
    request
      .catch(() => showToast("Could not update driving simulation"))
      .finally(() => { button.disabled = false; });
    return;
  }
  const scenarioId = button.dataset.scenario;
  const wasActive = activeScenario === scenarioId;
  button.disabled = true;
  bridge.call("setScenario", scenarioId, !wasActive)
    .then(() => {
      setScenarioActive(wasActive ? "" : scenarioId, !wasActive);
      showToast(wasActive ? `Scenario stopped: ${button.textContent}` : `Scenario triggered: ${button.textContent}`);
    })
    .catch(() => {})
    .finally(() => { button.disabled = false; });
});

document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
  document.querySelector(".tab.active").classList.remove("active");
  document.querySelectorAll(".tab").forEach(item => item.setAttribute("aria-selected", "false"));
  tab.classList.add("active");
  tab.setAttribute("aria-selected", "true");
  activeTab = tab.dataset.tab;
  renderControls(activeTab);
}));

document.querySelector("#reset-controls").addEventListener("click", () => {
  controlState = {};
  renderControls(activeTab);
  controlContent.querySelectorAll('.control-section[data-type="choices"]').forEach(section => {
    const booleanValue = section.dataset.valueKind === "boolean";
    const value = booleanValue ? section.dataset.default === "true" : section.dataset.default;
    bridge.call(booleanValue ? "boolChanged" : "stringChanged", section.dataset.class, section.dataset.key, value).catch(() => {});
  });
  controlContent.querySelectorAll('input[type="radio"]:checked').forEach(target => {
    const section = target.closest(".control-section");
    bridge.call("stringChanged", section.dataset.class, section.dataset.key, target.value).catch(() => {});
  });
  controlContent.querySelectorAll('input[type="checkbox"]').forEach(target => {
    const section = target.closest(".control-section");
    if (section.dataset.type === "activities") return;
    bridge.call("boolChanged", section.dataset.class, target.dataset.key, target.checked).catch(() => {});
  });
  controlContent.querySelectorAll('[data-type="activities"]').forEach(() => {
    bridge.call("stringChanged", "DriverPhysicalState", "activity", "No activity detected").catch(() => {});
  });
  controlContent.querySelectorAll('input[type="range"]').forEach(target => {
    const value = Number(target.value);
    const request = target.dataset.class === "DriverPreferences"
      ? bridge.call("setDriverPreference", target.dataset.key, value)
      : bridge.call(
        target.dataset.integer === "true" ? "intChanged" : "floatChanged",
        target.dataset.class,
        target.dataset.key,
        value
      );
    request.catch(() => {});
  });
  controlContent.querySelectorAll("textarea[data-key]").forEach(target => {
    bridge.call("setDriverPreference", target.dataset.key, target.value).catch(() => {});
  });
  showToast("Controls reset");
});

document.querySelector("#processing-toggle").addEventListener("change", event => {
  bridge.call("eventProcessingChanged", event.target.checked).catch(() => {});
});
document.querySelector("#urgency-select").addEventListener("change", event => {
  bridge.call("minimumUrgencyChanged", event.target.value).catch(() => {});
});
duplicateSuppressionToggle.addEventListener("change", event => {
  bridge.call("duplicateSuppressionChanged", event.target.checked).catch(() => {});
});

function addMessage(role, text) {
  if (role === "user") { pendingAssistantMessage = null; assistantTurnPaused = false; }
  welcomeState?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `<div class="meta">${role === "user" ? "You" : "Drive assistant"}</div><div class="bubble"></div>`;
  article.querySelector(".bubble").textContent = text;
  conversation.append(article);
  clearConversationButton.disabled = false;
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

function clearConversationView() {
  conversation.replaceChildren(welcomeState);
  pendingAssistantMessage = null;
  input.value = "";
  input.style.height = "auto";
  clearConversationButton.disabled = true;
}

function updateAssistantMessage(text, final = false) {
  // After an attention pause the turn already shows two bubbles (the paused
  // one closed with an ellipsis and the resumed one): the final full text
  // would only duplicate them, so acknowledge it and stop.
  if (final && assistantTurnPaused) {
    assistantTurnPaused = false;
    return;
  }
  if (pendingAssistantMessage === null) {
    pendingAssistantMessage = addMessage("assistant", text);
  } else {
    pendingAssistantMessage.querySelector(".bubble").textContent = text;
    conversation.scrollTop = conversation.scrollHeight;
  }
  if (final) pendingAssistantMessage = null;
}

function sendMessage(text) {
  const message = text.trim();
  if (!message) return;
  addMessage("user", message);
  bridge.call("userInput", message).catch(() => {});
  input.value = "";
  input.style.height = "auto";
}

document.querySelector("#composer").addEventListener("submit", event => {
  event.preventDefault();
  sendMessage(input.value);
});
clearConversationButton.addEventListener("click", () => {
  bridge.call("clearConversation").catch(() => {});
});
input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage(input.value);
  }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 110)}px`;
});
document.querySelectorAll(".suggestions button").forEach(button => button.addEventListener("click", () => sendMessage(button.textContent)));

function setConversation(active) {
  conversationActive = active;
  conversationButton.classList.toggle("active", active);
  conversationButton.setAttribute("aria-pressed", String(active));
  conversationButton.title = active ? "Stop conversation" : "Start conversation";
  conversationButton.setAttribute("aria-label", conversationButton.title);
  document.querySelector("#listening-badge").hidden = !active;
}
conversationButton.addEventListener("click", () => {
  bridge.call(conversationActive ? "stopConversation" : "startConversation").catch(() => {});
  if (conversationActive) setConversation(false);
});

function renderKnowledge(rawKnowledge) {
  const container = document.querySelector("#knowledge-content");
  let knowledge = rawKnowledge;
  if (typeof rawKnowledge === "string") {
    try { knowledge = JSON.parse(rawKnowledge); } catch (_) {}
  }
  if (typeof knowledge !== "object" || knowledge === null) {
    container.innerHTML = `<div class="knowledge-group"><div class="knowledge-value">${escapeHtml(rawKnowledge)}</div></div>`;
    return;
  }
  const formatLabel = label => label
    .replaceAll("_", " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\./g, " ");
  const groups = new Map();

  Object.entries(knowledge).forEach(([key, value]) => {
    if (typeof value === "object" && value !== null) {
      groups.set(formatLabel(key), Object.entries(value));
      return;
    }

    const [groupName, ...fieldParts] = key.split(".");
    const displayGroup = formatLabel(groupName);
    const displayField = formatLabel(fieldParts.join(".") || "Current status");
    groups.set(displayGroup, [...(groups.get(displayGroup) || []), [displayField, value]]);
  });

  container.innerHTML = [...groups.entries()].map(([groupName, items]) => {
    const rows = items.map(([, value]) => {
      const displayValue = typeof value === "object" ? JSON.stringify(value) : value;
      return `<div class="knowledge-item"><span class="knowledge-value">${escapeHtml(displayValue)}</span></div>`;
    }).join("");
    return `<section class="knowledge-group"><h3>${escapeHtml(groupName)}</h3>${rows}</section>`;
  }).join("");
}

function renderChangedEvents(changes) {
  const container = document.querySelector("#changed-events");
  if (!Array.isArray(changes) || changes.length === 0) return;

  container.querySelector(".changed-empty")?.remove();
  [...changes].reverse().forEach(change => {
    const item = document.createElement("div");
    item.className = "changed-event";
    item.textContent = String(change);
    container.prepend(item);
    setTimeout(() => {
      item.classList.add("is-expiring");
      setTimeout(() => item.remove(), 200);
    }, 1800);
  });

  while (container.children.length > 6) {
    container.lastElementChild?.remove();
  }
}

bridge.on("ready", data => {
  if (data.assistantStatus) setAssistantStatus(data.assistantStatus);
  renderKnowledge(data.knowledge);
  rememberControlState("DriverPreferences", data.driverPreferences);
  rememberControlState("DriverAgenda", data.driverAgendaState);
  rememberControlState("VehicleState", data.vehicleState);
  rememberControlState("DriverPhysicalState", data.driverPhysicalState);
  setActivityDetectionBusy(Boolean(data.activityDetectionBusy));
  const ducking = Number(data.musicDuckingFactor);
  musicDuckingFactor = Number.isFinite(ducking) && ducking > 0 && ducking <= 1 ? ducking : 0.25;
  if (Number.isFinite(Number(data.permissionReminderSeconds)) && Number(data.permissionReminderSeconds) > 0) {
    permissionReminderSeconds = Number(data.permissionReminderSeconds);
  }
  updateMusicVolume();
  renderMusic(data.musicState);
  rememberControlState("DetectedObjects", data.detectedObjectsState);
  setFriendSimulation(Boolean(data.friendMessageSimulationActive));
  duplicateSuppressionToggle.checked = Boolean(data.duplicateSuppressionEnabled);
  if (["driver", "vehicle", "scenario"].includes(activeTab)) renderControls(activeTab);
  conversationButton.disabled = !data.sttEnabled;
  bridge.call("eventProcessingChanged", document.querySelector("#processing-toggle").checked).catch(() => {});
  bridge.call("minimumUrgencyChanged", document.querySelector("#urgency-select").value).catch(() => {});
});
bridge.on("sttEnabledChanged", enabled => {
  conversationButton.disabled = !enabled[0];
});
bridge.on("responseUpdated", data => updateAssistantMessage(data[0]));
bridge.on("responseReceived", data => updateAssistantMessage(data[0], true));
bridge.on("conversationCleared", clearConversationView);
bridge.on("speakingToneChanged", data => { setSpeakingTone(data[0]); updateMusicVolume(); });
bridge.on("incomingMessageClassified", data => setIncomingMessageClassification(data[0]));
bridge.on("assistantStatusChanged", data => {
  // Broadcast payloads wrap bridge args in a list; the first arg carries
  // {status, flags} from the bridge.
  const [payload] = data;
  setAssistantStatus(payload.status, payload.flags);
  updateMusicVolume();
});
bridge.on("musicUpdated", data => renderMusic(data[0]));
bridge.on("friendMessageSimulationChanged", data => setFriendSimulation(Boolean(data[0])));
bridge.on("drivingSimulationChanged", data => {
  const state = data[0] || {};
  const url = (state.video_url || "").trim();
  const active = Boolean(state.active);
  // Re-render first so the video block (and its frame holder) exists.
  setDrivingSimulation(active);
  if (active && url) loadDrivingVideo(url);
});
bridge.on("scenarioCompleted", data => setScenarioActive("", false));
bridge.on("userSpeechReceived", data => addMessage("user", data[0]));
bridge.on("knowledgeUpdated", data => renderKnowledge(data[0]));
bridge.on("knowledgeChanged", data => renderChangedEvents(data[0]));
bridge.on("vehicleStateChanged", data => {
  const changed = rememberControlState("VehicleState", data[0]);
  updateMusicVolume();
  if (changed) updateControlsInPlace("VehicleState", data[0]);
});
bridge.on("detectedObjectsStateChanged", data => {
  const changed = rememberControlState("DetectedObjects", data[0]);
  if (changed) updateControlsInPlace("DetectedObjects", data[0]);
});
bridge.on("driverPreferencesChanged", data => {
  const changed = rememberControlState("DriverPreferences", data[0]);
  if (changed) updateControlsInPlace("DriverPreferences", data[0]);
});
bridge.on("driverAgendaChanged", data => {
  const changed = rememberControlState("DriverAgenda", data[0]);
  if (changed) updateControlsInPlace("DriverAgenda", data[0]);
});
bridge.on("conversationModeChanged", data => {
  setConversation(Boolean(data[0]));
  if (!data[0]) showToast("Conversation stopped");
});

renderControls(activeTab);
randomizeOfflineShape(document.querySelector("#connection-dot"));
bridge.connect();
