const CONTROL_GROUPS = {
  driver: [
    { title: "Activity", type: "options", cls: "DriverPhysicalState", key: "activity", value: "Idle", options: ["Idle", "Talking", "Driving", "Eating", "On the phone", "Sleeping", "About to exit"] },
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
    ]}
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
      ["Engine", "engine_on", false, "On", "Off"], ["Doors", "doors_locked", false, "Locked", "Unlocked"], ["Trunk", "trunk_open", false, "Open", "Closed"]
    ]},
    { title: "Motion & comfort", type: "ranges", controls: [
      ["Cabin temp.", "internal_temperature", 10, 40, 22, .5, "temp", "VehicleState"],
      ["Speed", "speed", 0, 200, 0, 1, "speed", "VehicleMotion"]
    ]},
    { title: "Detected objects", type: "ranges", cls: "DetectedObjects", integer: true, controls: [
      ["People around", "people_around", 0, 20, 0, 1, "integer"],
      ["Vehicles around", "vehicles_around", 0, 20, 0, 1, "integer"],
      ["Dangers around", "dangerous_objects_around", 0, 20, 0, 1, "integer"]
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
let activeTab = "driver";
let conversationActive = false;
let pendingAssistantMessage = null;
let toastTimer;

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = String(value);
  return element.innerHTML;
}

function setConnection(state) {
  const dot = document.querySelector("#connection-dot");
  const label = document.querySelector("#connection-label");
  dot.className = `status-dot ${state === "connecting" ? "" : state}`;
  label.textContent = state === "connected" ? "Assistant online" : state === "offline" ? "Reconnecting" : "Connecting";
}

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2600);
}

function formatValue(value, format) {
  const number = Number(value);
  if (format === "percent") return `${Math.round(number * 100)}%`;
  if (format === "temp") return `${number.toFixed(1)} C`;
  if (format === "speed") return `${number.toFixed(0)} km/h`;
  if (format === "integer") return String(Math.round(number));
  return number.toFixed(2);
}

function renderControls(tabName) {
  controlContent.innerHTML = CONTROL_GROUPS[tabName].map((group, groupIndex) => {
    if (group.type === "options") {
      const name = `${tabName}-${groupIndex}-${group.key}`;
      const options = group.options.map(option => `<label><input type="radio" name="${name}" value="${escapeHtml(option)}" ${option === group.value ? "checked" : ""}><span>${escapeHtml(option)}</span></label>`).join("");
      return `<section class="control-section" data-type="options" data-class="${group.cls}" data-key="${group.key}"><h3>${group.title}</h3><div class="segmented">${options}</div></section>`;
    }
    if (group.type === "toggles") {
      const controls = group.controls.map(([label, key, checked, on, off]) => `<label class="toggle-row"><span>${label}</span><span class="switch-control"><input type="checkbox" data-key="${key}" ${checked ? "checked" : ""}><span class="switch-track"></span><span data-state>${checked ? on : off}</span></span></label>`).join("");
      return `<section class="control-section" data-type="toggles" data-class="${group.cls}"><h3>${group.title}</h3>${controls}</section>`;
    }
    const controls = group.controls.map(([label, key, min, max, value, step, format, ownClass]) => `<div class="range-row"><label for="${tabName}-${key}">${label}</label><input id="${tabName}-${key}" type="range" min="${min}" max="${max}" value="${value}" step="${step}" data-key="${key}" data-class="${ownClass || group.cls}" data-format="${format}" data-integer="${group.integer || format === "integer"}"><output>${formatValue(value, format)}</output></div>`).join("");
    return `<section class="control-section" data-type="ranges"><h3>${group.title}</h3>${controls}</section>`;
  }).join("");
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
      bridge.call(method, inputElement.dataset.class, inputElement.dataset.key, value).catch(() => {});
    }, index * 150);
  }
}

controlContent.addEventListener("pointerdown", event => {
  if (event.target.matches('input[type="range"]')) event.target.dataset.start = event.target.value;
});
controlContent.addEventListener("input", event => {
  if (event.target.matches('input[type="range"]')) event.target.nextElementSibling.value = formatValue(event.target.value, event.target.dataset.format);
});
controlContent.addEventListener("change", event => {
  const target = event.target;
  if (target.matches('input[type="radio"]')) {
    const section = target.closest(".control-section");
    bridge.call("stringChanged", section.dataset.class, section.dataset.key, target.value).catch(() => {});
  } else if (target.matches('input[type="checkbox"]')) {
    const section = target.closest(".control-section");
    const group = CONTROL_GROUPS.vehicle.find(item => item.type === "toggles");
    const control = group.controls.find(item => item[1] === target.dataset.key);
    target.closest(".switch-control").querySelector("[data-state]").textContent = target.checked ? control[3] : control[4];
    bridge.call("boolChanged", section.dataset.class, target.dataset.key, target.checked).catch(() => {});
  } else if (target.matches('input[type="range"]')) {
    sendProgressive(target, Number(target.dataset.start ?? target.defaultValue), Number(target.value));
  }
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
  renderControls(activeTab);
  controlContent.querySelectorAll('input[type="radio"]:checked').forEach(target => {
    const section = target.closest(".control-section");
    bridge.call("stringChanged", section.dataset.class, section.dataset.key, target.value).catch(() => {});
  });
  controlContent.querySelectorAll('input[type="checkbox"]').forEach(target => {
    const section = target.closest(".control-section");
    bridge.call("boolChanged", section.dataset.class, target.dataset.key, target.checked).catch(() => {});
  });
  controlContent.querySelectorAll('input[type="range"]').forEach(target => {
    bridge.call(
      target.dataset.integer === "true" ? "intChanged" : "floatChanged",
      target.dataset.class,
      target.dataset.key,
      Number(target.value)
    ).catch(() => {});
  });
  showToast("Controls reset");
});

document.querySelector("#processing-toggle").addEventListener("change", event => {
  bridge.call("eventProcessingChanged", event.target.checked).catch(() => {});
});
document.querySelector("#urgency-select").addEventListener("change", event => {
  bridge.call("layaMinimumUrgencyChanged", event.target.value).catch(() => {});
});

function addMessage(role, text) {
  if (role === "user") pendingAssistantMessage = null;
  welcomeState?.remove();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `<div class="meta">${role === "user" ? "You" : "Drive assistant"}</div><div class="bubble"></div>`;
  article.querySelector(".bubble").textContent = text;
  conversation.append(article);
  conversation.scrollTop = conversation.scrollHeight;
  return article;
}

function updateAssistantMessage(text, final = false) {
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
    const rows = items.map(([key, value]) => {
      const displayValue = typeof value === "object" ? JSON.stringify(value) : value;
      return `<div class="knowledge-item"><span class="knowledge-key">${escapeHtml(key)}</span><span class="knowledge-value">${escapeHtml(displayValue)}</span></div>`;
    }).join("");
    return `<section class="knowledge-group"><h3>${escapeHtml(groupName)}</h3>${rows}</section>`;
  }).join("");
}

bridge.on("ready", data => {
  renderKnowledge(data.knowledge);
  conversationButton.disabled = !data.sttEnabled;
  bridge.call("eventProcessingChanged", document.querySelector("#processing-toggle").checked).catch(() => {});
  bridge.call("layaMinimumUrgencyChanged", document.querySelector("#urgency-select").value).catch(() => {});
});
bridge.on("responseUpdated", data => updateAssistantMessage(data[0]));
bridge.on("responseReceived", data => updateAssistantMessage(data[0], true));
bridge.on("userSpeechReceived", data => addMessage("user", data[0]));
bridge.on("knowledgeUpdated", data => renderKnowledge(data[0]));
bridge.on("conversationModeChanged", data => {
  setConversation(Boolean(data[0]));
  if (!data[0]) showToast("Conversation stopped");
});

renderControls(activeTab);
bridge.connect();
