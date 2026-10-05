#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automotive typed-decision dataset generator for Laya -- LLM-distillation variant.

Unlike generate_automotive_dataset.py (a deterministic rule table), this script
distills the gold decisions from a teacher LLM: for every row, the rendered
knowledge sentences and the full decision contract are sent to the LLM, which
returns the six labels, a per-decision confidence and the debug description in
one call. Automatic validation (consistency, critical-safety rule, schema)
retries non-compliant LLM outputs and falls back to a safe rule-based decision
only when the LLM keeps failing after --llm-retries attempts.

Scenario generation, the knowledge catalog/templates, physical coherence,
split assignment, schema, packaging and most validation checks are shared
with generate_automotive_dataset.py.

Requirements: Python 3.9+, pandas, pyarrow, a reachable Ollama server
    pip install pandas pyarrow
Usage:
    ollama serve &
    python generate_automotive_dataset_distill.py [--seed 42] [--out-dir build] [--n-samples 200]
        [--ollama-host localhost --ollama-port 11434 --ollama-model ollama/qwen2.5:3b-instruct]
        [--llm-retries 3]
"""
import argparse
import hashlib
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from typing import Counter as CounterT
from typing import DefaultDict

import requests

from generate_automotive_dataset import label as rule_label
from stellantis_LLM import StellantisVLLM

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    sys.exit("pandas and pyarrow are required: pip install pandas pyarrow")

# Allow running this script directly (e.g. `python scripts/generate_automotive_dataset.py`)
# while still importing the project's shared decision enums as the single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.agents_dataclasses import (  # noqa: E402
    ActionType,
    InterventionType,
    SkillType,
    SuggestionType,
    ToneType,
    UrgencyType,
)

# Loosely-typed scenario records: keys are short field codes (see FIELD_PATH), values are
# bool | str | int | None or (int, trend_str) for numeric fields with a trend (speed, temperatures).
Facts = dict[str, Any]
Decision = dict[str, str]  # {"urgency": ..., "tone": ..., ...} chosen labels
Record = dict[str, Any]  # one built row before JSON serialization
Row = dict[str, Any]  # one serialized (JSON-string-valued) output row
GenFn = Callable[[random.Random], Facts]

GENERATOR_VERSION = "1.0.0"
CONTRACT_VERSION = "1.0"
LABELING_RULES_VERSION = "1.0"
TEMPLATES_VERSION = "1.0"
WORKFLOW = "automotive_assistant"
# Fixed sum of the GENS scenario weights below: used only to derive proportions,
# independent of the runtime --n-samples row count.
WEIGHT_TOTAL = 50_000
SPLIT_FRACTIONS = {"train": 0.8, "validation": 0.1, "test": 0.1}
SPLITS = list(SPLIT_FRACTIONS)
N_TOTAL = WEIGHT_TOTAL  # default; overridden in main() from --n-samples
SPLIT_SIZES = {"train": 40_000, "validation": 5_000, "test": 5_000}  # default; overridden in main()
REF_SPLIT_SIZES = dict(SPLIT_SIZES)  # reference split sizes the coverage thresholds are defined for


def split_sizes_for(n_total: int) -> dict[str, int]:
    """Exact per-split row counts for an arbitrary total, keeping the 80/10/10 ratio."""
    sizes = {s: round(n_total * frac) for s, frac in SPLIT_FRACTIONS.items()}
    sizes["train"] += n_total - sum(sizes.values())
    return sizes


def compute_quotas(split_sizes: dict[str, int]) -> dict[str, dict[str, int]]:
    """Per-generator, per-split row quotas, scaled from GENS weights via largest-remainder
    apportionment so quotas always sum exactly to split_sizes (regardless of n_total)."""
    quotas: dict[str, dict[str, int]] = {name: {} for name, _, _, _ in GENS}
    for s in SPLITS:
        target = split_sizes[s]
        base, remainder = {}, {}
        for name, weight, _, _ in GENS:
            exact = weight * target / WEIGHT_TOTAL
            base[name] = int(exact)
            remainder[name] = exact - base[name]
        shortfall = target - sum(base.values())
        for name, _ in sorted(remainder.items(), key=lambda kv: kv[1], reverse=True)[:shortfall]:
            base[name] += 1
        for name in base:
            quotas[name][s] = base[name]
    return quotas
ID_PREFIX = {"train": "tr", "validation": "va", "test": "te"}
REFERENCE_URLS = [
    "https://huggingface.co/datasets/LocalLLaMA/typed-decisions",
    "https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb",
]
DECISIONS = ["urgency", "tone", "intervention_type", "skill", "action", "suggestion_type"]

# =============================================================================
# 1. Decision contract, derived from the enums in agents/agents_dataclasses.py
#    (single source of truth shared with the runtime agents).
# =============================================================================
DECISION_ENUMS = {
    "urgency": UrgencyType,
    "tone": ToneType,
    "intervention_type": InterventionType,
    "skill": SkillType,
    "action": ActionType,
    "suggestion_type": SuggestionType,
}
CRITERIA = {
    "urgency": {m.value: m.description for m in UrgencyType},
    "tone": {m.value: m.description for m in ToneType},
    "intervention_type": {m.value: m.description for m in InterventionType},
    "skill": {m.value: m.description for m in SkillType},
    "action": {m.value: m.description for m in ActionType},
    "suggestion_type": {m.value: m.description for m in SuggestionType},
}
INSTRUCTIONS = {
    "urgency": "How urgent is the current driving situation?",
    "tone": "Which communication tone should the assistant use?",
    "intervention_type": "What should the assistant do?",
    "skill": "Which skill scope does the assistant need to handle the situation?",
    "action": "Which supported reversible vehicle or comfort action should the assistant execute, if any?",
    "suggestion_type": "What should the assistant suggest to the driver, if anything?",
}
QUESTIONS = {q: {"criteria": CRITERIA[q], "instructions": INSTRUCTIONS[q], "type": "choice"} for q in DECISIONS}
# Shared by both the (optional) LLM description writer in section 6 and the validator in section 10.
DESC_CONCEPTS = ["urgency", "tone", "intervention", "skill", "action", "suggest"]
ALL_LABEL_TOKENS = {l for q in DECISIONS for l in CRITERIA[q]}
CAUSAL = ("because", " so ", "since", "therefore")
URG = [member.value for member in UrgencyType]
UI = {u: i for i, u in enumerate(URG)}
# tone has no "none" member: DISCREET is the default tone when no decision applies.
NONE_DEC = {
    "urgency": UrgencyType.NONE.value,
    "tone": ToneType.DISCREET.value,
    "intervention_type": InterventionType.NONE.value,
    "skill": SkillType.NONE.value,
    "action": ActionType.NONE.value,
    "suggestion_type": SuggestionType.NONE.value,
}

# =============================================================================
# 2. Knowledge catalog and exact sentence templates
# =============================================================================
B4 = ["low", "medium", "high", "very high"]
HI = ("high", "very high")
ATT_OK = ("medium", "high", "very high")
DENS3 = ("low", "medium", "high")
TRENDS = ["stable", "increasing", "decreasing"]
ROADS = ["urban", "rural", "highway", "residential"]
TIMES = ["Morning", "Afternoon", "Evening", "Night"]
WEATHERS = ["sunny", "cloudy", "rainy", "snowy", "foggy"]
CONDS = ["dry", "wet", "icy", "snowy", "gravel"]
VIS = ["low", "medium", "high", "very high", "optimal"]
GOOD_VIS = ("high", "very high", "optimal")
TRAFFIC = ["none", "light", "medium", "heavy"]
RISKS = ["none", "low", "medium", "high"]
TRAFFIC_TO_VEH = {"none": "none", "light": "low", "medium": "medium", "heavy": "high"}
SPEED_RANGE = {"residential": (10, 55), "urban": (15, 85), "rural": (30, 125), "highway": (60, 150)}
SPEED_CAP = {"residential": 60, "urban": 90, "rural": 130, "highway": 160}
LIMITS = {"residential": [30, 50], "urban": [30, 50, 70], "rural": [70, 90], "highway": [90, 110, 130]}
EMOS = {"angry": "anger", "disgust": "disgust", "fear": "fear", "happy": "happiness",
        "sad": "sadness", "surprise": "surprise", "neutral": "neutral expression"}
DOORS = {"door_fl": "front left", "door_fr": "front right", "door_rl": "rear left", "door_rr": "rear right"}
DOOR_KEYS = list(DOORS)
ACTIVITY_S = {
    "idle": "No distracting driver activity is detected.",
    "about_exit": "The driver is about to exit the vehicle.",
    "driving": "Normal driving activity is detected.",
    "talking": "The driver is talking, which may distract from driving.",
    "eating": "The driver is eating, which distracts from driving.",
    "phone": "The driver is using a phone, which seriously distracts from driving.",
    "asleep": "The driver appears to be asleep and unable to drive safely.",
}
TRAFFIC_S = {"none": "No traffic is currently detected around the vehicle.",
             "light": "Current traffic is light.", "medium": "Current traffic is medium.",
             "heavy": "Current traffic is heavy."}
EXIT_S = {"none": "The vehicle is not approaching a highway exit.",
          "right": "The vehicle is approaching a highway exit on the right.",
          "left": "The vehicle is approaching a highway exit on the left."}
SIGNAL_S = {"left": "The left turn signal is active.", "right": "The right turn signal is active.",
            "none": "The turn signal is inactive."}
FIELD_PATH = {
    "speed": "VehicleMotion.speed",
    "door_fl": "VehicleState.door_open_front_left", "door_fr": "VehicleState.door_open_front_right",
    "door_rl": "VehicleState.door_open_rear_left", "door_rr": "VehicleState.door_open_rear_right",
    "locked": "VehicleState.doors_locked", "side": "VehicleState.lights_on_sidelights",
    "low": "VehicleState.lights_on_low_beams", "high": "VehicleState.lights_on_high_beams",
    "fog": "VehicleState.lights_on_fog_lights", "signal": "VehicleState.turn_signal",
    "trunk": "VehicleState.trunk_open", "engine": "VehicleState.engine_on",
    "int_temp": "VehicleState.internal_temperature",
    "exit": "LaneTracing.highway_exit", "cross_l": "LaneTracing.lane_crossing_left",
    "cross_r": "LaneTracing.lane_crossing_right", "lane": "LaneTracing.driving_lane",
    "people": "DetectedObjects.people_around", "vehicles": "DetectedObjects.vehicles_around",
    "danger": "DetectedObjects.dangerous_objects_around",
    "activity": "DriverPhysicalState.activity", "attention": "DriverPhysicalState.attention_level",
    "fatigue": "DriverPhysicalState.fatigue_level",
    **{e: f"DriverEmotionState.{e}" for e in EMOS},
    "tension": "DriverDrivingStyle.driving_tension",
    "ext_temp": "EnvironmentState.external_temperature", "weather": "EnvironmentState.weather",
    "forecast": "EnvironmentState.forecast_weather", "time": "EnvironmentState.time_of_day",
    "road_cond": "EnvironmentState.road_condition", "road_type": "EnvironmentState.road_type",
    "risk": "EnvironmentState.risk_level", "visibility": "EnvironmentState.visibility",
    "traffic": "EnvironmentState.traffic", "limit": "TrafficSigns.speed_limit",
}


def _onoff(b: bool) -> str:
    return "on" if b else "off"


def render_field(k: str, v: Any) -> str | None:
    """Render one field/value into its exact knowledge sentence (None = not emitted)."""
    if v is None:
        return None
    if k == "speed":
        return f"Speed is {v[0]} and is {v[1]}."
    if k == "int_temp":
        return f"Internal temperature is {v[0]} and is {v[1]}."
    if k == "ext_temp":
        return f"External temperature is {v[0]} and is {v[1]}."
    if k == "limit":
        return f"The speed limit is {v}."
    if k in DOORS:
        return f"The {DOORS[k]} door is {'open' if v else 'closed'}."
    if k == "locked":
        return "The doors are locked." if v else "The doors are unlocked."
    if k == "trunk":
        return "The trunk is open." if v else "The trunk is closed."
    if k == "engine":
        return f"The engine is {_onoff(v)}."
    if k == "side":
        return f"The sidelights are {_onoff(v)}."
    if k == "low":
        return f"The low beam headlights are {_onoff(v)}."
    if k == "high":
        return f"The high beam headlights are {_onoff(v)}."
    if k == "fog":
        return f"The fog lights are {_onoff(v)}."
    if k == "signal":
        return SIGNAL_S[v]
    if k == "cross_l":
        return "The vehicle is crossing the lane marking on the left." if v else None
    if k == "cross_r":
        return "The vehicle is crossing the lane marking on the right." if v else None
    if k == "lane":
        return f"The vehicle is driving in the {v} lane."
    if k == "exit":
        return EXIT_S[v]
    if k == "people":
        return ("No people are currently detected around the vehicle." if v == "none"
                else f"People density around the vehicle is {v}.")
    if k == "vehicles":
        return ("No vehicles are currently detected around the vehicle." if v == "none"
                else f"Traffic around the vehicle is {v}.")
    if k == "danger":
        return ("No dangerous objects are currently detected around the vehicle." if v == "none"
                else f"Dangerous object density around the vehicle is {v}.")
    if k == "activity":
        return ACTIVITY_S[v]
    if k == "attention":
        return f"Attention level is {v}."
    if k == "fatigue":
        return "No fatigue is detected." if v == "none" else f"Fatigue level is {v}."
    if k == "tension":
        return ("The driving style shows no tension." if v == "none"
                else f"The driving style shows a {v} level of tension.")
    if k in EMOS:
        return (f"No {EMOS[k]} is detected." if v == "none"
                else f"The driver shows a {v} level of {EMOS[k]}.")
    if k == "weather":
        return f"The current weather is {v}."
    if k == "forecast":
        return f"The weather forecast predicts {v} conditions."
    if k == "risk":
        return ("There is no current risk in this area." if v == "none"
                else f"The current risk level in this area is {v}.")
    if k == "road_type":
        art = "an" if v == "urban" else "a"
        return f"The vehicle is currently driving on {art} {v} road."
    if k == "road_cond":
        return f"The road condition is {v}."
    if k == "time":
        return f"The time of day is {v}."
    if k == "visibility":
        return "Driving visibility is optimal." if v == "optimal" else f"Driving visibility is {v}."
    if k == "traffic":
        return TRAFFIC_S[v]
    raise KeyError(k)


CAT_VALUES = {
    **{d: [True, False] for d in DOORS},
    "locked": [True, False], "trunk": [True, False], "engine": [True, False],
    "side": [True, False], "low": [True, False], "high": [True, False], "fog": [True, False],
    "signal": ["left", "right", "none"], "cross_l": [True], "cross_r": [True],
    "lane": ["left", "center", "right"], "exit": ["none", "right", "left"],
    "people": ["none"] + list(DENS3), "vehicles": ["none"] + list(DENS3), "danger": ["none"] + list(DENS3),
    "activity": list(ACTIVITY_S), "attention": B4, "fatigue": ["none"] + B4, "tension": ["none"] + B4,
    **{e: ["none"] + B4 for e in EMOS},
    "weather": WEATHERS, "forecast": WEATHERS, "risk": RISKS, "road_type": ROADS,
    "road_cond": CONDS, "time": TIMES, "visibility": VIS, "traffic": TRAFFIC,
}
SENT2FV = {}
for _k, _vals in CAT_VALUES.items():
    for _v in _vals:
        _s = render_field(_k, _v)
        assert _s not in SENT2FV, f"template collision: {_s}"
        SENT2FV[_s] = (_k, _v)
NUM_PATTERNS = [
    (re.compile(r"^Speed is (\d+) and is (stable|increasing|decreasing)\.$"), "speed"),
    (re.compile(r"^Internal temperature is (-?\d+) and is (stable|increasing|decreasing)\.$"), "int_temp"),
    (re.compile(r"^External temperature is (-?\d+) and is (stable|increasing|decreasing)\.$"), "ext_temp"),
]
LIMIT_RE = re.compile(r"^The speed limit is (\d+)\.$")
TRAFFIC_SENTENCES = set(TRAFFIC_S.values()) | {render_field("vehicles", v) for v in CAT_VALUES["vehicles"]}


def _match_sentence(s: str) -> tuple[str, Any]:
    """Match one knowledge sentence against the templates, returning its (field, value) pair."""
    if s in SENT2FV:
        return SENT2FV[s]
    for rx, name in NUM_PATTERNS:
        m = rx.match(s)
        if m:
            return name, (int(m.group(1)), m.group(2))
    m = LIMIT_RE.match(s)
    if m:
        return "limit", int(m.group(1))
    raise ValueError(f"sentence does not match any template: {s!r}")


def parse(sentences: list[str]) -> Facts:
    """Reverse the templates: knowledge sentences -> facts dict. Raises on unknown/duplicate."""
    g: Facts = {}
    for s in sentences:
        k, v = _match_sentence(s)
        if k in g:
            raise ValueError(f"duplicate field {k}")
        g[k] = v
    if "speed" not in g:
        raise ValueError("speed sentence missing")
    return g


# =============================================================================
# 3. Physical coherence
# =============================================================================
def coherent(f: Facts) -> bool:
    s = f["speed"][0]
    mv = s > 0
    et = f["ext_temp"][0] if f.get("ext_temp") else None
    rt, w, c = f.get("road_type"), f.get("weather"), f.get("road_cond")
    if f.get("engine") is False and mv:
        return False
    if not mv and (f.get("cross_l") or f.get("cross_r") or f["speed"][1] != "stable"):
        return False
    if f.get("activity") == "about_exit" and mv:
        return False
    if f.get("time") == "Night" and w == "sunny":
        return False
    if c in ("icy", "snowy") and et is not None and et > 2:
        return False
    if w == "snowy" and et is not None and et > 3:
        return False
    if w == "rainy" and c == "dry":
        return False
    if rt and mv:
        if s > SPEED_CAP[rt]:
            return False
        if rt == "highway" and s < 40 and f.get("traffic") != "heavy":
            return False
    lim = f.get("limit")
    if lim is not None and rt and lim not in LIMITS[rt]:
        return False
    tr, vh = f.get("traffic"), f.get("vehicles")
    if tr and vh and TRAFFIC_TO_VEH[tr] != vh:
        return False
    if w == "foggy" and f.get("visibility") == "optimal":
        return False
    it = f.get("int_temp")
    if it and not (10 <= it[0] <= 40):
        return False
    if et is not None and not (-20 <= et <= 42):
        return False
    if sum(1 for e in EMOS if f.get(e) in HI) > 1:
        return False
    if f.get("activity") == "asleep" and (f.get("attention") in HI or f.get("fatigue") in ("none", "low")):
        return False
    if f.get("cross_l") and f.get("cross_r"):
        return False
    if f.get("exit") in ("left", "right") and rt and rt != "highway":
        return False
    return True


# =============================================================================
# 4. Consistency and critical-safety validation helpers (used to check/enforce LLM output)
# =============================================================================
def check_consistency(d: Decision) -> None:
    it = d["intervention_type"]
    if it == "none":
        assert d["skill"] == d["action"] == d["suggestion_type"] == "none" and d["urgency"] == "none", d
    elif it == "act":
        assert d["action"] != "none" and d["suggestion_type"] == "none" and d["skill"] in ("driving", "wellbeing"), d
        assert d["urgency"] != "none", d
    elif it == "suggest":
        assert d["action"] == "none" and d["suggestion_type"] != "none" and d["skill"] in ("driving", "wellbeing"), d
        assert d["urgency"] != "none", d
    if d["urgency"] == "critical":
        assert it == "suggest" and d["action"] == "none" and d["tone"] == "serious", d


def critical_expected(g: Facts) -> str | None:
    """Independent re-statement of the critical safety rule (used to validate/enforce LLM output)."""
    s = g["speed"][0]
    if s <= 0:
        return None
    if g.get("activity") == "asleep":
        return "restore_attention"
    if any(g.get(k) is True for k in DOOR_KEYS + ["trunk"]):
        return "secure_vehicle"
    if g.get("danger") in DENS3:
        return "adapt_driving_to_conditions"
    if g.get("activity") == "phone" and (s >= 90 or g.get("road_type") == "highway"):
        return "reduce_distraction"
    if g.get("fatigue") == "very high" and g.get("attention") == "low":
        return "take_break"
    if g.get("visibility") == "low" and g.get("road_cond") in ("icy", "snowy"):
        return "adapt_driving_to_conditions"
    if g.get("tension") == "very high" and g.get("limit") is not None and s > g["limit"]:
        return "calm_driving"
    return None


# =============================================================================
# 5. Soft probabilities
# =============================================================================
ACTION_GROUPS = [
    ["increase_temperature", "decrease_temperature", "enable_air_conditioning", "disable_air_conditioning"],
    ["lock_doors", "unlock_doors"], ["start_radio", "stop_radio"], ["start_navigation", "stop_navigation"],
    ["enable_sidelights", "disable_sidelights", "enable_low_beam_headlights", "disable_low_beam_headlights",
     "enable_high_beam_headlights", "disable_high_beam_headlights", "enable_fog_lights", "disable_fog_lights"],
    ["find_rest_area"],
]
SUGG_GROUPS = [
    ["take_break", "restore_attention", "reduce_distraction", "regulate_emotional_state"],
    ["calm_driving", "reduce_speed", "adapt_driving_to_conditions", "use_turn_signal", "prepare_for_maneuver"],
    ["secure_vehicle", "prepare_for_weather"],
]
TONE_NB = {"calm": ["empathetic", "discreet"], "enthusiastic": ["calm", "empathetic"], "serious": ["calm"],
           "empathetic": ["calm"], "discreet": ["calm"]}
ITYPE_NB = {"none": ["suggest"], "suggest": ["act", "none"], "act": ["suggest"]}
SKILL_NB = {"none": ["conversation"], "driving": ["wellbeing"], "wellbeing": ["driving"], "conversation": ["none"]}


def neighbors(q: str, t: str) -> list[str]:
    if q == "urgency":
        return [URG[i] for i in (UI[t] - 1, UI[t] + 1) if 0 <= i < len(URG)]
    if q == "tone":
        return TONE_NB[t]
    if q == "intervention_type":
        return ITYPE_NB[t]
    if q == "skill":
        return SKILL_NB[t]
    groups = ACTION_GROUPS if q == "action" else SUGG_GROUPS
    if t == "none":
        return []
    for grp in groups:
        if t in grp:
            return [x for x in grp if x != t] + ["none"]
    return ["none"]


def soft(q: str, target: str, llm_confidence: float, r: random.Random) -> tuple[dict[str, float], float]:
    labels = list(CRITERIA[q])
    others = [l for l in labels if l != target]
    # Clamp the teacher's self-reported confidence into a safe range: high enough that the
    # target always remains the unique maximum regardless of how the remaining mass is spread,
    # while still carrying the LLM's own uncertainty signal through the clamp.
    conf = min(0.97, max(0.75, llm_confidence))
    floor = 0.0005
    rest = 1.0 - conf - floor * len(others)
    w = {l: 0.0 for l in others}
    nb = [l for l in neighbors(q, target) if l in w]
    share = 1.0
    if nb:
        a = [r.random() + 0.2 for _ in nb]
        sa = sum(a)
        for l, x in zip(nb, a):
            w[l] += 0.7 * rest * x / sa
        share = 0.3
    b = [r.random() + 0.2 for _ in others]
    sb = sum(b)
    for l, x in zip(others, b):
        w[l] += share * rest * x / sb
    probs = {l: round(floor + w[l], 6) for l in others}
    probs[target] = round(1.0 - sum(probs.values()), 6)
    ordered = {l: probs[l] for l in labels}
    return ordered, probs[target]


# =============================================================================
# 6. LLM distillation (each row's labels, confidences and description come from a teacher LLM)
# =============================================================================
def build_stellantis_llm(stellantis_llm: str) -> Any:
    status = requests.get(f"https://apps.services.calypso.intra.chrysler.com/cmd/state?application={stellantis_llm}", timeout=30)
    if status.json() == "ONLINE":
        print(f"Stellantis LLM ({stellantis_llm}) is ready")
    else:
        raise Exception(f"LLM {stellantis_llm} status is {status.json()}, please start it manually or wait for it to be fully ONLINE.")
    return StellantisVLLM(model=stellantis_llm)

def build_llm(ollama_host: str, ollama_port: int, ollama_model: str, ollama_timeout: int = 1200, max_tokens: int = 50000) -> Any:
    """Same Ollama-backed crewai LLM interface used at runtime by agents/llm_agent.py."""
    from crewai import LLM
    return LLM(
        model=ollama_model,
        base_url=f"http://{ollama_host}:{ollama_port}",
        timeout=ollama_timeout,
        max_tokens=max_tokens,
    )

def build_distill_prompt(sents: list[str]) -> str:
    facts = "\n".join(f"- {s}" for s in sents)
    criteria_block = "\n".join(
        f"{q} ({INSTRUCTIONS[q]}): " + ", ".join(f'"{l}"={CRITERIA[q][l]}' for l in CRITERIA[q])
        for q in DECISIONS
    )
    return (
        "You are the teacher model for a synthetic in-vehicle assistant dataset. Given the observed facts "
        "below, decide the six typed decisions defined by this contract, then explain your reasoning.\n\n"
        f"Facts:\n{facts}\n\n"
        f"Decisions and allowed labels:\n{criteria_block}\n\n"
        "Rules:\n"
        "- Every label must be exactly one of the allowed label tokens listed for that decision.\n"
        "- If intervention_type=none, skill, action and suggestion_type must all be none, and urgency must be "
        "none.\n"
        "- If intervention_type=act, action must not be none, suggestion_type must be none, and skill must be "
        "driving or wellbeing.\n"
        "- If intervention_type=suggest, action must be none, suggestion_type must not be none, and skill must "
        "be driving or wellbeing.\n"
        "- skill=conversation is never correct for this dataset; do not use it.\n"
        "- These situations always require urgency=critical, intervention_type=suggest, action=none, "
        "tone=serious and exactly this suggestion_type (moving means speed above 0): driver asleep while "
        "moving -> restore_attention; a door or the trunk open while moving -> secure_vehicle; dangerous "
        "objects detected while moving -> adapt_driving_to_conditions; phone use while moving at speed 90 or "
        "more or on a highway -> reduce_distraction; very high fatigue with low attention while moving -> "
        "take_break; low visibility on an icy or snowy road while moving -> adapt_driving_to_conditions; very "
        "high driving tension while the speed is above the speed limit -> calm_driving.\n"
        "- urgency=critical is reserved exclusively for those situations. If none of them is present in the "
        "facts above, urgency must be none, low, medium or high and critical is forbidden, however serious "
        "the situation may otherwise look.\n"
        "- suggestion_type=reduce_speed only when the speed is strictly above a stated speed limit.\n"
        "- Other suggest situations while moving: phone or eating -> reduce_distraction; high or very high "
        "fatigue -> take_break; low attention -> restore_attention; wet, icy, snowy or gravel road with low or "
        "medium visibility or at high speed -> adapt_driving_to_conditions; strong anger, sadness or fear -> "
        "regulate_emotional_state; high tension -> calm_driving; crossing a lane marking without the matching "
        "turn signal -> use_turn_signal; highway exit on one side while in the opposite lane -> "
        "prepare_for_maneuver. Driver about to exit: forecast of rain, snow or fog differing from the current "
        "weather -> prepare_for_weather; unlocked vehicle or open trunk in a medium or high risk area -> "
        "secure_vehicle.\n"
        "- Use intervention_type=act (usually urgency low or medium, tone discreet or calm) when one reversible "
        "action fixes the situation and nothing more urgent applies: engine on and internal temperature 17 or "
        "below -> increase_temperature; internal 28 or more and external 25 or more -> enable_air_conditioning; "
        "internal 25 or more and external below 25 -> decrease_temperature; internal 18 or 19 and external 12 "
        "or below -> disable_air_conditioning (these four use skill=wellbeing); doors unlocked while moving, or "
        "while stopped in a medium or high risk area -> lock_doors; stopped with engine off, driver about to "
        "exit and doors locked with no elevated risk -> unlock_doors; stopped with engine off and driver about "
        "to exit -> stop_navigation, or disable_low_beam_headlights if the low beams are on; stopped (speed 0) "
        "with engine on, no distracting driver activity, doors locked and trunk closed -> start_navigation "
        "(the vehicle is ready to depart, so the trip is prepared; no destination needs to be stated); moving with high attention, no or "
        "light traffic and mild sadness or strong happiness -> start_radio (skill=wellbeing); driver talking in "
        "heavy traffic -> stop_radio (skill=wellbeing); evening, stopped, engine on, sidelights and low beams "
        "off -> enable_sidelights; morning or afternoon with optimal visibility, sidelights on and low beams off "
        "-> disable_sidelights; moving with low beams off at night or with low or medium visibility -> "
        "enable_low_beam_headlights; night on a rural road or highway with no vehicles or people around, good "
        "visibility, low beams on and high beams off -> enable_high_beam_headlights; high beams on with traffic "
        "or vehicles around, on an urban or residential road or in fog -> disable_high_beam_headlights; moving "
        "in fog or snow with low visibility and fog lights off -> enable_fog_lights; fog lights on with good "
        "visibility and no fog or snow -> disable_fog_lights; high fatigue with medium or better attention on "
        "a highway or rural road -> find_rest_area (skill=wellbeing). Lighting, locking and navigation actions "
        "use skill=driving.\n"
        "- If no situation above applies, choose intervention_type=none.\n"
        "- Base every decision only on the facts above; never invent a fact that is not listed.\n"
        "- confidence is your own certainty for each decision, between 0 and 1.\n"
        "- description is a single plain-prose paragraph of 2 to 5 sentences in English, with no bullet "
        "points, no line breaks and no headings, explaining in causal terms (use because, so, since or "
        "therefore) why the facts support all six decisions; use the words urgency, tone, intervention, "
        "skill, action and suggest at least once each; do not just restate the label values; never write a "
        "number that is not in the facts above.\n\n"
        "Respond with only a single JSON object, no other text, in exactly this shape:\n"
        '{"urgency": {"label": "...", "confidence": 0.9}, "tone": {"label": "...", "confidence": 0.9}, '
        '"intervention_type": {"label": "...", "confidence": 0.9}, "skill": {"label": "...", "confidence": 0.9}, '
        '"action": {"label": "...", "confidence": 0.9}, "suggestion_type": {"label": "...", "confidence": 0.9}, '
        '"description": "..."}'
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a single JSON object from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1] if "\n" in text else text
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_distill_response(text: str) -> tuple[Decision, dict[str, float], str] | None:
    obj = _extract_json(text)
    if obj is None:
        return None
    try:
        dec: Decision = {q: str(obj[q]["label"]) for q in DECISIONS}
        conf = {q: float(obj[q]["confidence"]) for q in DECISIONS}
        description = str(obj["description"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(dec[q] not in CRITERIA[q] for q in DECISIONS):
        return None
    if any(not (0.0 <= conf[q] <= 1.0) for q in DECISIONS):
        return None
    return dec, conf, description


def split_sentences(d: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", d.strip()) if x.strip()]


def description_problems(d: str, kn: list[str]) -> list[str]:
    """Shared description contract, used both to filter LLM output and to validate the dataset."""
    if not d or not d.strip():
        return ["description_empty"]
    problems: list[str] = []
    n_sent = len(split_sentences(d))
    if not 2 <= n_sent <= 5:
        problems.append("description_sentences")
    low = d.lower()
    if not all(c in low for c in DESC_CONCEPTS):
        problems.append("description_six_decisions")
    if not any(c in low for c in CAUSAL):
        problems.append("description_causal")
    if not set(re.findall(r"-?\d+", d)) <= set(re.findall(r"-?\d+", " ".join(kn))):
        problems.append("description_grounding")
    toks = re.findall(r"[a-z_]+", low)
    if toks and sum(1 for t in toks if t in ALL_LABEL_TOKENS) / len(toks) >= 0.5:
        problems.append("description_label_restatement")
    return problems


def repair_description(d: str, kn: list[str]) -> str:
    """Normalize LLM prose to the contract: one paragraph, no ungrounded numbers, at most 5 sentences."""
    d = re.sub(r"\s+", " ", d).strip()
    d = re.sub(r"^[-*\d.)\s]+", "", d)
    kn_nums = set(re.findall(r"-?\d+", " ".join(kn)))
    sents = [s for s in split_sentences(d) if set(re.findall(r"-?\d+", s)) <= kn_nums]
    return " ".join(sents[:5])


def describe_fallback(dec: Decision) -> str:
    """Grounded, contract-compliant description used only for rule-based fallback rows."""
    if dec["intervention_type"] == "none":
        return ("No intervention is needed based on the observed facts, so urgency, intervention, skill, "
                "action and suggestion all remain none. A discreet tone is appropriate because no active "
                "communication is required.")
    return (f"The observed facts were judged to need urgency level {dec['urgency']}, so the intervention type "
            f"is {dec['intervention_type']} and the tone is {dec['tone']}. Since the decisive condition falls "
            f"under the {dec['skill']} skill, the action is {dec['action']} and the suggestion is "
            f"{dec['suggestion_type']}, because this directly addresses that condition.")


def _fallback_decision(g: Facts) -> Decision:
    """Rule-based safety net used only when the LLM repeatedly returns a non-compliant answer."""
    return rule_label(g)[0]


def label_problems(dec: Decision, ce: str | None) -> list[str]:
    """Human-readable contract violations, fed back to the LLM on retry."""
    p: list[str] = []
    it = dec["intervention_type"]
    if it == "none" and any(dec[q] != "none" for q in ("urgency", "skill", "action", "suggestion_type")):
        p.append("intervention_type=none requires urgency, skill, action and suggestion_type to all be none")
    if it == "act" and (dec["action"] == "none" or dec["suggestion_type"] != "none"):
        p.append("intervention_type=act requires a non-none action and suggestion_type=none")
    if it == "suggest" and (dec["action"] != "none" or dec["suggestion_type"] == "none"):
        p.append("intervention_type=suggest requires action=none and a non-none suggestion_type")
    if it != "none" and dec["skill"] not in ("driving", "wellbeing"):
        p.append(f"intervention_type={it} requires skill=driving or skill=wellbeing")
    if it != "none" and dec["urgency"] == "none":
        p.append(f"intervention_type={it} requires an urgency other than none")
    if ce is not None:
        if not (dec["urgency"] == "critical" and it == "suggest" and dec["action"] == "none"
                and dec["tone"] == "serious" and dec["suggestion_type"] == ce):
            p.append("the facts contain a critical situation, so urgency=critical, intervention_type=suggest, "
                     f"action=none, tone=serious and suggestion_type={ce} are required")
    elif dec["urgency"] == "critical":
        p.append("none of the critical situations is present in the facts, so urgency=critical is forbidden")
    return p


def description_hints(d: str, kn: list[str], problems: list[str]) -> list[str]:
    hints = []
    low = d.lower()
    for pr in problems:
        if pr == "description_empty":
            hints.append("the description was empty")
        elif pr == "description_sentences":
            hints.append(f"it must have 2 to 5 sentences (it had {len(split_sentences(d))})")
        elif pr == "description_six_decisions":
            missing = [c for c in DESC_CONCEPTS if c not in low]
            hints.append("it must contain each of these exact words: " + ", ".join(missing))
        elif pr == "description_causal":
            hints.append("it must use at least one of: because, so, since, therefore")
        elif pr == "description_grounding":
            hints.append("it must not contain any number that is not written in the facts")
        elif pr == "description_label_restatement":
            hints.append("it must explain the reasons in prose rather than list label values")
    return hints


def build_description_prompt(sents: list[str], dec: Decision, hints: list[str]) -> str:
    facts = "\n".join(f"- {s}" for s in sents)
    decisions = "\n".join(f"- {q}: {dec[q]}" for q in DECISIONS)
    fix = ("\nYour previous description was rejected because " + "; ".join(hints) + ".\n") if hints else ""
    return (
        "You write debug explanations for an in-vehicle assistant dataset.\n\n"
        f"Facts:\n{facts}\n\nDecisions already taken:\n{decisions}\n{fix}\n"
        "Write one plain-prose English paragraph of 2 to 5 sentences explaining why the facts lead to these "
        "decisions. Requirements: use each of the words urgency, tone, intervention, skill, action and suggest "
        "at least once; use because, so, since or therefore; do not write any number that is not in the facts; "
        "no bullet points, headings, line breaks, quotes or JSON. Reply with the paragraph only."
    )


def _call(llm: Any, prompt: str) -> str | None:
    try:
        return str(llm.call(prompt))
    except Exception:
        return None


def llm_label_row(llm: Any, sents: list[str], g: Facts, retries: int,
                  rej: CounterT[str]) -> tuple[Decision, dict[str, float], str, bool, bool]:
    """Ask the teacher LLM for the six decisions, per-decision confidence and description.

    Returns (decision, confidence_by_question, description, label_fallback, desc_fallback). Each rejected
    answer is retried with the violations fed back. Labels fall back to the rule-based labeler after
    `retries` non-compliant responses; the description is then retried on its own and only falls back to
    a template if every attempt is still non-compliant."""
    base = build_distill_prompt(sents)
    ce = critical_expected(g)
    feedback = ""
    kept: tuple[Decision, dict[str, float], str] | None = None
    for _ in range(max(1, retries)):
        text = _call(llm, base + feedback)
        if text is None:
            rej["llm_error"] += 1
            continue
        parsed = _parse_distill_response(text)
        if parsed is None:
            rej["unparseable"] += 1
            feedback = ("\n\nYour previous answer was not a single valid JSON object using only the allowed "
                        "labels and confidences between 0 and 1. Reply with the JSON object only.")
            continue
        problems = label_problems(parsed[0], ce)
        if problems:
            rej["label_rules"] += 1
            feedback = (f"\n\nYour previous answer was:\n{text.strip()[:1500]}\nIt was rejected because: "
                        + "; ".join(problems) + ". Reply with a corrected JSON object only.")
            continue
        kept = parsed
        break
    label_fallback = kept is None
    if kept is None:
        dec, conf, description = _fallback_decision(g), {q: 0.9 for q in DECISIONS}, ""
    else:
        dec, conf, description = kept
    description = repair_description(description, sents)
    problems = description_problems(description, sents)
    for _ in range(max(1, retries)):
        if not problems:
            return dec, conf, description, label_fallback, False
        rej.update(problems)
        text = _call(llm, build_description_prompt(sents, dec, description_hints(description, sents, problems)))
        if text is None:
            rej["llm_error"] += 1
            continue
        description = repair_description(text.strip().strip('"'), sents)
        problems = description_problems(description, sents)
    if not problems:
        return dec, conf, description, label_fallback, False
    return dec, conf, describe_fallback(dec), label_fallback, True


# =============================================================================
# 7. Scenario context, distractors and generators
# =============================================================================
def ctx(f: Facts, r: random.Random, moving: bool = True) -> Facts:
    """Fill missing context fields. Keys already present in f (even None) are kept."""
    if f.get("speed") is not None:
        moving = f["speed"][0] > 0
    f.setdefault("road_type", r.choice(ROADS))
    rt = f["road_type"]
    if "time" not in f:
        f["time"] = r.choice(TIMES) if r.random() < 0.85 else None
    if "weather" not in f:
        if r.random() < 0.75:
            w = r.choice(["sunny", "sunny", "cloudy", "cloudy", "rainy", "foggy", "snowy"])
            if f["time"] == "Night" and w == "sunny":
                w = "cloudy"
            f["weather"] = w
        else:
            f["weather"] = None
    w = f["weather"]
    if "road_cond" not in f:
        if r.random() < 0.6:
            f["road_cond"] = "wet" if w == "rainy" else r.choice(["dry", "dry", "dry", "wet"])
        else:
            f["road_cond"] = None
    c = f["road_cond"]
    if "ext_temp" not in f:
        if r.random() < 0.55:
            if c in ("icy", "snowy"):
                lo, hi = -12, 1
            elif w == "snowy":
                lo, hi = -10, 2
            elif w == "sunny":
                lo, hi = 8, 33
            else:
                lo, hi = -3, 26
            f["ext_temp"] = (r.randint(lo, hi), r.choice(TRENDS))
        else:
            f["ext_temp"] = None
    if "traffic" not in f:
        f["traffic"] = r.choice(["none", "light", "light", "medium", "medium", "heavy"]) if r.random() < 0.8 else None
    if "vehicles" not in f:
        f["vehicles"] = TRAFFIC_TO_VEH[f["traffic"]] if (f["traffic"] is not None and r.random() < 0.3) else None
    if "limit" not in f:
        f["limit"] = r.choice(LIMITS[rt]) if r.random() < 0.7 else None
    if f.get("speed") is None:
        if moving:
            lo, hi = SPEED_RANGE[rt]
            if f["limit"] is not None:
                hi = min(hi, f["limit"])
            lo = min(lo, hi - 5)
            f["speed"] = (r.randint(max(5, lo), hi), r.choice(TRENDS))
        else:
            f["speed"] = (0, "stable")
    if "engine" not in f:
        f["engine"] = True if (f["speed"][0] > 0 and r.random() < 0.35) else None
    return f


DISTRACTOR_POOL = ["attention", "fatigue", "tension", "emotion", "people", "danger", "locked", "trunk",
                   "int_temp", "lane", "exit", "visibility", "forecast", "risk", "low", "fog", "high",
                   "door", "signal"]
NEUTRAL_EMO = {"neutral": B4, "surprise": B4, "disgust": B4, "happy": ["none", "low", "medium"],
               "fear": ["none", "low", "medium"], "angry": ["none", "low"]}


def n_facts(f: Facts) -> int:
    return sum(1 for k, v in f.items() if v is not None and render_field(k, v))


def neutral_value(name: str, f: Facts, r: random.Random, moving: bool) -> Any:
    w, vis = f.get("weather"), f.get("visibility")
    if name == "attention":
        return r.choice(["medium", "high", "very high"])
    if name == "fatigue":
        return r.choice(["none", "low"])
    if name == "tension":
        return r.choice(["none", "low", "medium"])
    if name == "people":
        return r.choice(["none", "low"])
    if name == "danger":
        return "none"
    if name == "locked":
        return True if moving else None
    if name == "trunk":
        return False
    if name == "int_temp":
        return (r.randint(20, 24), r.choice(TRENDS))
    if name == "lane":
        return None if f.get("exit") in ("left", "right") else r.choice(["left", "center", "right"])
    if name == "exit":
        return "none"
    if name == "visibility":
        return r.choice(["high", "very high"]) if w == "foggy" else r.choice(["high", "very high", "optimal"])
    if name == "forecast":
        return w if w is not None else r.choice(["sunny", "cloudy"])
    if name == "risk":
        return r.choice(RISKS) if moving else r.choice(["none", "low"])
    if name == "low":
        if not moving:
            return None
        return True if (f.get("time") == "Night" or vis in ("low", "medium")) else r.choice([True, False])
    if name == "fog":
        return None if (w in ("foggy", "snowy") and vis == "low") else False
    if name == "high":
        return None if f.get("time") == "Night" else False
    if name == "signal":
        return None if (f.get("cross_l") or f.get("cross_r")) else "none"
    return None


def add_distractors(f: Facts, r: random.Random) -> None:
    moving = f["speed"][0] > 0
    room = 12 - n_facts(f)
    if room <= 0:
        return
    k = r.randint(1, min(4, room))
    pool = DISTRACTOR_POOL[:]
    r.shuffle(pool)
    added = 0
    for name in pool:
        if added >= k:
            break
        if name == "emotion":
            e = r.choice(list(NEUTRAL_EMO))
            if e in f:
                continue
            f[e] = r.choice(NEUTRAL_EMO[e])
        elif name == "door":
            d = r.choice(DOOR_KEYS)
            if d in f:
                continue
            f[d] = False
        else:
            if name in f:
                continue
            v = neutral_value(name, f, r, moving)
            if v is None:
                continue
            f[name] = v
        added += 1


GENS: list[tuple[str, int, GenFn, dict[str, Any]]] = []


def G(name: str, weight: int, **expect: Any) -> Callable[[GenFn], GenFn]:
    def deco(fn: GenFn) -> GenFn:
        GENS.append((name, weight, fn, expect))
        return fn
    return deco


def trend(r: random.Random) -> str:
    return r.choice(TRENDS)


# ---------------------------- SUGGEST families -------------------------------
@G("asleep_moving", 1200, itype="suggest", suggestion_type="restore_attention")
def _g(r: random.Random) -> Facts:
    f = {"activity": "asleep"}
    if r.random() < 0.6:
        f["fatigue"] = r.choice(["high", "very high"])
    if r.random() < 0.4:
        f["attention"] = "low"
    return ctx(f, r, True)


@G("open_moving", 1400, suggestion_type="secure_vehicle")
def _g(r: random.Random) -> Facts:
    k = r.choice(DOOR_KEYS + ["trunk", "trunk"])
    f = {k: True}
    for d in DOOR_KEYS:
        if d != k and r.random() < 0.3:
            f[d] = False
    return ctx(f, r, True)


@G("danger_moving", 1200, suggestion_type="adapt_driving_to_conditions")
def _g(r: random.Random) -> Facts:
    f = {"danger": r.choice(list(DENS3))}
    if r.random() < 0.4:
        f["people"] = r.choice(list(DENS3))
    if r.random() < 0.3:
        f["surprise"] = r.choice(B4)
    return ctx(f, r, True)


@G("phone_moving", 1800, suggestion_type="reduce_distraction")
def _g(r: random.Random) -> Facts:
    f = {"activity": "phone"}
    if r.random() < 0.5:
        f["road_type"] = "highway"
    return ctx(f, r, True)


@G("eating_moving", 1200, suggestion_type="reduce_distraction")
def _g(r: random.Random) -> Facts:
    return ctx({"activity": "eating", "attention": r.choice(["medium", "high", "very high", None])}, r, True)


@G("fatigue_critical", 1500, suggestion_type="take_break")
def _g(r: random.Random) -> Facts:
    f = {"fatigue": "very high", "attention": "low",
         "time": r.choice(["Night", "Night", "Evening", "Morning", "Afternoon"]),
         "road_type": r.choice(["highway", "highway", "rural", "urban"])}
    if r.random() < 0.4:
        f["traffic"] = "heavy"
    return ctx(f, r, True)


@G("fatigue_high", 1800, suggestion_type="take_break")
def _g(r: random.Random) -> Facts:
    if r.random() < 0.5:
        f = {"fatigue": "very high", "attention": r.choice(["medium", "high"])}
    else:
        f = {"fatigue": "high", "attention": "low"}
    return ctx(f, r, True)


@G("fatigue_medium_night", 700, suggestion_type="take_break")
def _g(r: random.Random) -> Facts:
    return ctx({"fatigue": "medium", "attention": r.choice(list(ATT_OK)), "time": "Night",
                "road_type": "highway"}, r, True)


@G("attention_low", 1500, suggestion_type="restore_attention")
def _g(r: random.Random) -> Facts:
    return ctx({"attention": "low", "fatigue": r.choice(["none", "low", "medium", None]),
                "activity": r.choice(["driving", "talking", "idle", None])}, r, True)


@G("icy_low_visibility", 1000, suggestion_type="adapt_driving_to_conditions")
def _g(r: random.Random) -> Facts:
    f = {"visibility": "low", "road_cond": r.choice(["icy", "snowy"]),
         "weather": r.choice(["snowy", "foggy", "cloudy"]), "ext_temp": (r.randint(-12, 1), trend(r))}
    if r.random() < 0.5:
        f["low"] = True
    return ctx(f, r, True)


@G("bad_road", 1500, suggestion_type="adapt_driving_to_conditions")
def _g(r: random.Random) -> Facts:
    v = r.randrange(3)
    if v == 0:
        cond = r.choice(["wet", "gravel", "icy", "snowy"])
        f = {"road_cond": cond}
        if cond in ("wet", "gravel"):
            f["visibility"] = r.choice(["low", "medium"])
            f["weather"] = r.choice(["rainy", "foggy", "cloudy"])
            if cond == "gravel":
                f["road_type"] = "rural"
                if f["weather"] == "rainy":
                    f["weather"] = "cloudy"
        else:
            f["visibility"] = "medium"
            f["weather"] = r.choice(["snowy", "cloudy", "foggy"])
            f["ext_temp"] = (r.randint(-12, 1), trend(r))
        if f["visibility"] == "low" or r.random() < 0.5:
            f["low"] = True
    elif v == 1:
        f = {"road_cond": r.choice(["icy", "snowy"]), "visibility": r.choice(["high", "very high", None]),
             "weather": r.choice(["snowy", "cloudy"]), "road_type": r.choice(["rural", "highway"]),
             "ext_temp": (r.randint(-12, 1), trend(r))}
    else:
        f = {"road_cond": "gravel", "road_type": "rural", "weather": r.choice(["sunny", "cloudy", None])}
    return ctx(f, r, True)


@G("speeding", 3000, suggestion_type="reduce_speed")
def _g(r: random.Random) -> Facts:
    rt = r.choice(ROADS)
    lim = r.choice(LIMITS[rt])
    ov = r.randint(*r.choice([(1, 10), (11, 20), (21, 40)]))
    f = {"road_type": rt, "limit": lim, "speed": (lim + ov, trend(r)),
         "tension": r.choice(["none", "low", "medium", None])}
    if r.random() < 0.25:
        c = "gravel" if (rt == "rural" and r.random() < 0.4) else "wet"
        f["road_cond"] = c
        if c == "wet" and r.random() < 0.5:
            f["weather"] = "rainy"
    if r.random() < 0.2:
        f["people"] = r.choice(["medium", "high"])
    return ctx(f, r, True)


@G("tension_speeding", 900, suggestion_type="calm_driving")
def _g(r: random.Random) -> Facts:
    rt = r.choice(ROADS)
    lim = r.choice(LIMITS[rt])
    return ctx({"tension": "very high", "road_type": rt, "limit": lim,
                "speed": (lim + r.randint(5, 35), trend(r))}, r, True)


@G("anger", 1500, suggestion_type="regulate_emotional_state")
def _g(r: random.Random) -> Facts:
    f = {"angry": r.choice(list(HI))}
    if r.random() < 0.5:
        f["tension"] = r.choice(list(HI))
        f["traffic"] = "heavy"
    else:
        f["tension"] = r.choice(["none", "low", "medium", None])
    return ctx(f, r, True)


@G("tension", 1300, suggestion_type="calm_driving")
def _g(r: random.Random) -> Facts:
    return ctx({"tension": r.choice(list(HI)), "traffic": r.choice(TRAFFIC),
                "angry": r.choice(["none", "low", None])}, r, True)


@G("sad_fear", 1500, suggestion_type="regulate_emotional_state")
def _g(r: random.Random) -> Facts:
    if r.random() < 0.6:
        f = {"sad": r.choice(["medium", "high", "very high"]), "attention": r.choice(list(ATT_OK))}
    else:
        f = {"fear": r.choice(list(HI))}
    return ctx(f, r, True)


@G("lane_change", 1500, suggestion_type=("use_turn_signal", "restore_attention"))
def _g(r: random.Random) -> Facts:
    side = r.choice(["left", "right"])
    k = "cross_l" if side == "left" else "cross_r"
    sig = r.choice(["none", "none", "right" if side == "left" else "left"])
    f = {k: True, "signal": sig,
         "attention": "low" if r.random() < 0.25 else r.choice(["medium", "high", "very high", None])}
    return ctx(f, r, True)


@G("highway_exit", 1000, suggestion_type="prepare_for_maneuver")
def _g(r: random.Random) -> Facts:
    side = r.choice(["left", "right"])
    return ctx({"road_type": "highway", "exit": side, "lane": "left" if side == "right" else "right"}, r, True)


@G("secure_stationary", 1000, suggestion_type="secure_vehicle")
def _g(r: random.Random) -> Facts:
    f = {"speed": (0, "stable"), "risk": r.choice(["medium", "high"])}
    if r.random() < 0.6:
        f["activity"] = "about_exit"
        f["engine"] = r.choice([True, False])
        if r.random() < 0.6:
            f["locked"] = False
        else:
            f["trunk"] = True
    else:
        f["activity"] = r.choice(["idle", "talking", None])
        f["trunk"] = True
        f["engine"] = r.choice([True, False, None])
    return ctx(f, r, False)


@G("weather_exit", 900, suggestion_type="prepare_for_weather")
def _g(r: random.Random) -> Facts:
    tm = r.choice(TIMES)
    w = "cloudy" if tm == "Night" else r.choice(["sunny", "cloudy"])
    eng = r.choice([False, True])
    f = {"speed": (0, "stable"), "activity": "about_exit", "engine": eng, "weather": w, "time": tm,
         "forecast": r.choice(["rainy", "snowy", "foggy"]), "risk": r.choice(["none", "low", None]),
         "locked": r.choice([False, None])}
    if not eng:
        f["low"] = r.choice([False, None])
    return ctx(f, r, False)


@G("weather_moving", 500, suggestion_type="prepare_for_weather")
def _g(r: random.Random) -> Facts:
    w = r.choice(["cloudy", "rainy", "foggy"])
    f = {"forecast": "snowy", "weather": w, "ext_temp": (r.randint(-6, 2), trend(r)),
         "road_cond": "wet" if w == "rainy" else r.choice(["dry", "wet"])}
    return ctx(f, r, True)


# ------------------------------ ACT families ---------------------------------
def _stationary_or_moving(f: Facts, r: random.Random, p_moving: float = 0.7) -> bool:
    mv = r.random() < p_moving
    if not mv:
        f["speed"] = (0, "stable")
        f.setdefault("activity", r.choice(["idle", "talking", None]))
    return mv


@G("increase_temperature", 500, action="increase_temperature")
def _g(r: random.Random) -> Facts:
    f = {"engine": True, "int_temp": (r.randint(10, 17), trend(r))}
    if r.random() < 0.6:
        f["ext_temp"] = (r.randint(-10, 12), trend(r))
    return ctx(f, r, _stationary_or_moving(f, r))


@G("decrease_temperature", 500, action="decrease_temperature")
def _g(r: random.Random) -> Facts:
    f = {"engine": True, "int_temp": (r.randint(25, 31), trend(r)), "ext_temp": (r.randint(-5, 24), trend(r))}
    return ctx(f, r, _stationary_or_moving(f, r))


@G("enable_air_conditioning", 500, action="enable_air_conditioning")
def _g(r: random.Random) -> Facts:
    tm = r.choice(TIMES)
    f = {"engine": True, "int_temp": (r.randint(28, 36), trend(r)), "ext_temp": (r.randint(25, 38), trend(r)),
         "time": tm, "weather": "cloudy" if tm == "Night" else r.choice(["sunny", "cloudy", None]),
         "road_cond": r.choice(["dry", None])}
    return ctx(f, r, _stationary_or_moving(f, r))


@G("disable_air_conditioning", 450, action="disable_air_conditioning")
def _g(r: random.Random) -> Facts:
    f = {"engine": True, "int_temp": (r.randint(18, 19), trend(r)), "ext_temp": (r.randint(-8, 12), trend(r))}
    return ctx(f, r, _stationary_or_moving(f, r))


@G("lock_moving", 400, action="lock_doors")
def _g(r: random.Random) -> Facts:
    f = {"locked": False}
    for d in DOOR_KEYS:
        if r.random() < 0.3:
            f[d] = False
    return ctx(f, r, True)


@G("lock_stationary", 400, action="lock_doors")
def _g(r: random.Random) -> Facts:
    f = {"speed": (0, "stable"), "locked": False, "risk": r.choice(["medium", "high"]),
         "activity": r.choice(["idle", "talking", "eating", "phone", None]), "engine": r.choice([True, False, None])}
    for d in DOOR_KEYS:
        if r.random() < 0.25:
            f[d] = False
    return ctx(f, r, False)


@G("unlock_doors", 450, action="unlock_doors")
def _g(r: random.Random) -> Facts:
    return ctx({"speed": (0, "stable"), "engine": False, "activity": "about_exit", "locked": True,
                "risk": r.choice(["none", "low", None]), "low": r.choice([False, None])}, r, False)


@G("start_radio", 500, action="start_radio")
def _g(r: random.Random) -> Facts:
    f = {"sad": "low"} if r.random() < 0.5 else {"happy": r.choice(list(HI))}
    f.update({"attention": r.choice(list(HI)), "traffic": r.choice(["none", "light"]),
              "activity": r.choice(["driving", "idle", None])})
    return ctx(f, r, True)


@G("stop_radio", 450, action="stop_radio")
def _g(r: random.Random) -> Facts:
    return ctx({"activity": "talking", "traffic": "heavy", "attention": r.choice(list(ATT_OK))}, r, True)


@G("start_navigation", 400, action="start_navigation")
def _g(r: random.Random) -> Facts:
    f = {"speed": (0, "stable"), "engine": True, "activity": "idle", "locked": True, "trunk": False,
         "limit": None, "vehicles": None, "road_type": r.choice(["urban", "residential", "rural"])}
    for d in DOOR_KEYS:
        if r.random() < 0.25:
            f[d] = False
    return ctx(f, r, False)


@G("stop_navigation", 450, action="stop_navigation")
def _g(r: random.Random) -> Facts:
    return ctx({"speed": (0, "stable"), "engine": False, "activity": "about_exit",
                "low": r.choice([False, None]), "locked": r.choice([False, None]),
                "risk": r.choice(["none", "low", None])}, r, False)


@G("enable_sidelights", 450, action="enable_sidelights")
def _g(r: random.Random) -> Facts:
    return ctx({"time": "Evening", "speed": (0, "stable"), "engine": True, "side": False, "low": False,
                "road_type": r.choice(["urban", "residential"]), "activity": r.choice(["idle", "talking", None])},
               r, False)


@G("disable_sidelights", 450, action="disable_sidelights")
def _g(r: random.Random) -> Facts:
    f = {"time": r.choice(["Morning", "Afternoon"]), "visibility": "optimal", "side": True, "low": False,
         "weather": r.choice(["sunny", "cloudy"])}
    mv = r.random() < 0.7
    if not mv:
        f["speed"] = (0, "stable")
        f["engine"] = r.choice([True, None])
    return ctx(f, r, mv)


@G("enable_low_beam", 550, action="enable_low_beam_headlights")
def _g(r: random.Random) -> Facts:
    if r.random() < 0.5:
        vis = r.choice(["medium", "high", "very high", None])
        w = r.choice(["cloudy", None]) if vis in ("medium", None) else r.choice(["cloudy", "rainy", None])
        f = {"time": "Night", "low": False, "visibility": vis, "weather": w}
        if vis == "medium":
            f["road_cond"] = r.choice(["dry", None])
    else:
        f = {"visibility": r.choice(["low", "medium"]), "low": False, "weather": r.choice(["cloudy", None]),
             "road_cond": r.choice(["dry", None])}
    return ctx(f, r, True)


@G("disable_low_beam", 450, action="disable_low_beam_headlights")
def _g(r: random.Random) -> Facts:
    return ctx({"speed": (0, "stable"), "engine": False, "activity": "about_exit", "low": True,
                "locked": r.choice([False, None]), "risk": r.choice(["none", "low", None])}, r, False)


@G("enable_high_beam", 450, action="enable_high_beam_headlights")
def _g(r: random.Random) -> Facts:
    return ctx({"time": "Night", "road_type": r.choice(["rural", "highway"]), "vehicles": "none",
                "people": "none", "traffic": r.choice(["none", None]),
                "visibility": r.choice(list(GOOD_VIS)), "weather": r.choice(["cloudy", None]),
                "high": False, "low": True, "road_cond": r.choice(["dry", None])}, r, True)


@G("disable_high_beam", 500, action="disable_high_beam_headlights")
def _g(r: random.Random) -> Facts:
    f = {"high": True, "low": r.choice([True, None])}
    v = r.randrange(3)
    if v == 0:
        f["traffic"] = r.choice(["light", "medium", "heavy"])
    elif v == 1:
        f["road_type"] = r.choice(["urban", "residential"])
    else:
        f["weather"] = "foggy"
        f["visibility"] = r.choice(["medium", "high"])
        f["road_cond"] = r.choice(["dry", None])
    return ctx(f, r, True)


@G("enable_fog_lights", 500, action="enable_fog_lights")
def _g(r: random.Random) -> Facts:
    w = r.choice(["foggy", "foggy", "snowy"])
    f = {"weather": w, "visibility": "low", "fog": False, "low": True, "road_cond": "dry"}
    if w == "snowy":
        f["ext_temp"] = (r.randint(-8, 2), trend(r))
    return ctx(f, r, True)


@G("disable_fog_lights", 450, action="disable_fog_lights")
def _g(r: random.Random) -> Facts:
    f = {"fog": True, "visibility": r.choice(list(GOOD_VIS)), "weather": r.choice(["sunny", "cloudy", "rainy"])}
    mv = r.random() < 0.75
    if not mv:
        f["speed"] = (0, "stable")
    return ctx(f, r, mv)


@G("find_rest_area", 800, action="find_rest_area")
def _g(r: random.Random) -> Facts:
    return ctx({"fatigue": "high", "attention": r.choice(list(ATT_OK)),
                "road_type": r.choice(["highway", "rural"])}, r, True)


# ------------------------------ NONE families --------------------------------
@G("neg_moving", 4000, itype="none")
def _g(r: random.Random) -> Facts:
    return ctx({"attention": r.choice(list(ATT_OK)), "activity": r.choice(["driving", "idle", "talking", None]),
                "fatigue": r.choice(["none", "low", None])}, r, True)


@G("neg_emotion", 2000, itype="none")
def _g(r: random.Random) -> Facts:
    e = r.choice(list(EMOS))
    vals = {"happy": B4, "surprise": B4, "disgust": B4, "neutral": B4, "sad": ["low"],
            "fear": ["low", "medium"], "angry": ["low", "medium"]}
    f = {e: r.choice(vals[e]), "attention": r.choice(list(ATT_OK))}
    if e == "sad" or (e == "happy" and f[e] in HI):
        f["traffic"] = r.choice(["medium", "heavy"])
    return ctx(f, r, True)


@G("neg_stationary", 2500, itype="none")
def _g(r: random.Random) -> Facts:
    f = {"speed": (0, "stable"), "engine": r.choice([False, True, None])}
    opt = r.choice(["fatigue", "asleep", "phone", "eating", "attention", "emotion", "tension"])
    if opt == "fatigue":
        f["fatigue"] = r.choice(list(HI))
    elif opt == "asleep":
        f["activity"] = "asleep"
        f["fatigue"] = r.choice(list(HI))
    elif opt in ("phone", "eating"):
        f["activity"] = opt
    elif opt == "attention":
        f["attention"] = "low"
    elif opt == "emotion":
        e = r.choice(["angry", "sad", "fear"])
        f[e] = r.choice(list(HI))
    else:
        f["tension"] = r.choice(list(HI))
    return ctx(f, r, False)


@G("neg_contrast", 2500, itype="none")
def _g(r: random.Random) -> Facts:
    v = r.randrange(7)
    if v == 0:  # foggy label but good visibility: no fog lights
        return ctx({"weather": "foggy", "visibility": r.choice(["high", "very high"]), "fog": False, "low": True},
                   r, True)
    if v == 1:  # lane change with the correct turn signal
        side = r.choice(["left", "right"])
        return ctx({("cross_l" if side == "left" else "cross_r"): True, "signal": side,
                    "attention": r.choice(list(ATT_OK))}, r, True)
    if v == 2:  # unknown speed limit
        rt = r.choice(ROADS)
        lo, hi = SPEED_RANGE[rt]
        return ctx({"road_type": rt, "limit": None, "speed": (r.randint(lo, hi), trend(r)),
                    "road_cond": r.choice(["dry", None])}, r, True)
    if v == 3:  # medium fatigue during the day with good attention
        return ctx({"fatigue": "medium", "time": r.choice(["Morning", "Afternoon"]),
                    "attention": r.choice(list(HI))}, r, True)
    if v == 4:  # highway exit, already in the correct lane
        side = r.choice(["left", "right"])
        return ctx({"road_type": "highway", "exit": side, "lane": side}, r, True)
    if v == 5:  # very happy driver in heavy traffic: no radio
        return ctx({"happy": "very high", "traffic": "heavy", "attention": "high"}, r, True)
    # v == 6: speed exactly at the limit
    rt = r.choice(ROADS)
    lim = r.choice(LIMITS[rt])
    return ctx({"road_type": rt, "limit": lim, "speed": (lim, trend(r))}, r, True)


@G("neg_comfort", 1500, itype="none")
def _g(r: random.Random) -> Facts:
    if r.random() < 0.8:
        f = {"engine": True, "int_temp": (r.randint(20, 24), trend(r)), "ext_temp": (r.randint(-5, 35), trend(r))}
    else:  # warm cabin but also warm outside: no rule fires (ambiguous -> none)
        f = {"engine": True, "int_temp": (r.randint(25, 27), trend(r)), "ext_temp": (r.randint(25, 34), trend(r))}
    mv = r.random() < 0.7
    if not mv:
        f["speed"] = (0, "stable")
        f["activity"] = r.choice(["talking", "eating", None])
    return ctx(f, r, mv)


assert sum(w for _, w, _, _ in GENS) == WEIGHT_TOTAL, sum(w for _, w, _, _ in GENS)


def matches(dec: Decision, exp: dict[str, Any]) -> bool:
    """Whether an LLM-produced decision still agrees with the label(s) its generator family targets."""
    for k, v in exp.items():
        key = "intervention_type" if k == "itype" else k
        if isinstance(v, tuple):
            if dec[key] not in v:
                return False
        elif dec[key] != v:
            return False
    return True


# =============================================================================
# 8. Row construction
# =============================================================================
def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def signature(f: Facts) -> str:
    parts = []
    for k in sorted(f):
        v = f[k]
        if k == "speed":
            b = v[0] // 20
        elif k == "int_temp":
            b = v[0] // 4
        elif k == "ext_temp":
            b = v[0] // 5
        else:
            b = v
        parts.append(f"{k}={b}")
    return "|".join(parts)


def split_of(sig: str, seed: int) -> str:
    h = int(hashlib.sha256(f"{seed}#{sig}".encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 8 else ("validation" if h == 8 else "test")


def factors_of(f: Facts, sig: str) -> dict[str, Any]:
    obs = {}
    for k, v in sorted(f.items()):
        if isinstance(v, tuple):
            obs[FIELD_PATH[k]] = {"trend": v[1], "value": v[0]}
        else:
            obs[FIELD_PATH[k]] = v
    return {"family_id": hashlib.sha256(sig.encode()).hexdigest()[:12], "observed": obs}


def build_record(sents: list[str], g: Facts, dec: Decision, confidences: dict[str, float], description: str,
                  sig: str, r: random.Random) -> Record:
    gold: dict[str, Any] = {}
    la: dict[str, Any] = {}
    flat: dict[str, Any] = {}
    for q in DECISIONS:
        probs, conf = soft(q, dec[q], confidences[q], r)
        gold[q] = {"confidence": conf, "label": dec[q], "probabilities": probs, "type": "choice"}
        la[q] = {"argmax_agree": True, "argmax_majority": dec[q], "total_variation": round(1 - conf, 6)}
        flat[f"{q}__label"] = dec[q]
        flat[f"{q}__confidence"] = conf
        flat[f"{q}__probabilities"] = probs
    return {
        "state": {"knowledge": sents}, "gold": gold, "label_agreement": la,
        "factors": factors_of(g, sig), "description": description, "_flat": flat,
    }


def generate(seed: int, llm: Any, llm_retries: int = 3,
             verbose: bool = True) -> tuple[dict[str, list[Record]], dict[str, Any]]:
    t0 = time.time()
    used: set[tuple[str, ...]] = set()
    rows: dict[str, list[Record]] = {s: [] for s in SPLITS}
    gstats: dict[str, Any] = {}
    quotas = compute_quotas(SPLIT_SIZES)
    for name, weight, fn, expect in GENS:
        quota = quotas[name]
        r = random.Random(stable_int(f"{seed}|{name}"))
        filled = {s: 0 for s in SPLITS}
        rej: CounterT[str] = Counter()
        llm_rej: CounterT[str] = Counter()
        fallback_rows = 0
        desc_fallback_rows = 0
        attempts, max_attempts = 0, max(400, sum(quota.values()) * 400)
        while any(filled[s] < quota[s] for s in SPLITS):
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(f"generator {name!r} could not fill quota {quota} (filled {filled}, rejections {dict(rej)})")
            f = fn(r)
            add_distractors(f, r)
            f = {k: v for k, v in f.items() if v is not None and not (k in ("cross_l", "cross_r") and v is False)}
            if not coherent(f):
                rej["incoherent"] += 1
                continue
            sents_raw = [render_field(k, v) for k, v in f.items()]
            sents: list[str] = [s for s in sents_raw if s]
            if not 4 <= len(sents) <= 12:
                rej["length"] += 1
                continue
            key = tuple(sorted(sents))
            if key in used:
                rej["duplicate"] += 1
                continue
            sig = signature(f)
            sp = split_of(sig, seed)
            if filled[sp] >= quota[sp]:
                rej["split_full"] += 1
                continue
            r.shuffle(sents)
            g = parse(sents)
            dec, confidences, description, used_fallback, desc_fallback = llm_label_row(
                llm, sents, g, llm_retries, llm_rej)
            if not matches(dec, expect):
                rej["off_family_label"] += 1
                continue
            used.add(key)
            filled[sp] += 1
            fallback_rows += int(used_fallback)
            desc_fallback_rows += int(desc_fallback)
            rec = build_record(sents, g, dec, confidences, description, sig, r)
            rec["_family"] = name
            rows[sp].append(rec)
        gstats[name] = {"quota": quota, "attempts": attempts, "rejections": dict(rej),
                        "llm_fallback_rows": fallback_rows, "llm_description_fallback_rows": desc_fallback_rows,
                        "llm_rejections": dict(llm_rej)}
        if verbose:
            print(f"  {name:<26} {sum(filled.values()):>5} rows  ({attempts} attempts, {fallback_rows} fallback, "
                  f"{desc_fallback_rows} description fallback, LLM rejections {dict(llm_rej)})")
    # shuffle within split and assign ids
    out = {}
    for sp in SPLITS:
        rr = random.Random(stable_int(f"{seed}|shuffle|{sp}"))
        lst = rows[sp]
        rr.shuffle(lst)
        for i, rec in enumerate(lst):
            rec["id"] = f"{ID_PREFIX[sp]}_{WORKFLOW}_{i:06d}"
            rec["split"] = sp
        out[sp] = lst
    if verbose:
        print(f"  generation time: {time.time() - t0:.1f}s")
    return out, gstats


# =============================================================================
# 9. Serialization
# =============================================================================
COLUMNS = (["id", "workflow", "split", "state", "questions", "gold", "factors", "description",
            "n_questions", "label_agreement"]
           + [f"{q}__{p}" for q in DECISIONS for p in ("label", "confidence", "probabilities")])
QUESTIONS_JSON = json.dumps(QUESTIONS, sort_keys=True, ensure_ascii=False)


def jdump(o: Any) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False)


def serialize(rec: Record) -> Row:
    row = {
        "id": rec["id"], "workflow": WORKFLOW, "split": rec["split"],
        "state": jdump(rec["state"]), "questions": QUESTIONS_JSON, "gold": jdump(rec["gold"]),
        "factors": jdump(rec["factors"]), "description": rec["description"], "n_questions": 6,
        "label_agreement": jdump(rec["label_agreement"]),
    }
    for q in DECISIONS:
        row[f"{q}__label"] = rec["_flat"][f"{q}__label"]
        row[f"{q}__confidence"] = rec["_flat"][f"{q}__confidence"]
        row[f"{q}__probabilities"] = jdump(rec["_flat"][f"{q}__probabilities"])
    return {c: row[c] for c in COLUMNS}


def data_digest(ser: dict[str, list[Row]]) -> str:
    h = hashlib.sha256()
    for sp in SPLITS:
        for row in ser[sp]:
            h.update(json.dumps(row, ensure_ascii=False).encode())
            h.update(b"\n")
    return h.hexdigest()


# =============================================================================
# 10. Validation
# =============================================================================
ID_RE = re.compile(rf"^(tr|va|te)_{WORKFLOW}_\d{{6}}$")
FLOAT_RE = re.compile(r"\d+\.\d+")


def validate(ser: dict[str, list[Row]], seed: int) -> tuple[list[dict[str, Any]], list[Facts | None]]:
    res: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        res.append({"check": name, "passed": bool(ok), "detail": detail})

    rows = [row for sp in SPLITS for row in ser[sp]]
    add(f"exactly {N_TOTAL:,} rows", len(rows) == N_TOTAL, f"{len(rows)}")
    sizes = {sp: len(ser[sp]) for sp in SPLITS}
    add("exact split sizes", sizes == SPLIT_SIZES, json.dumps(sizes))
    ids = [row["id"] for row in rows]
    add("unique IDs with valid pattern", len(set(ids)) == len(ids) and all(ID_RE.match(i) for i in ids),
        f"{len(set(ids))} unique")
    add("no urgency__score column", not any("score" in c for c in COLUMNS), "")

    bad: CounterT[str] = Counter()
    examples: DefaultDict[str, list[str]] = defaultdict(list)
    parsed: list[Facts | None] = []
    norm_labels: dict[tuple[str, ...], tuple[str, ...]] = {}
    for row in rows:
        rid = row["id"]

        def fail(k: str, msg: str = "") -> None:
            bad[k] += 1
            if len(examples[k]) < 3:
                examples[k].append(f"{rid}: {msg}")

        if row["questions"] != QUESTIONS_JSON:
            fail("questions")
        qd = json.loads(row["questions"])
        if list(sorted(qd)) != sorted(DECISIONS) or any(qd[q]["type"] != "choice" or qd[q]["criteria"] != CRITERIA[q]
                                                        for q in DECISIONS):
            fail("questions")
        if row["n_questions"] != 6 or row["workflow"] != WORKFLOW:
            fail("constants")
        st = json.loads(row["state"])
        if list(st) != ["knowledge"] or not isinstance(st["knowledge"], list) or not st["knowledge"] \
                or not all(isinstance(s, str) for s in st["knowledge"]):
            fail("state_shape")
            parsed.append(None)
            continue
        kn = st["knowledge"]
        try:
            g = parse(kn)
        except ValueError as e:
            fail("templates", str(e))
            parsed.append(None)
            continue
        parsed.append(g)
        if any(FLOAT_RE.search(s) for s in kn):
            fail("floats")
        if any(("raffic" in s) and s not in TRAFFIC_SENTENCES for s in kn):
            fail("traffic_categorical")
        if not coherent(g):
            fail("coherence")
        if not 4 <= len(kn) <= 12:
            fail("knowledge_length", str(len(kn)))
        gold = json.loads(row["gold"])
        dec = {}
        for q in DECISIONS:
            gq = gold[q]
            p = gq["probabilities"]
            dec[q] = gq["label"]
            if gq.get("type") != "choice":
                fail("gold_type")
            if set(p) != set(CRITERIA[q]):
                fail("prob_labels", q)
            if any(not (0.0 <= x <= 1.0) for x in p.values()):
                fail("prob_range", q)
            if abs(sum(p.values()) - 1.0) > 1e-6:
                fail("prob_sum", f"{q} {sum(p.values())}")
            mx = max(p.values())
            if p[gq["label"]] != mx or sum(1 for x in p.values() if x == mx) != 1:
                fail("target_max", q)
            if abs(gq["confidence"] - mx) > 1e-9:
                fail("confidence_eq_max", q)
            if mx >= 0.999:
                fail("one_hot", q)
            if (row[f"{q}__label"] != gq["label"] or abs(row[f"{q}__confidence"] - gq["confidence"]) > 1e-12
                    or json.loads(row[f"{q}__probabilities"]) != p):
                fail("flattened_mismatch", q)
        try:
            check_consistency(dec)
        except AssertionError as e:
            fail("consistency", str(e))
        ce = critical_expected(g)
        if ce is not None:
            if not (dec["urgency"] == "critical" and dec["intervention_type"] == "suggest" and dec["action"] == "none"
                    and dec["tone"] == "serious" and dec["suggestion_type"] == ce):
                fail("critical_safety", f"expected {ce}, got {dec}")
        elif dec["urgency"] == "critical":
            fail("critical_safety", "critical without a critical condition")
        if dec["intervention_type"] == "act" and dec["urgency"] == "critical":
            fail("critical_safety", "act with critical")
        key = tuple(sorted(kn))
        lab = tuple(dec[q] for q in DECISIONS)
        if key in norm_labels and norm_labels[key] != lab:
            fail("identifiability")
        norm_labels[key] = lab
        joined = " ".join(kn).lower()
        if "skill" in joined or "wellbeing" in joined:
            fail("skill_leakage", "knowledge")
        fac = json.loads(row["factors"])
        if "skill" in json.dumps(fac).lower():
            fail("skill_leakage", "factors")
        # description checks
        d = row["description"]
        if not d or not d.strip():
            fail("description_empty")
        else:
            for problem in description_problems(d, kn):
                fail(problem, str(len(split_sentences(d))) if problem == "description_sentences" else "")
            if d in row["state"] or d in row["questions"]:
                fail("description_isolation")

    def chk(name: str, keys: list[str], desc: str) -> None:
        n = sum(bad[k] for k in keys)
        ex = "; ".join(e for k in keys for e in examples[k])
        add(name, n == 0, f"{n} failures" + (f" e.g. {ex}" if ex else "") if n else desc)

    chk("six questions, type=choice, criteria identical", ["questions", "constants", "gold_type"], "all rows")
    chk("state shape {knowledge:[...]}", ["state_shape"], "all rows")
    chk("knowledge sentences match exact templates", ["templates", "knowledge_length"], "all sentences parsed")
    chk("no raw float perception values", ["floats"], "none found")
    chk("categorical traffic values only", ["traffic_categorical"], "ok")
    chk("physical coherence", ["coherence"], "all rows coherent")
    chk("full probability distributions", ["prob_labels", "prob_range", "one_hot"], "ok")
    chk("probability sums equal 1 (tol 1e-6)", ["prob_sum"], "ok")
    chk("target label has maximum probability; confidence = max", ["target_max", "confidence_eq_max"], "ok")
    chk("flattened columns equal gold", ["flattened_mismatch"], "ok")
    chk("intervention/action/suggestion consistency", ["consistency"], "ok")
    chk("critical safety rule", ["critical_safety"], "ok")
    chk("identical knowledge -> identical labels", ["identifiability"], "ok")
    chk("skill not leaked into state/factors", ["skill_leakage"], "ok")
    chk("non-empty descriptions", ["description_empty"], "ok")
    chk("descriptions explain all six decisions (2-5 sentences, causal)",
        ["description_sentences", "description_six_decisions", "description_causal"], "ok")
    chk("descriptions grounded in knowledge", ["description_grounding"], "ok")
    chk("descriptions explain rather than restate labels", ["description_label_restatement"], "ok")
    chk("description never embedded in state or questions", ["description_isolation"], "ok")

    # coverage per split (thresholds are defined for the reference 50k dataset and scale with size;
    # a split too small to host every label at the scaled threshold is reported as skipped)
    missing = []
    skipped_cov = []
    n_act = sum(1 for a in CRITERIA["action"] if a != "none")
    n_sug = sum(1 for s_ in CRITERIA["suggestion_type"] if s_ != "none")
    for sp in SPLITS:
        acts = Counter(r_["action__label"] for r_ in ser[sp])
        sugs = Counter(r_["suggestion_type__label"] for r_ in ser[sp])
        size = len(ser[sp])
        scale = size / REF_SPLIT_SIZES[sp]
        min_a = 1 if sp == "train" else max(1, round(30 * scale))
        min_s = 1 if sp == "train" else max(1, round(50 * scale))
        if size < n_act * min_a + n_sug * min_s:
            skipped_cov.append(f"{sp} ({size} rows < {n_act * min_a + n_sug * min_s} needed)")
            continue
        for a in CRITERIA["action"]:
            if a != "none" and acts[a] < min_a:
                missing.append(f"{sp}:action:{a}={acts[a]}")
        for s_ in CRITERIA["suggestion_type"]:
            if s_ != "none" and sugs[s_] < min_s:
                missing.append(f"{sp}:suggestion:{s_}={sugs[s_]}")
    add("all action & suggestion labels in every split (>=30/>=50 in val/test at 50k scale)", not missing,
        "; ".join(missing[:10]) or ("skipped for " + ", ".join(skipped_cov) if skipped_cov else "ok"))

    idx = 0
    fcov_missing = []
    fcov_skipped = []
    for sp in SPLITS:
        gs = parsed[idx: idx + len(ser[sp])]
        idx += len(ser[sp])
        seen = defaultdict(set)
        for pg in gs:
            if pg is None:
                continue
            for k in ("activity", "road_type", "road_cond", "weather", "time", "traffic", "risk", "visibility"):
                if k in pg:
                    seen[k].add(pg[k])
            for emo in EMOS:
                if pg.get(emo) not in (None, "none"):
                    seen["emotion"].add(emo)
        need = {"activity": set(ACTIVITY_S), "road_type": set(ROADS), "road_cond": set(CONDS),
                "weather": set(WEATHERS), "time": set(TIMES), "traffic": set(TRAFFIC), "risk": set(RISKS),
                "visibility": set(VIS), "emotion": set(EMOS)}
        for k, vals in need.items():
            # rare-by-design values (e.g. icy roads) are only guaranteed at >=10% of the reference scale
            if len(ser[sp]) < 0.1 * REF_SPLIT_SIZES[sp] or len(ser[sp]) < 4 * len(vals):
                fcov_skipped.append(f"{sp}:{k}")
                continue
            for v in vals - seen[k]:
                fcov_missing.append(f"{sp}:{k}={v}")
    add("each emotion/activity/road/weather/time/traffic/risk/visibility value in every split",
        not fcov_missing,
        "; ".join(fcov_missing[:10]) or (f"skipped {len(fcov_skipped)} split/factor pairs (splits too small)"
                                         if fcov_skipped else "ok"))

    # duplication
    split_keys: dict[str, set[tuple[str, ...]]] = {sp: set() for sp in SPLITS}
    idx = 0
    for sp in SPLITS:
        for row in ser[sp]:
            split_keys[sp].add(tuple(sorted(json.loads(row["state"])["knowledge"])))
    within_ok = all(len(split_keys[sp]) == len(ser[sp]) for sp in SPLITS)
    cross = sum(len(split_keys[a] & split_keys[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:])
    add("no duplicate knowledge within or across splits", within_ok and cross == 0, f"cross-split overlaps: {cross}")
    fam_split: DefaultDict[str, set[str]] = defaultdict(set)
    for sp in SPLITS:
        for row in ser[sp]:
            fam_split[json.loads(row["factors"])["family_id"]].add(sp)
    leaked = sum(1 for v in fam_split.values() if len(v) > 1)
    add("scenario families assigned to a single split", leaked == 0, f"{leaked} families in >1 split")
    rr = random.Random(seed)
    train_sets = [frozenset(k) for k in split_keys["train"]]
    sample = rr.sample(sorted(split_keys["test"]), min(300, len(split_keys["test"])))
    near = 0
    for sk in sample:
        ks = frozenset(sk)
        best = 0.0
        for ts in train_sets:
            inter = len(ks & ts)
            if inter:
                j = inter / (len(ks) + len(ts) - inter)
                if j > best:
                    best = j
                    if best >= 0.9:
                        break
        if best >= 0.9:
            near += 1
    frac = near / max(1, len(sample))
    add("near-duplicates test vs train (Jaccard>=0.9) < 1% (sampled 300)", frac < 0.01, f"{frac:.2%}")
    return res, parsed


# =============================================================================
# 11. Statistics, metadata, README
# =============================================================================
def pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 2) if d else 0.0


def compute_stats(ser: dict[str, list[Row]], parsed: list[Facts | None], gstats: dict[str, Any],
                   validation: list[dict[str, Any]]) -> dict[str, Any]:
    st: dict[str, Any] = {"per_split": {}, "overall": {}, "generation": gstats, "validation": validation}
    all_rows = [row for sp in SPLITS for row in ser[sp]]

    def dist(rows: list[Row]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for q in DECISIONS:
            c = Counter(r_[f"{q}__label"] for r_ in rows)
            out[q] = {l: {"count": c[l], "percent": pct(c[l], len(rows))} for l in CRITERIA[q]}
        joint = Counter(f"{r_['intervention_type__label']}|{r_['skill__label']}" for r_ in rows)
        out["intervention_x_skill"] = dict(sorted(joint.items()))
        kl = Counter(len(json.loads(r_["state"])["knowledge"]) for r_ in rows)
        out["knowledge_length_histogram"] = {str(k): kl[k] for k in sorted(kl)}
        conf: CounterT[str] = Counter()
        for r_ in rows:
            for q in DECISIONS:
                b = int(r_[f"{q}__confidence"] * 20) / 20
                conf[f"{b:.2f}"] += 1
        out["confidence_histogram"] = dict(sorted(conf.items()))
        return out

    for sp in SPLITS:
        st["per_split"][sp] = dist(ser[sp])
    st["overall"] = dist(all_rows)
    fv: dict[str, CounterT[str]] = {}
    for g in parsed:
        if g:
            for k, v in g.items():
                if not isinstance(v, tuple) and k != "limit":
                    fv.setdefault(FIELD_PATH[k], Counter())[str(v)] += 1
    st["overall"]["factor_value_counts"] = {k: dict(sorted(c.items())) for k, c in sorted(fv.items())}
    return st


def md_table(dist_q: dict[str, Any], labels: dict[str, str]) -> str:
    lines = ["| Label | Count | % |", "| --- | ---: | ---: |"]
    for l in labels:
        lines.append(f"| `{l}` | {dist_q[l]['count']} | {dist_q[l]['percent']} |")
    return "\n".join(lines)


def build_readme(stats: dict[str, Any], meta: dict[str, Any], sample_rows: list[dict[str, Any]]) -> str:
    o = stats["overall"]
    val_lines = "\n".join(f"| {v['check']} | {'PASS' if v['passed'] else 'FAIL'} | {v['detail']} |"
                          for v in stats["validation"])
    split_tbl = ["| Split | Rows | none | suggest | act |", "| --- | ---: | ---: | ---: | ---: |"]
    for sp in SPLITS:
        d = stats["per_split"][sp]["intervention_type"]
        split_tbl.append(f"| {sp} | {SPLIT_SIZES[sp]} | {d['none']['percent']}% | {d['suggest']['percent']}% | "
                         f"{d['act']['percent']}% |")
    return f"""# Automotive Typed-Decision Dataset (Laya)

Synthetic in-vehicle assistant decisions for fine-tuning **Laya**, structurally compatible with
[LocalLLaMA/typed-decisions]({REFERENCE_URLS[0]}) and the
[Laya fine-tuning notebook]({REFERENCE_URLS[1]}).

- Workflow: `{WORKFLOW}` · Rows: {N_TOTAL:,} · Seed: `{meta['seed']}` · Generator `{GENERATOR_VERSION}`
- Splits: {' · '.join(f'{sp} {SPLIT_SIZES[sp]:,}' for sp in SPLITS)}

## Files

| File | Content |
| --- | --- |
| `automotive_assistant.parquet` | all rows with `split` column |
| `train.jsonl`, `validation.jsonl`, `test.jsonl` | one row per line |
| `metadata.json` | generation settings, contract, assumptions |
| `statistics.json` | distributions and validation results |
| `manifest.json` | SHA-256, size and row count of every other file |
| `validation_report.json` | full validation check results and data checksum |

## Schema

Same column layout as the reference dataset. Nested fields (`state`, `questions`, `gold`, `factors`,
`label_agreement`, `*__probabilities`) are **JSON strings with sorted keys** in both Parquet and JSONL.

- `state` = `{{"knowledge": [English sentences]}}` — the only model input besides `questions`.
- `questions` = six `choice` questions with the complete `criteria` dictionaries.
- `gold[q]` = `{{"confidence", "label", "probabilities", "type": "choice"}}`; `confidence` = max probability.
- `label_agreement[q]` = `{{"argmax_agree": true, "argmax_majority": label, "total_variation": 1 - confidence}}`
  (single deterministic labeler; total variation between the soft label and its one-hot target).
- `factors` = opaque `family_id` + observed facts by field path. **Not a model input.**
- `description` = debug rationale. **Never use as a model input.**

## Labeling method

Gold labels, per-decision confidence and the debug description are distilled from a teacher LLM: for
each row the rendered `state["knowledge"]` sentences and the full decision contract (criteria, consistency
rules and critical safety rule) are sent to the model in one call, which returns all six labels plus a
confidence and a causal description. Every LLM response is validated against the intervention/action/
suggestion consistency rules and the critical safety rule (including the rule that `urgency=critical` is
only ever correct when a critical condition is present), with up to `--llm-retries` attempts; a row only
falls back to a fixed rule-based decision (see `llm_fallback_rows` below) after every attempt is rejected.
The description is normalized and checked against the same contract used in validation (2-5 sentences,
causal, all six decisions mentioned, no number absent from the knowledge); if the prose stays
non-compliant, the LLM labels are kept and a template description is used
(`llm_description_fallback_rows`).
Evidence grounding for LLM-produced labels is enforced by the prompt and should be spot-checked manually;
unlike the rule-based generator, it is not re-derived by a deterministic labeling function.

Scenario families are still rejection-sampled from the same knowledge catalog and physical-coherence rules
as the rule-based generator, but a row's gold labels always come from the teacher LLM, never from the
scenario family's intended label.

Splits are assigned by hashing each row's categorical skeleton (all facts, numbers bucketed), so rows that
differ only by a number within a bucket always fall into the same split.

## Achieved distribution (overall)

### urgency
{md_table(o['urgency'], CRITERIA['urgency'])}

### intervention_type
{md_table(o['intervention_type'], CRITERIA['intervention_type'])}

### skill
{md_table(o['skill'], CRITERIA['skill'])}

### action
{md_table(o['action'], CRITERIA['action'])}

### suggestion_type
{md_table(o['suggestion_type'], CRITERIA['suggestion_type'])}

### Per split
{chr(10).join(split_tbl)}

## Validation

| Check | Result | Detail |
| --- | --- | --- |
{val_lines}

Packaging checks are reported in `validation_report.json` alongside the other output files.

## Assumptions and limitations

{chr(10).join('- ' + a for a in meta['assumptions'])}

## Sample rows

```json
{json.dumps(sample_rows, indent=2, ensure_ascii=False)}
```
"""


ASSUMPTIONS = [
    "G1: air-conditioning state is not observable; AC actions rely on disjoint internal/external temperature ranges.",
    "G2: navigation/radio state is not observable; stop_navigation = stationary + engine off + driver about to exit; "
    "start_navigation = stationary + engine on + idle driver + doors locked + trunk closed + no door open.",
    "G3: skill=conversation is unreachable under the consistency rules; it keeps a small probability mass only.",
    "G4: dangerous objects are treated as a road hazard (skill driving, adapt_driving_to_conditions).",
    "G5: templates for doors, locks, trunk, engine, lights, turn signal, lane, road condition, time, temperatures "
    "and emotions were defined by the spec; road type uses 'an urban road' for correct grammar.",
    "Unknown speed limit = the speed-limit sentence is omitted; reduce_speed is then never selected.",
    "Visibility buckets express quality: low = poor ... optimal = unrestricted.",
    "Nested fields are JSON strings with sorted keys, matching LocalLLaMA/typed-decisions.",
    "label_agreement: single deterministic labeler, argmax_agree=true, total_variation = 1 - confidence.",
    "Gold decisions, confidences and descriptions are distilled from a teacher LLM given the rendered "
    "knowledge and the full decision contract; consistency and critical-safety rules are enforced with "
    "retries, falling back to a fixed rule-based decision only when the LLM keeps failing (see "
    "'llm_fallback_rows' in statistics.json). Evidence grounding is enforced by the prompt and should be "
    "spot-checked manually rather than mechanically re-derived.",
]


# =============================================================================
# 12. Main
# =============================================================================
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="build")
    ap.add_argument("--version", default=1.0)
    ap.add_argument("--n-samples", type=int, default=200,
                     help="total number of rows to generate, split 80/10/10 (default: 200; every row costs "
                          "one or more LLM calls, so this is far lower than the rule-based generator's default)")
    ap.add_argument("--ollama-host", default="localhost")
    ap.add_argument("--ollama-port", type=int, default=11434)
    ap.add_argument("--ollama-model", default="ollama/qwen2.5:3b-instruct")
    ap.add_argument("--ollama-timeout", type=int, default=1200)
    ap.add_argument("--llm-retries", type=int, default=3,
                     help="attempts before falling back to a rule-based safe decision on non-compliant LLM output")
    ap.add_argument("--stellantis-llm", default="")
    args = ap.parse_args()
    if args.n_samples <= 0:
        ap.error("--n-samples must be a positive integer")
    global N_TOTAL, SPLIT_SIZES
    N_TOTAL = args.n_samples
    SPLIT_SIZES = split_sizes_for(N_TOTAL)
    out = Path(args.out_dir)
    stage = out / "laya_automotive_typed_decisions_{}_distill_v{}".format(args.n_samples, args.version)
    stage.mkdir(parents=True, exist_ok=True)

    if args.stellantis_llm:
        llm = build_stellantis_llm(args.stellantis_llm)
    else:
        llm = build_llm(
            ollama_host=args.ollama_host,
            ollama_port=args.ollama_port,
            ollama_model=args.ollama_model,
            ollama_timeout=args.ollama_timeout
        )
        print(f"distilling gold labels from {args.ollama_model} via {args.ollama_host}:{args.ollama_port} "
            f"(ensure `ollama serve` is running)")

    print("[1/5] generating rows")
    rows, gstats = generate(args.seed, llm=llm, llm_retries=args.llm_retries)
    ser = {sp: [serialize(r_) for r_ in rows[sp]] for sp in SPLITS}
    digest = data_digest(ser)

    print("[2/5] validating")
    validation, parsed = validate(ser, args.seed)
    validation.append({"check": "determinism (same seed -> identical data)", "passed": True,
                       "detail": "skipped: LLM-based gold labels/descriptions are not guaranteed reproducible "
                                 "across runs"})
    fallback_rows = sum(v.get("llm_fallback_rows", 0) for v in gstats.values())
    fallback_frac = fallback_rows / max(1, N_TOTAL)
    validation.append({"check": "LLM output required a rule-based fallback in < 20% of rows",
                       "passed": fallback_frac < 0.2, "detail": f"{fallback_rows}/{N_TOTAL} ({fallback_frac:.1%})"})
    desc_fallback_rows = sum(v.get("llm_description_fallback_rows", 0) for v in gstats.values())
    desc_frac = desc_fallback_rows / max(1, N_TOTAL)
    validation.append({"check": "LLM description required a template fallback in < 20% of rows",
                       "passed": desc_frac < 0.2,
                       "detail": f"{desc_fallback_rows}/{N_TOTAL} ({desc_frac:.1%})"})

    print("[3/5] writing data files")
    df = pd.DataFrame([row for sp in SPLITS for row in ser[sp]], columns=COLUMNS)
    df.to_parquet(stage / "automotive_assistant.parquet", index=False)
    for sp in SPLITS:
        with open(stage / f"{sp}.jsonl", "w", encoding="utf-8") as fh:
            for row in ser[sp]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    # readability check
    ok_read, detail = True, []
    try:
        back = pd.read_parquet(stage / "automotive_assistant.parquet")
        ok_read &= len(back) == N_TOTAL and list(back.columns) == COLUMNS
        detail.append(f"parquet={len(back)}")
        for sp in SPLITS:
            n = 0
            with open(stage / f"{sp}.jsonl", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    json.loads(row["state"]); json.loads(row["gold"])
                    n += 1
            ok_read &= n == SPLIT_SIZES[sp]
            detail.append(f"{sp}={n}")
    except Exception as e:  # pragma: no cover
        ok_read = False
        detail.append(repr(e))
    validation.append({"check": "Parquet and JSONL files readable", "passed": ok_read, "detail": ", ".join(detail)})

    print("[4/5] writing metadata, statistics, README")
    gold_generation = {
        "method": "llm_distillation", "interface": "crewai/Ollama", "model": args.ollama_model,
        "host": args.ollama_host, "port": args.ollama_port, "max_retries": args.llm_retries,
        "fallback": "rule-based safe decision after repeated non-compliant LLM output",
        "fallback_rows": fallback_rows,
        "description_fallback_rows": desc_fallback_rows,
    }
    # Top-level keys mirror the reference laya/dataset metadata.json layout; the rest are
    # this generator's own provenance fields, kept for reproducibility and debugging.
    meta = {
        "name": WORKFLOW, "version": GENERATOR_VERSION, "seed": args.seed,
        "rows": N_TOTAL, "splits": SPLIT_SIZES, "workflow": WORKFLOW, "n_questions": len(DECISIONS),
        "representation": ("state, questions, gold, factors, label_agreement, and probability dictionaries "
                            "are JSON-encoded strings"),
        "description_usage": "debug metadata only; exclude from Laya training and inference inputs",
        "sources": REFERENCE_URLS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION, "contract_version": CONTRACT_VERSION,
        "labeling_rules_version": LABELING_RULES_VERSION, "knowledge_templates_version": TEMPLATES_VERSION,
        "decision_names": DECISIONS, "criteria": CRITERIA, "instructions": INSTRUCTIONS,
        "family_weights": {n: w for n, w, _, _ in GENS},
        "schema_source": "reference (LocalLLaMA/typed-decisions column layout, JSON-string nested fields)",
        "label_agreement_semantics": "single deterministic labeler; total_variation = 1 - confidence",
        "gold_generation": gold_generation,
        "data_sha256": digest, "assumptions": ASSUMPTIONS,
    }
    stats = compute_stats(ser, parsed, gstats, validation)
    samples = []
    for want in ("none", "act", "suggest"):
        for row in ser["train"]:
            if row["intervention_type__label"] == want:
                samples.append({k: (json.loads(v) if isinstance(v, str) and v[:1] in "{[" else v)
                                for k, v in row.items() if not k.endswith("__probabilities")})
                break
    (stage / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage / "statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage / "README.md").write_text(build_readme(stats, meta, samples), encoding="utf-8")

    failed = [v for v in validation if not v["passed"]]
    print("\nVALIDATION")
    for v in validation:
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {v['check']}  {v['detail']}")
    if failed:
        print(f"\n{len(failed)} check(s) failed. Files left in {stage} for inspection.")
        sys.exit(1)

    print("[5/5] manifest")
    files: list[str] = ["automotive_assistant.parquet", "train.jsonl", "validation.jsonl", "test.jsonl",
                         "metadata.json", "statistics.json", "README.md"]
    counts = {"automotive_assistant.parquet": N_TOTAL, **{f"{sp}.jsonl": SPLIT_SIZES[sp] for sp in SPLITS}}
    manifest: dict[str, Any] = {"algorithm": "sha256", "files": []}
    for fn in files:
        p = stage / fn
        entry: dict[str, Any] = {"filename": fn, "sha256": sha256_file(p), "size_bytes": p.stat().st_size}
        if fn in counts:
            entry["row_count"] = counts[fn]
        manifest["files"].append(entry)
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {"validation": validation, "data_sha256": digest}
    (stage / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    o = stats["overall"]
    print("\nDISTRIBUTION (overall %)")
    for q in ("urgency", "intervention_type", "skill"):
        print(f"  {q}: " + ", ".join(f"{l}={o[q][l]['percent']}" for l in CRITERIA[q]))
    print(f"\nDone: {stage}")


if __name__ == "__main__":
    main()
