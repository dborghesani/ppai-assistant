# Automotive Typed-Decision Dataset — Generation Specification

> **Audience:** an autonomous coding/data agent.
> **Goal:** generate a synthetic automotive decision-making dataset of **exactly 50,000 rows** for fine-tuning **Laya**, packaged as a ZIP archive.
> **Keywords:** **MUST** / **MUST NOT** = mandatory. **SHOULD** = strong default; deviate only with a documented reason. **MAY** = optional.

---

## 0. How to use this document

### 0.1 Rule priority

When two rules appear to conflict, apply them in this order (1 wins):

1. **Critical safety rule** (§4.3)
2. **Intervention consistency rules** (§4.2)
3. **Evidence grounding and identifiability** (§4.1, §8)
4. **Schema compatibility** with the reference dataset and notebook (§9)
5. **Exact knowledge sentence templates** (§6)
6. **Coverage and distribution targets** (§7, §12)

Coverage targets **never** justify a weakly supported or mislabeled row. If a coverage target cannot be met with valid evidence, report the shortfall instead of forcing it.

### 0.2 Core design principle

Labels **MUST** be produced by a single **deterministic labeling function**:

```
gold = label(state["knowledge"])
```

- The function reads **only** the serialized knowledge sentences (or the equivalent parsed buckets).
- It **MUST NOT** read hidden generation variables, scenario-family IDs, or author intent.
- Because labels are a pure function of the knowledge, two rows with identical knowledge always get identical labels, and validation can re-derive every label (§13).

### 0.3 References (inspect before building)

- Dataset: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- Notebook: https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb

If the references are reachable, you **MUST** mirror their exact column types and nested-field serialization (native structs vs JSON strings, key names inside `gold`, `label_agreement` semantics). If they are not reachable, use the fallback schema in §9 and state this in the final report and README.

---

## 1. Deliverables

A single ZIP archive named `automotive_typed_decisions.zip` containing exactly:

| File | Content |
| --- | --- |
| `automotive_assistant.parquet` | All 50,000 rows, with a `split` column |
| `train.jsonl` | 40,000 rows, one JSON object per line |
| `validation.jsonl` | 5,000 rows |
| `test.jsonl` | 5,000 rows |
| `metadata.json` | Generation settings (§14.1) |
| `statistics.json` | Distributions and validation results (§14.2) |
| `manifest.json` | SHA-256 and byte size of every other file (§14.3) |
| `README.md` | Dataset card (§14.4) |

- Random seed: **MUST** be deterministic. Default `seed = 42`. Record it in `metadata.json`.
- Re-running the generator with the same seed **MUST** reproduce byte-identical data files.

---

## 2. Execution workflow

Follow these steps in order:

1. **Inspect references** (§0.3) and fix the output schema.
2. **Implement the knowledge renderer** (§6): bucket values → exact English sentences.
3. **Implement the labeling function** (§4, §5): knowledge → six gold labels. Write its rule table into `README.md`.
4. **Define scenario families** (§7): each family = a decisive condition + a set of varying context/distractor factors.
5. **Assign families to splits** before generating any row (§12.2).
6. **Generate rows** per family: sample coherent factors (§6.4) → render knowledge → shuffle sentence order (seeded) → compute gold labels → compute probabilities (§10) → write description (§11). The description is always written **last**.
7. **Deduplicate** (§12.3).
8. **Validate** every check in §13. Fix and regenerate until all pass.
9. **Write files**, compute the manifest, build the ZIP, re-open the ZIP and re-verify.
10. **Report** results (§15).

---

## 3. Decision contract (verbatim — do not modify)

Every row has exactly six decisions, in this order, all with `type="choice"`:
`urgency`, `tone`, `intervention_type`, `skill`, `action`, `suggestion_type`.

Each table below is the **complete `criteria` dictionary** (label → description) for its question. Labels are stable tokens. Do not invent, rename, reorder or reword labels or descriptions.

### 3.1 `urgency`
How urgent the situation is. No numeric urgency score may exist anywhere.

| Label | Description |
| --- | --- |
| `none` | No intervention is needed. |
| `low` | Minor issue or optional assistance. |
| `medium` | Intervention is advisable soon. |
| `high` | Prompt intervention is strongly advisable. |
| `critical` | Immediate attention is required to reduce a serious safety risk. |

### 3.2 `tone`
Response style (not a voice identity or TTS model).

| Label | Description |
| --- | --- |
| `calm` | Use a calm and reassuring tone. |
| `enthusiastic` | Use a lively and encouraging tone. |
| `serious` | Use a serious and direct tone. |
| `empathetic` | Use a warm and understanding tone. |
| `discreet` | Use a discreet and non-intrusive tone. |

### 3.3 `intervention_type`

| Label | Description |
| --- | --- |
| `none` | Do not intervene. |
| `suggest` | Communicate a contextual recommendation without directly controlling a function. |
| `act` | Execute one supported reversible vehicle or comfort action. |

### 3.4 `skill`

| Label | Description |
| --- | --- |
| `none` | No skill is required. |
| `conversation` | General conversation, social interaction, or dialogue management. |
| `driving` | Driving, navigation, vehicle state, road safety or coaching. |
| `wellbeing` | Fatigue, attention, emotion or comfort. |

### 3.5 `action`

| Label | Description |
| --- | --- |
| `none` | No direct action. |
| `increase_temperature` | increase temperature |
| `decrease_temperature` | decrease temperature |
| `enable_air_conditioning` | enable air conditioning |
| `disable_air_conditioning` | disable air conditioning |
| `lock_doors` | lock doors |
| `unlock_doors` | unlock doors |
| `start_radio` | start radio |
| `stop_radio` | stop radio |
| `start_navigation` | start navigation |
| `stop_navigation` | stop navigation |
| `enable_sidelights` | enable sidelights |
| `disable_sidelights` | disable sidelights |
| `enable_low_beam_headlights` | enable low beam headlights |
| `disable_low_beam_headlights` | disable low beam headlights |
| `enable_high_beam_headlights` | enable high beam headlights |
| `disable_high_beam_headlights` | disable high beam headlights |
| `enable_fog_lights` | enable fog lights |
| `disable_fog_lights` | disable fog lights |
| `find_rest_area` | Find the next rest area. |

### 3.6 `suggestion_type`
Semantic intent of a spoken suggestion, not its final wording.

| Label | Description |
| --- | --- |
| `none` | No textual suggestion is needed. |
| `take_break` | Recommend stopping at the next safe opportunity because of fatigue or sleepiness. |
| `restore_attention` | Prompt the driver to restore complete attention. |
| `reduce_distraction` | Recommend stopping or postponing a distracting activity. |
| `regulate_emotional_state` | Recommend calming down or taking a short pause when safe. |
| `calm_driving` | Recommend smoother, less tense and less aggressive driving. |
| `reduce_speed` | Recommend reducing speed to a safer or compliant level. |
| `use_turn_signal` | Recommend using the turn signal before a maneuver. |
| `adapt_driving_to_conditions` | Recommend adapting speed, following distance or driving style to road, traffic, visibility, weather or hazards. |
| `secure_vehicle` | Recommend securing doors, trunk or another relevant vehicle state. |
| `prepare_for_weather` | Recommend preparing for current or forecast weather. |
| `prepare_for_maneuver` | Recommend preparing for an upcoming exit, lane change or navigation event. |

---

## 4. Global decision rules

### 4.1 Evidence grounding

- Every non-`none` label **MUST** address a condition explicitly stated in `state["knowledge"]`.
- If materially different labels would be equally reasonable for the same knowledge, either (a) add an allowed knowledge fact that disambiguates, or (b) choose the least specific valid decision. Never hide ambiguity behind soft probabilities.
- One row = one decisive condition. If several conditions are triggered, the decisive one is selected by §4.5.

### 4.2 Intervention consistency (mandatory)

| `intervention_type` | `skill` | `action` | `suggestion_type` |
| --- | --- | --- | --- |
| `none` | `none` | `none` | `none` |
| `act` | `driving` or `wellbeing` | not `none` | `none` |
| `suggest` | `driving` or `wellbeing` | `none` | not `none` |

Additional consistency: `urgency = none` ⇔ `intervention_type = none`.

> Consequence: `skill = conversation` can never be a gold label under these rules. It **MUST** remain in the `criteria` dictionary and receive a small probability mass, but it is never the target.

### 4.3 Critical safety rule

The situations below **MUST** produce `urgency=critical`, `intervention_type=suggest`, `action=none`, `tone=serious`, and the listed suggestion:

| Critical situation (all conditions must be in the knowledge) | `suggestion_type` | `skill` |
| --- | --- | --- |
| Driver appears asleep **and** speed > 0 | `restore_attention` | `wellbeing` |
| Any door or trunk open **and** speed > 0 | `secure_vehicle` | `driving` |
| Dangerous objects detected (any density) **and** speed > 0 | `adapt_driving_to_conditions` | `driving` |
| Driver using a phone **and** (speed ≥ 90 **or** road type highway) | `reduce_distraction` | `wellbeing` |
| Fatigue very high **and** attention low **and** speed > 0 | `take_break` | `wellbeing` |
| Visibility low **and** road icy or snowy **and** speed > 0 | `adapt_driving_to_conditions` | `driving` |
| Driving tension very high **and** speed above the known speed limit | `calm_driving` | `driving` |

`act` is never used for these situations. `act` is limited to supported, reversible, contextually justified actions (§5.1).

### 4.4 Skill inference

`skill` is an **output** only. It **MUST NOT** appear, directly or encoded, in `state`, `id`, `factors`, family IDs, or any other model-visible field.

- `wellbeing`: the decisive evidence concerns fatigue, attention, emotion, physical state, driver activity, or cabin comfort (temperature, air conditioning, radio, rest area).
- `driving`: the decisive evidence concerns speed, speed limit, navigation, vehicle state (doors, trunk, locks, lights, turn signal), road condition, traffic, visibility, lane behavior, hazards, or driving coaching.
- Shared fields (tension, weather, visibility, traffic, …) are resolved by the decisive condition: e.g. tension + speeding → `driving` (`calm_driving`); anger + tension → `wellbeing` (`regulate_emotional_state`).
- `none`: only when `intervention_type = none`.

### 4.5 Decisive-condition precedence

When several triggers fire in the same row:

1. Choose the trigger with the **highest urgency**.
2. Break ties with this fixed priority order:
   `sleeping_while_moving` > `open_door_or_trunk_while_moving` > `dangerous_objects` > `phone_use` > `fatigue_attention` > `visibility_road_condition` > `speeding` > `tension_emotion` > `lane_and_signal` > `lighting` > `highway_exit` > `vehicle_security_stationary` > `weather_preparation` > `cabin_comfort` > `infotainment_navigation`.
3. Non-decisive triggered facts MAY remain in the knowledge as context, but the description must say why they were not decisive.

### 4.6 Urgency guidance

| Urgency | Typical evidence |
| --- | --- |
| `none` | No trigger fires. Includes benign emotions, stationary vehicle with no security issue, correct lights, speed at/below limit, correct turn signal use. |
| `low` | Comfort and infotainment actions; forecast-weather preparation; minor lighting mismatch in daylight; speed 1–10 km/h over the limit in good conditions; mild emotion with good attention. |
| `medium` | High fatigue with adequate attention while moving; eating while moving; lane crossing without signal; missing low beams at night; unnecessary high beams with vehicles around; speed 11–20 km/h over limit; high tension in heavy traffic; wrong lane before a highway exit. |
| `high` | Fatigue very high (attention not low); high fatigue + low attention; low attention at speed ≥ 90 or in heavy traffic; phone use at lower speed; speed > 20 km/h over limit; anger (high/very high) + high tension + heavy traffic; speeding on wet/icy/snowy/gravel roads. |
| `critical` | Only the situations in §4.3. |

Escalation modifiers (apply at most **one** level up, never above the critical definition): Night, heavy traffic, highway at speed ≥ 90, people density medium/high in urban or residential areas.

### 4.7 Tone mapping

| Condition | Tone |
| --- | --- |
| `intervention_type = none` | `discreet` |
| `urgency = critical` | `serious` |
| `urgency = high`, safety-driven (speed, distraction, fatigue, visibility, lane) | `serious` |
| Anger or tension is decisive (non-critical) | `calm` |
| Sadness or fear is decisive | `empathetic` |
| Fatigue or attention decisive at low/medium urgency | `calm` |
| Low-urgency `act` for comfort, lights, locks, navigation | `discreet` |
| Low-urgency, non-safety intervention **and** driver happy (high/very high) | `enthusiastic` |

---

## 5. Label trigger tables

These are the default deterministic rules. Numeric thresholds MAY be adjusted, but the final rules **MUST** be deterministic, non-overlapping, documented in the README, and used unchanged across all splits. "Moving" = speed > 0. "Stationary" = speed 0.

### 5.1 Actions (`intervention_type = act`)

A comfort, infotainment or lighting action **MUST NOT** fire if any higher-priority trigger (§4.5) is present.

| Action | Skill | Required knowledge (all) | Typical urgency |
| --- | --- | --- | --- |
| `increase_temperature` | wellbeing | engine on; internal temperature ≤ 17 | low |
| `decrease_temperature` | wellbeing | engine on; internal temperature 25–27; external temperature < 25 | low |
| `enable_air_conditioning` | wellbeing | engine on; internal temperature ≥ 28; external temperature ≥ 25 | low/medium |
| `disable_air_conditioning` | wellbeing | engine on; internal temperature 18–19; external temperature ≤ 12 | low |
| `lock_doors` | driving | doors unlocked; all doors closed; speed ≥ 10; risk none/low | low |
| `lock_doors` | driving | doors unlocked; stationary; driver **not** about to exit; risk medium/high | medium |
| `unlock_doors` | driving | doors locked; stationary; engine off; driver about to exit; risk none/low | low |
| `start_radio` | wellbeing | moving; driver sad (low) or happy (high/very high); attention high; traffic none/light; no other trigger | low |
| `stop_radio` | wellbeing | moving; driver talking; traffic heavy; attention medium or better | low |
| `start_navigation` | driving | engine on; stationary; approaching nothing; normal/no activity; — *see Appendix A, gap G2* | low |
| `stop_navigation` | driving | stationary; engine off; driver about to exit | low |
| `enable_sidelights` | driving | Evening; stationary; engine on; all lights off | low |
| `disable_sidelights` | driving | Morning/Afternoon; visibility optimal; sidelights on; low beams off | low |
| `enable_low_beam_headlights` | driving | moving; low beams off; Night **or** visibility low/medium | medium |
| `disable_low_beam_headlights` | driving | stationary; engine off; driver about to exit; low beams on | low |
| `enable_high_beam_headlights` | driving | moving; Night; rural or highway; no vehicles; no people; visibility high/very high/optimal; weather not foggy; high beams off; low beams on | low |
| `disable_high_beam_headlights` | driving | high beams on; and (vehicles/traffic present **or** urban/residential **or** foggy) | medium |
| `enable_fog_lights` | driving | moving; weather foggy or snowy; visibility low; fog lights off | medium |
| `disable_fog_lights` | driving | fog lights on; visibility high/very high/optimal; weather not foggy/snowy | low |
| `find_rest_area` | wellbeing | moving; highway or rural; fatigue high; attention medium or high | medium |

Contrast required: `Foggy` weather with visibility high/optimal **MUST NOT** trigger `enable_fog_lights`.

### 5.2 Suggestions (`intervention_type = suggest`)

| Suggestion | Skill | Required knowledge (examples of valid triggers) |
| --- | --- | --- |
| `take_break` | wellbeing | moving; fatigue very high; or fatigue high + attention low (critical if very high + low, §4.3) |
| `restore_attention` | wellbeing | moving; attention low with fatigue low/medium; or asleep while moving (critical) |
| `reduce_distraction` | wellbeing | moving; eating or phone use; talking + attention low is `restore_attention` instead |
| `regulate_emotional_state` | wellbeing | moving; anger high/very high; or sadness medium/high; or fear high/very high |
| `calm_driving` | driving | tension high/very high (without dominant anger); critical when very high + speeding |
| `reduce_speed` | driving | speed above the known speed limit (field present) |
| `use_turn_signal` | driving | lane crossing left/right with turn signal inactive or opposite side; attention medium or better |
| `adapt_driving_to_conditions` | driving | speed ≤ limit or limit unknown, but road wet/icy/snowy/gravel with low/medium visibility or heavy traffic; dangerous objects while moving |
| `secure_vehicle` | driving | door or trunk open while moving (critical); stationary + about to exit + risk medium/high + (unlocked or trunk/door open) |
| `prepare_for_weather` | wellbeing when driver about to exit (personal preparation); driving when moving (e.g. forecast snowy + external ≤ 2) | forecast differs from current weather and is rainy/snowy/foggy |
| `prepare_for_maneuver` | driving | approaching highway exit on one side while driving lane is on the opposite side |

Rules:
- `reduce_speed` requires the speed limit sentence. If the speed limit is absent (unknown), never use `reduce_speed` based on the limit.
- Stationary vehicles never receive fatigue, attention, distraction, emotion, speed or lane suggestions.
- Every suggestion SHOULD depend on **two or more** knowledge facts (see §8.1). Single-fact mappings such as "fatigue → take_break" are forbidden as a pattern.

---

## 6. Knowledge representation

### 6.1 State shape

```json
{
  "state": {
    "knowledge": [
      "Driving visibility is low.",
      "The current weather is foggy."
    ]
  }
}
```

`state` has exactly one key, `knowledge`: a list of English sentences. No `event` key, raw telemetry, dataclass names, field paths, skill hints or routing metadata.

- Each row SHOULD contain 4–12 sentences.
- Each field appears at most once per row.
- Sentence order **MUST** be shuffled with the seeded RNG so position carries no label information.
- Rows SHOULD include 1–4 **distractor** facts (coherent but non-decisive) so the model must identify the decisive condition.

### 6.2 Allowed field catalog (complete — no other fields)

| Scope | Fields |
| --- | --- |
| driving | `VehicleMotion.speed`, `VehicleState.door_open_front_left`, `VehicleState.door_open_front_right`, `VehicleState.door_open_rear_left`, `VehicleState.door_open_rear_right`, `VehicleState.doors_locked`, `VehicleState.lights_on_sidelights`, `VehicleState.lights_on_low_beams`, `VehicleState.lights_on_high_beams`, `VehicleState.lights_on_fog_lights`, `VehicleState.turn_signal`, `VehicleState.trunk_open`, `VehicleState.engine_on`, `VehicleState.internal_temperature`, `LaneTracing.highway_exit`, `LaneTracing.lane_crossing_left`, `LaneTracing.lane_crossing_right`, `LaneTracing.driving_lane` |
| wellbeing | `DetectedObjects.people_around`, `DetectedObjects.vehicles_around`, `DetectedObjects.dangerous_objects_around`, `DriverPhysicalState.activity`, `DriverPhysicalState.attention_level`, `DriverPhysicalState.fatigue_level`, `DriverEmotionState.angry`, `DriverEmotionState.disgust`, `DriverEmotionState.fear`, `DriverEmotionState.happy`, `DriverEmotionState.sad`, `DriverEmotionState.surprise`, `DriverEmotionState.neutral` |
| shared | `DriverDrivingStyle.driving_tension`, `EnvironmentState.external_temperature`, `EnvironmentState.weather`, `EnvironmentState.forecast_weather`, `EnvironmentState.time_of_day`, `EnvironmentState.road_condition`, `EnvironmentState.road_type`, `EnvironmentState.risk_level`, `EnvironmentState.visibility`, `EnvironmentState.traffic`, `TrafficSigns.speed_limit` |

Scope is informational for the generator only; it must never be serialized.

### 6.3 Sentence templates (exact wording — the only allowed sentences)

Placeholders in `<angle brackets>` are replaced by exactly one listed value. Nothing else may vary.

**Numeric fields** (integers, no units, trend ∈ `stable|increasing|decreasing`):

| Field | Template |
| --- | --- |
| speed | `Speed is <int> and is <trend>.` (stationary: `Speed is 0 and is stable.`) |
| internal temperature | `Internal temperature is <int> and is <trend>.` |
| external temperature | `External temperature is <int> and is <trend>.` |
| speed limit | `The speed limit is <int>.` — **omit the sentence when the limit is unknown** |

**Boolean / state fields:**

| Field | Templates |
| --- | --- |
| each door | `The front left door is open.` / `The front left door is closed.` (same pattern for `front right`, `rear left`, `rear right`) |
| doors locked | `The doors are locked.` / `The doors are unlocked.` |
| trunk | `The trunk is open.` / `The trunk is closed.` |
| engine | `The engine is on.` / `The engine is off.` |
| sidelights | `The sidelights are on.` / `The sidelights are off.` |
| low beams | `The low beam headlights are on.` / `The low beam headlights are off.` |
| high beams | `The high beam headlights are on.` / `The high beam headlights are off.` |
| fog lights | `The fog lights are on.` / `The fog lights are off.` |
| turn signal | `The left turn signal is active.` / `The right turn signal is active.` / `The turn signal is inactive.` |
| lane crossing left | `The vehicle is crossing the lane marking on the left.` (emit only when true) |
| lane crossing right | `The vehicle is crossing the lane marking on the right.` (emit only when true) |
| driving lane | `The vehicle is driving in the <left\|center\|right> lane.` |

**Counts / densities (exact):**

| Field | Templates |
| --- | --- |
| people around | `No people are currently detected around the vehicle.` / `People density around the vehicle is <low\|medium\|high>.` |
| vehicles around | `No vehicles are currently detected around the vehicle.` / `Traffic around the vehicle is <low\|medium\|high>.` |
| dangerous objects | `No dangerous objects are currently detected around the vehicle.` / `Dangerous object density around the vehicle is <low\|medium\|high>.` |

**Intensities** (bucket ∈ `low|medium|high|very high`; never expose floats):

| Field | Template | Zero-state |
| --- | --- | --- |
| attention | `Attention level is <bucket>.` | — (always bucketed) |
| fatigue | `Fatigue level is <bucket>.` | `No fatigue is detected.` |
| driving tension | `The driving style shows a <bucket> level of tension.` | `The driving style shows no tension.` |
| angry | `The driver shows a <bucket> level of anger.` | `No anger is detected.` |
| disgust | `The driver shows a <bucket> level of disgust.` | `No disgust is detected.` |
| fear | `The driver shows a <bucket> level of fear.` | `No fear is detected.` |
| happy | `The driver shows a <bucket> level of happiness.` | `No happiness is detected.` |
| sad | `The driver shows a <bucket> level of sadness.` | `No sadness is detected.` |
| surprise | `The driver shows a <bucket> level of surprise.` | `No surprise is detected.` |
| neutral | `The driver shows a <bucket> level of neutral expression.` | `No neutral expression is detected.` |

**Categorical fields (exact):**

| Field | Templates |
| --- | --- |
| highway exit | `The vehicle is not approaching a highway exit.` / `The vehicle is approaching a highway exit on the right.` / `The vehicle is approaching a highway exit on the left.` |
| traffic | `No traffic is currently detected around the vehicle.` / `Current traffic is light.` / `Current traffic is medium.` / `Current traffic is heavy.` |
| activity | `No distracting driver activity is detected.` / `The driver is about to exit the vehicle.` / `Normal driving activity is detected.` / `The driver is talking, which may distract from driving.` / `The driver is eating, which distracts from driving.` / `The driver is using a phone, which seriously distracts from driving.` / `The driver appears to be asleep and unable to drive safely.` |
| weather | `The current weather is <sunny\|cloudy\|rainy\|snowy\|foggy>.` |
| forecast | `The weather forecast predicts <sunny\|cloudy\|rainy\|snowy\|foggy> conditions.` |
| risk | `There is no current risk in this area.` / `The current risk level in this area is <low\|medium\|high>.` |
| road type | `The vehicle is currently driving on a <urban\|rural\|highway\|residential> road.` |
| road condition | `The road condition is <dry\|wet\|icy\|snowy\|gravel>.` |
| time of day | `The time of day is <Morning\|Afternoon\|Evening\|Night>.` |
| visibility | `Driving visibility is optimal.` / `Driving visibility is <low\|medium\|high\|very high>.` |

**Visibility scale (quality, not reduction):** `low` = poor, `medium` = reduced, `high` = good, `very high` = very good, `optimal` = unrestricted.

**Activity mapping** for the requested scenarios: idle → `No distracting driver activity is detected.`; driving → `Normal driving activity is detected.`; talking, eating, on the phone, about to exit, sleeping → their respective sentences.

### 6.4 Physical coherence constraints

Every row **MUST** be physically plausible. Reject and resample rows that violate:

- Engine off ⇒ speed 0. Speed > 0 ⇒ engine on.
- Speed 0 ⇒ no lane crossing, trend `stable`.
- Driver about to exit ⇒ speed 0.
- Sunny weather ⇏ Night (use cloudy/other at night if weather is present).
- Icy or snowy road ⇒ external temperature ≤ 2.
- Snowy weather ⇒ external temperature ≤ 3.
- Speed on residential roads ≤ 60; urban ≤ 90; highway speeds 60–160; rural ≤ 130.
- Speed limits from {30, 50, 70, 90, 110, 130}, consistent with road type (residential 30/50, urban 30/50/70, rural 70/90, highway 90/110/130).
- `Current traffic is …` and `Traffic around the vehicle is …`, if both present, **MUST** agree: none↔none, light↔low, medium↔medium, heavy↔high.
- Heavy traffic ⇒ vehicles not "none".
- Foggy weather ⇒ visibility not `optimal` unless the row is a deliberate fog-with-good-visibility contrast (then `high` or `very high` is allowed; `optimal` is not).
- At most one emotion at `high`/`very high`, except explicit mixed-emotion scenarios.
- Internal temperature 10–40; external temperature −20 to 42.

---

## 7. Scenario coverage

Build scenario families that cover all of the following. Each family has one decisive condition and varying context. Include **negative (no-intervention) families** for every domain.

1. **Emotions:** angry, disgust, fear, happy, sad, surprise, neutral — at all intensity buckets, including intensities that do **not** require intervention.
2. **Activity:** idle, talking, driving, eating, on the phone, about to exit, sleeping — significance depends on speed, traffic and road type (stationary vs moving MUST contrast).
3. **Driving tension:** low, medium, high, very high (nervous/tense/aggressive driving, never physical violence).
4. **Fatigue & attention:** combined with time of day, road type, traffic, speed, visibility, activity.
5. **Traffic:** none, light, medium, heavy.
6. **Visibility:** optimal, low, medium, high, very high.
7. **Driving coaching:** lane changes with correct / missing / wrong-side turn signal; inappropriate or missing lighting; unnecessary high beams; speed below, at and above the limit; unknown-limit rows; driving style not adapted to road or traffic.
8. **Road condition:** dry, wet, icy, snowy, gravel.
9. **Current & forecast weather:** sunny, cloudy, rainy, snowy, foggy — including forecast preparation when the driver is about to exit.
10. **Time of day:** Morning, Afternoon, Evening, Night — combined with fatigue, attention, visibility and lighting.
11. **Road type:** urban, rural, highway, residential.
12. **Area risk:** none, low, medium, high.
13. **Vehicle security:** locked/unlocked, each door open/closed, trunk open/closed, stationary vs moving, driver about to exit, area risk.
14. **Safety:** dangerous objects, door/trunk open while moving, severe distraction, sleeping while moving, dangerous weather/visibility/speed/road combinations.
15. **Cabin comfort:** internal vs external temperature, driver wellbeing, activity, vehicle state.
16. **Navigation & infotainment:** valid start/stop navigation and radio situations (§5.1).

---

## 8. Identifiability and contrastive examples

### 8.1 Multi-condition triggers

Suggestions SHOULD depend on combinations. Target pattern:

```
fatigue very high + attention low + Night + highway + heavy traffic
  → urgency=critical, tone=serious, intervention=suggest,
    skill=wellbeing, action=none, suggestion=take_break
```

### 8.2 Contrastive pairs (required)

Generate pairs that differ in **one** relevant fact where that fact changes the decision. Unrelated decisions stay the same. Required pairs include:

| Base | Changed fact | Effect |
| --- | --- | --- |
| High fatigue, speed 120, highway | speed 0, engine off | intervention → none |
| Sadness medium, attention high, moving | sadness low | suggest → act `start_radio` or none (per rule table) |
| Anger high + tension high + heavy traffic | traffic light | urgency high → medium |
| Foggy, visibility low, fog lights off, moving | visibility high | act `enable_fog_lights` → none |
| Unlocked, moving, no risk | stationary, not exiting, high risk | `lock_doors` low → `lock_doors` medium |
| Lane crossing left, turn signal inactive | left turn signal active | suggest `use_turn_signal` → none |
| Speed 80, limit 70 | limit sentence removed | `reduce_speed` → none (unless conditions trigger `adapt_driving_to_conditions`) |
| Phone use, speed 40, urban | speed 110, highway | high → critical |

### 8.3 Consistency test (applies to all domains)

- Identical knowledge ⇒ identical gold labels (guaranteed by §0.2).
- Near-identical knowledge with different labels ⇒ the differing fact MUST be a trigger input for the changed label.
- A fact change that affects a trigger ⇒ the label MUST change accordingly.

---

## 9. Record schema

### 9.1 Required columns

`id`, `workflow`, `split`, `state`, `questions`, `gold`, `factors`, `description`, `n_questions`, `label_agreement`, and for each decision `d` in the six decisions: `d__label`, `d__confidence`, `d__probabilities`.

- `id`: `auto_<split>_<6-digit index>`, e.g. `auto_train_000001`. Unique. Must not encode family, skill or label.
- `workflow`: `"automotive_assistant"`.
- `split`: `train` | `validation` | `test`.
- `n_questions`: `6`.
- `questions`: identical in every row and split (see below).
- `factors`: generator-side structured values (field path → bucket or number) for the facts **present in the knowledge**, plus an opaque `family_id` (hash). Must not contain labels, skill, scope, or decisive-condition names. Not a model input.
- `description`: see §11. Not a model input.
- `label_agreement`: mirror the reference dataset's semantics. Fallback: `1.0` (single deterministic labeler); document the choice.
- No `urgency__score` or any numeric urgency column.

### 9.2 Fallback nested structure (use only if the reference cannot be inspected)

```json
{
  "questions": {
    "urgency": {
      "type": "choice",
      "criteria": { "none": "No intervention is needed.", "low": "Minor issue or optional assistance.", "...": "..." }
    },
    "tone": { "type": "choice", "criteria": { "...": "..." } },
    "intervention_type": { "type": "choice", "criteria": { "...": "..." } },
    "skill": { "type": "choice", "criteria": { "...": "..." } },
    "action": { "type": "choice", "criteria": { "...": "..." } },
    "suggestion_type": { "type": "choice", "criteria": { "...": "..." } }
  },
  "gold": {
    "urgency": {
      "type": "choice",
      "label": "critical",
      "confidence": 0.91,
      "probabilities": { "none": 0.002, "low": 0.008, "medium": 0.02, "high": 0.06, "critical": 0.91 }
    }
  }
}
```

- JSONL: nested objects as native JSON.
- Parquet: same representation the notebook expects; if unknown, nested structures stored as JSON strings and documented in the README.
- Flattened columns MUST equal the corresponding `gold` values exactly.
- Verify compatibility by loading a sample with the notebook's own data-loading code when possible.

---

## 10. Probabilities

For every decision in every row:

1. Include **every** allowed label of that question (including `conversation` for `skill`).
2. Target probability = confidence, sampled with the seeded RNG:
   - standard rows: 0.85–0.97
   - documented boundary rows (a threshold within one bucket/5 km/h): 0.75–0.85
3. Distribute the remaining mass: ~70% to semantically adjacent labels (adjacent urgency levels; labels sharing the same skill or domain), ~30% spread over all others; every non-target label ≥ 0.0005.
4. Round to 4 decimals, then add the rounding residue to the target so the sum is exactly 1.0 (tolerance 1e-6).
5. Target MUST be the unique maximum; `confidence` MUST equal the maximum probability.
6. Not all distributions may be one-hot; none should be.

Soft labels express calibrated uncertainty, not unresolved ambiguity (§4.1).

---

## 11. Debug description

- Top-level field `description`, non-empty, 2–5 sentences, natural English.
- Written **after** knowledge and gold labels are final. It never influences or validates labels.
- Never included in `state`, `questions`, or any training/inference input.
- It MUST explain the causal reasoning for **all six** decisions, referring only to facts present in the knowledge.

Required content:
- the decisive fact(s) and why they set the urgency;
- why the tone fits;
- why the skill domain applies;
- for `act`: which observed condition makes the reversible action appropriate;
- for `suggest`: which combination of facts motivates the recommendation, and why no action is taken;
- for `none`: why the observed conditions do not justify intervention;
- if distractor triggers are present, one short clause on why they are not decisive.

Good:
> The driver shows very high fatigue and low attention while driving at highway speed at night. This is an immediate safety concern, so urgency is critical and a serious tone is appropriate. Because the risk concerns the driver's physical condition and attention, the relevant skill is wellbeing. The assistant should recommend rather than control the vehicle, so it suggests without acting, and the suggestion is to take a break at the next safe opportunity.

Bad (label restatement — forbidden):
> Urgency is critical, tone is serious, intervention is suggest, skill is wellbeing, action is none, and suggestion is take_break.

Implementation: use several phrasing templates per decision domain and fill them only from parsed knowledge facts. Do not mention any number, entity or condition absent from the knowledge.

---

## 12. Distribution, splits and duplication

### 12.1 Distribution targets (soft; never override §0.1)

| Aspect | Target |
| --- | --- |
| `intervention_type` | suggest ≈ 50–60%, act ≈ 15–25%, none ≈ 20–30% |
| `urgency` | every label ≥ 8% overall |
| Actions | every non-`none` action present in **every** split, ≥ 30 rows in validation and test each |
| Suggestions | every non-`none` suggestion present in **every** split, ≥ 50 rows in validation and test each |
| Factor coverage | each emotion, activity, road type, road condition, weather, time of day, traffic level and risk level present in every split |
| Negatives | no-intervention rows spread across all domains, not dominated by one template |
| Balance side-effects | actions must not correlate with unrelated facts (e.g. `start_radio` must not co-occur mainly with fog) |

Report the achieved distributions; do not relabel to hit targets.

### 12.2 Split assignment

- Group rows into **scenario families** (same decisive condition + same template skeleton).
- Assign families to splits **before** generation (≈ 80/10/10 of families), stratified by decisive condition so every label appears in every split.
- Generate variations inside each family's split only.
- Exact split sizes: train 40,000; validation 5,000; test 5,000.

### 12.3 Duplication

- No two rows (in any split) may have the same **normalized knowledge set** (sentences sorted).
- No normalized knowledge set may appear in more than one split.
- Near-duplicates across splits: Jaccard similarity of knowledge sets ≥ 0.9 between rows of different splits SHOULD be < 1% of test rows; report the measured value.

---

## 13. Validation (automated; all MUST pass before zipping)

| # | Check | Pass criterion |
| --- | --- | --- |
| 1 | Row count | exactly 50,000 |
| 2 | Split sizes | 40,000 / 5,000 / 5,000 |
| 3 | IDs | unique; pattern matches §9.1 |
| 4 | Questions | exactly 6 keys, correct names, all `type="choice"`, criteria identical to §3 |
| 5 | No score column | no `urgency__score` or numeric urgency anywhere |
| 6 | State shape | exactly `{"knowledge": [...]}`, non-empty list of strings |
| 7 | Knowledge templates | every sentence matches a §6.3 regex; ≤ 1 sentence per field |
| 8 | No raw floats | no decimal numbers in knowledge; intensity/density/categorical fields never numeric |
| 9 | Traffic | only the categorical traffic sentences |
| 10 | Coherence | all §6.4 constraints hold |
| 11 | Probabilities | all labels present, 0 ≤ p ≤ 1, sum = 1 ± 1e-6, target unique max, confidence = max |
| 12 | Flattened columns | equal to `gold` values |
| 13 | Consistency rules | §4.2 holds for every row |
| 14 | Critical safety | every §4.3 situation → critical/suggest/none/serious; no `act` with critical urgency |
| 15 | Label re-derivation | re-parsing knowledge and re-running the labeling function reproduces 100% of gold labels |
| 16 | Evidence | every non-`none` action/suggestion has its §5 required facts in the knowledge |
| 17 | Identifiability | no two rows with identical normalized knowledge have different gold labels |
| 18 | Skill leakage | skill (or scope) not present in `state`, `id`, `factors`, `family_id`; a classifier-free check: no knowledge sentence contains the words "wellbeing", "driving skill", "skill" |
| 19 | Coverage | every action and suggestion in every split (§12.1); every factor value in every split |
| 20 | Duplication | §12.3 criteria |
| 21 | Descriptions | non-empty; 2–5 sentences; reference each of the six decision concepts; contain a causal connector (because/so/therefore/since); every number in the description appears in the knowledge; not a bare label list (fewer than 50% of tokens are label tokens) |
| 22 | Description isolation | description text not present in `state` or `questions` |
| 23 | Files readable | Parquet and all JSONL re-loaded successfully; row counts match |
| 24 | Manifest | SHA-256 and sizes recomputed from the ZIP contents match `manifest.json` |
| 25 | ZIP integrity | `zipfile.testzip()` returns `None` |
| 26 | Determinism | regenerating with the same seed yields identical SHA-256 for data files |

Checks that are inherently semantic (e.g. "no action assigned only for balance") are enforced structurally by #15 and #16: every label is produced by the rule table from knowledge, never assigned directly.

---

## 14. Supporting files

### 14.1 `metadata.json`
`dataset_name`, `version`, `created_at` (ISO 8601), `seed`, `generator_version`, `n_rows`, `split_sizes`, `decision_names`, full `criteria` per decision, `knowledge_templates_version`, `labeling_rules_version`, `reference_urls`, `schema_source` (`reference` or `fallback`), `label_agreement_semantics`, `assumptions` (list).

### 14.2 `statistics.json`
Per split and overall: label counts and percentages for all six decisions; joint `intervention_type × skill`; `action` and `suggestion_type` counts; factor value counts; knowledge length histogram; confidence histogram; duplicate and near-duplicate metrics; the full validation table (§13) with pass/fail and measured values.

### 14.3 `manifest.json`
For every other file in the ZIP: `filename`, `sha256`, `size_bytes`, `row_count` (data files). The manifest does not hash itself.

### 14.4 `README.md`
Dataset card: purpose, file list, schema with a sample row, decision contract, knowledge templates, labeling rule table (§4–§5 as finally implemented), split methodology, achieved distributions, validation summary, known limitations and assumptions, usage with the Laya notebook, and a note that `description` and `factors` must not be used as model inputs.

---

## 15. Final report (return to the user)

After building the ZIP, report:
1. Pass/fail for every §13 check with measured values.
2. Achieved distributions (urgency, intervention, skill, action, suggestion) per split.
3. Any target not met and why.
4. Schema source (reference vs fallback) and any assumptions made.
5. Three sample rows (one `none`, one `act`, one `suggest`) in full.

---

## Appendix A — Known gaps and assumptions

The allowed catalog cannot represent some states. Handle them as follows and list them in `metadata.json → assumptions`:

- **G1 – Air-conditioning state is not observable.** AC actions are grounded only in internal/external temperature ranges (§5.1). Ranges are disjoint to keep labels identifiable.
- **G2 – Navigation and radio state are not observable.** `stop_navigation` is grounded in end-of-trip evidence (stationary, engine off, driver about to exit). `start_navigation` has no strong evidence in the catalog; the default rule is a stationary vehicle with engine on, doors closed, no distracting activity **and** no other trigger, used sparingly and documented. If this is judged insufficiently grounded, report it instead of forcing coverage.
- **G3 – `skill = conversation` is unreachable** under §4.2. It stays in the criteria with non-zero probability mass but never as a target.
- **G4 – Dangerous objects** are catalogued under wellbeing fields but treated as a road hazard (`driving`, `adapt_driving_to_conditions`) because the decisive condition is external.
- **G5 – Templates marked in §6.3** for doors, lights, lock, trunk, engine, turn signal, lane, road condition, time of day, temperatures and emotions were not specified in the original brief and are defined here for consistency. Keep them fixed across all splits.
