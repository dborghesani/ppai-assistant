# Automotive typed-decision dataset generation

## Objective and source of truth

Generate a **synthetic English automotive decision dataset** for fine-tuning Laya, structurally compatible with the [LocalLLaMA typed-decisions dataset](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) and the [Laya fine-tuning notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb).

The intended deliverable contains **exactly 50,000 rows**. Treat the decision contract and the allowed knowledge catalog below as authoritative. Never satisfy a size, split, or coverage requirement by inventing facts, mislabeling a case, or leaking a target into model input. **If the contract is infeasible as written, report the precise blockers rather than claiming a valid dataset.**

### Deliverable

Create one ZIP archive containing exactly:

```text
automotive_assistant.parquet
train.jsonl
validation.jsonl
test.jsonl
metadata.json
statistics.json
manifest.json
README.md
```

| Split | Rows |
| --- | ---: |
| train | 40,000 |
| validation | 5,000 |
| test | 5,000 |

Use a deterministic seed, record it in `metadata.json`, and ensure the Parquet file contains all three splits. The JSONL files must contain the corresponding rows, with the same schema and values. Report achieved distributions, not intended distributions.

## 1. Typed-decision contract

Every row contains **exactly six questions**, in this order: `urgency`, `tone`, `intervention_type`, `skill`, `action`, `suggestion_type`. Every question has `type: "choice"`, an English `instructions` string, and a `criteria` object containing **all and only** the labels and exact descriptions below. Preserve insertion order consistently across rows and splits; the notebook uses the order of `criteria` keys to construct target vectors.

### Urgency

| Label | Exact criterion |
| --- | --- |
| `none` | No intervention is needed. |
| `low` | Minor issue or optional assistance. |
| `medium` | Intervention is advisable soon. |
| `high` | Prompt intervention is strongly advisable. |
| `critical` | Immediate attention is required to reduce a serious safety risk. |

Urgency is a choice, **not a score**. If no intervention is warranted, use `urgency=none`. Do not create `urgency__score`.

### Tone

| Label | Exact criterion |
| --- | --- |
| `calm` | Use a calm and reassuring tone. |
| `enthusiastic` | Use a lively and encouraging tone. |
| `serious` | Use a serious and direct tone. |
| `empathetic` | Use a warm and understanding tone. |
| `discreet` | Use a discreet and non-intrusive tone. |

Tone describes response style, not a voice identity or TTS model. For no intervention, `discreet` is generally appropriate; do not infer an emotional state from silence.

### Intervention type

| Label | Exact criterion |
| --- | --- |
| `none` | Do not intervene. |
| `suggest` | Communicate a contextual recommendation without directly controlling a function. |
| `act` | Execute one supported reversible vehicle or comfort action. |

### Skill

| Label | Exact criterion |
| --- | --- |
| `none` | No skill is required. |
| `conversation` | General conversation, social interaction, or dialogue management. |
| `driving` | Driving, navigation, vehicle state, road safety or coaching. |
| `wellbeing` | Fatigue, attention, emotion or comfort. |

Infer `skill` from the decisive observable condition. Choose `wellbeing` for fatigue, attention, emotion, physical state, driver activity, or comfort; choose `driving` for speed, navigation, vehicle state, road conditions, traffic, visibility, lighting, lane behavior, or coaching. Shared evidence is resolved by the condition that actually motivates the intervention. Under the consistency rules below, `conversation` is **not a reachable gold label**. Do not force it into the dataset.

### Action

| Label | Exact criterion |
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

An action must address an observed need, be reversible and contextually appropriate, and have its **preconditions evidenced in knowledge**. Never infer the current state of radio, navigation, or air conditioning from temperature, fatigue, location, or other unrelated signals. Do not start an infotainment service merely because a driver appears happy. `find_rest_area` is an assistance action, not autonomous control of the vehicle.

### Suggestion type

| Label | Exact criterion |
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

`suggestion_type` is semantic intent, not final driver-facing wording. Do not invent labels or change the descriptions.

### Mandatory cross-decision constraints

| `intervention_type` | `skill` | `action` | `suggestion_type` |
| --- | --- | --- | --- |
| `none` | `none` | `none` | `none` |
| `act` | `driving` or `wellbeing` | non-`none` | `none` |
| `suggest` | `driving` or `wellbeing` | `none` | non-`none` |

Use `urgency=none` only when `intervention_type=none`. A minor optional action still has `urgency=low`. If the input is ambiguous, do not invent a specific action or suggestion. Where no intervention is justified, use the complete `none` combination; where a concern is clear but its remedy is not, include only a justified, sufficiently general suggestion if the contract supports it.

### Safety priority

The following situations normally require `urgency=critical`, `intervention_type=suggest`, `action=none`, and a contextually appropriate suggestion: sleeping while moving; an open door or trunk while moving; a dangerous nearby object; severe distraction at high speed; very high fatigue with low attention while moving; very poor visibility with icy/snowy road conditions; extreme driving tension combined with speeding. Do not assign an autonomous safety-critical action in these cases. A reversible convenience action must not override a more urgent safety recommendation.

## 2. Observable state and knowledge catalog

Every `state` is a JSON object with **exactly one key**, `knowledge`, whose value is an array of English sentences:

```json
{
  "knowledge": [
    "Driving visibility is low.",
    "The current weather is foggy."
  ]
}
```

Do not put `event`, raw telemetry, field paths, source scope, scenario family, routing hints, `description`, or the intended `skill` in `state`. Every target must be identifiable from these sentences alone. No field outside this catalog may be used to generate knowledge.

### Driving-only source fields

```text
VehicleMotion.speed
VehicleState.door_open_front_left
VehicleState.door_open_front_right
VehicleState.door_open_rear_left
VehicleState.door_open_rear_right
VehicleState.doors_locked
VehicleState.lights_on_sidelights
VehicleState.lights_on_low_beams
VehicleState.lights_on_high_beams
VehicleState.lights_on_fog_lights
VehicleState.turn_signal
VehicleState.trunk_open
VehicleState.engine_on
VehicleState.internal_temperature
LaneTracing.highway_exit
LaneTracing.lane_crossing_left
LaneTracing.lane_crossing_right
LaneTracing.driving_lane
```

`VehicleMotion.speed` is numeric km/h when available; `VehicleState.internal_temperature` is a numeric physical measurement in degrees Celsius.

### Wellbeing-only source fields

```text
DetectedObjects.people_around
DetectedObjects.vehicles_around
DetectedObjects.dangerous_objects_around
DriverPhysicalState.activity
DriverPhysicalState.attention_level
DriverPhysicalState.fatigue_level
DriverEmotionState.angry
DriverEmotionState.disgust
DriverEmotionState.fear
DriverEmotionState.happy
DriverEmotionState.sad
DriverEmotionState.surprise
DriverEmotionState.neutral
```

### Shared source fields

```text
DriverDrivingStyle.driving_tension
EnvironmentState.external_temperature
EnvironmentState.weather
EnvironmentState.forecast_weather
EnvironmentState.time_of_day
EnvironmentState.road_condition
EnvironmentState.road_type
EnvironmentState.risk_level
EnvironmentState.visibility
EnvironmentState.traffic
TrafficSigns.speed_limit
```

`EnvironmentState.external_temperature` and known `TrafficSigns.speed_limit` are numeric physical measurements. `EnvironmentState.visibility` may be numeric only when it is a physical percent measurement; otherwise use its categorical bucket.

### Knowledge rendering

- **Booleans:** render field-aware states such as open/closed, locked/unlocked, on/off, active/inactive. Make the relevant component explicit, for example, a specific door or light.
- **Counts:** use exactly the applicable sentence below. Do not expose a source count or float.
- **Intensities:** use field-specific wording and the buckets `low`, `medium`, `high`, `very high`, or an exact zero-state sentence. Never expose the source float.
- **Categories:** use the field-specific sentence forms below, with one allowed category substituted.
- **Physical numbers:** use `Display name is <number> and is <stable|increasing|decreasing>.` when a trend is available. Do not add units unless the required template explicitly includes them. The known speed limit may be rendered as `The speed limit is <number>.` Do not fabricate a speed limit when unknown.

**Exact count sentences:**

```text
No people are currently detected around the vehicle.
People density around the vehicle is <low|medium|high>.
No vehicles are currently detected around the vehicle.
Traffic around the vehicle is <low|medium|high>.
No dangerous objects are currently detected around the vehicle.
Dangerous object density around the vehicle is <low|medium|high>.
```

**Exact category sentences:**

```text
The vehicle is not approaching a highway exit.
The vehicle is approaching a highway exit on the right.
The vehicle is approaching a highway exit on the left.

No traffic is currently detected around the vehicle.
Current traffic is light.
Current traffic is medium.
Current traffic is heavy.

No distracting driver activity is detected.
The driver is about to exit the vehicle.
Normal driving activity is detected.
The driver is talking, which may distract from driving.
The driver is eating, which distracts from driving.
The driver is using a phone, which seriously distracts from driving.
The driver appears to be asleep and unable to drive safely.

The current weather is <sunny|cloudy|rainy|snowy|foggy>.
The weather forecast predicts <sunny|cloudy|rainy|snowy|foggy> conditions.
There is no current risk in this area.
The current risk level in this area is <low|medium|high>.
The vehicle is currently driving on an urban road.
The vehicle is currently driving on a rural road.
The vehicle is currently driving on a highway road.
The vehicle is currently driving on a residential road.
Driving visibility is optimal.
Driving visibility is <low|medium|high|very high>.
```

A placeholder denotes a choice of **one** listed token; do not print angle brackets or slash-separated alternatives in a row. Preserve category capitalization where the supplied examples require it, such as `The time of day is Night.` and `The road condition is Icy.` Avoid contradictory facts about the same field and avoid implying motion from road type alone; use speed or another unambiguous motion observation.

Example:

```json
[
  "Speed is 90 and is stable.",
  "The speed limit is 70.",
  "The driving style shows a high level of tension.",
  "Attention level is low.",
  "Fatigue level is very high.",
  "Current traffic is heavy.",
  "The vehicle is currently driving on a highway road.",
  "The time of day is Night."
]
```

## 3. Evidence-first generation

Generate a coherent state first, then decide the six labels **from serialized knowledge**, then construct probabilities, and **only afterward** write the diagnostic description. Hidden generator variables, template names, split IDs, author intent, and metadata must never affect a label unless the relevant fact appears in `state["knowledge"]`.

For each candidate row:

1. Check internal consistency, observability, and the safety priority.
2. Identify the decisive condition and all relevant competing conditions.
3. Select `urgency`, `tone`, `intervention_type`, `skill`, `action`, and `suggestion_type` from evidence alone.
4. Check whether another materially different decision is equally plausible. If so, add a **permitted and genuinely observed** discriminating fact, select the least specific valid decision, or reject the row.
5. Generate evidence-informed soft probabilities **after** labels are fixed. Soft labels must not conceal ambiguity.
6. Write `description` only after state and gold decisions are final. Do not use it to choose or validate labels.

When multiple problems coexist, prioritize the most urgent supported intervention. Do not add random weather, emotion, activity, risk, or road facts to a finished case if they would change its safety assessment. Contrastive pairs should change one meaningful fact, change only decisions justified by that fact, and keep all unrelated decisions stable. Identical or semantically equivalent knowledge must not receive contradictory gold labels.

### Required scenario coverage

Cover all of the following **in each split where feasible under evidence and safety rules**:

- Emotions: angry, disgust, fear, happy, sad, surprise, neutral; multiple intensities and benign counterexamples.
- Activity: idle, talking, driving, eating, phone use, about to exit, sleeping; stationary versus moving, speed, traffic, and road type must matter.
- Driving tension: low, medium, high, very high; interpret as nervous/tense/aggressive driving, not physical violence.
- Fatigue and attention crossed with time, road type, traffic, speed, visibility, and activity.
- Traffic: none, light, medium, heavy. Visibility: optimal, low, medium, high, very high.
- Coaching: lane changes with/without the correct signal; appropriate and inappropriate lighting; unnecessary high beams; speeds below/at/above a **known** limit and unknown-limit negatives; driving style in context.
- Road conditions: Dry, Wet, Icy, Snowy, Gravel. Current and forecast weather: Sunny, Cloudy, Rainy, Snowy, Foggy.
- Time: Morning, Afternoon, Evening, Night. Road: Urban, Rural, Highway, Residential. Area risk: none, low, medium, high.
- Vehicle state: individual doors and trunk open/closed, doors locked/unlocked, stationary/moving, exiting, area risk.
- Safety: dangerous objects, open door/trunk while moving, severe distraction, sleeping while moving, hazardous weather/visibility/speed/road combinations.
- Cabin comfort: cabin and external temperatures combined with driver state and activity. Do not assert air-conditioning status unless the catalog is amended.
- Navigation and infotainment: only actions whose necessary state and need are actually observable; otherwise mark their coverage as blocked.

Suggestions should **predominantly depend on multiple relevant facts**, not one-to-one label shortcuts. Examples of intended contrasts: high fatigue while parked versus at highway speed; medium sadness with good attention versus sadness with impaired attention; anger plus tension and heavy traffic; fog with good visibility versus fog with low visibility; unlocked doors while moving in a low-risk area versus stationary in a high-risk area. These are design goals, not automatic labels.

## 4. Row schema and model-input boundary

Each row contains at least:

```text
id, workflow, split, state, questions, gold, factors, description,
n_questions, label_agreement,
urgency__label, urgency__confidence, urgency__probabilities,
tone__label, tone__confidence, tone__probabilities,
intervention_type__label, intervention_type__confidence, intervention_type__probabilities,
skill__label, skill__confidence, skill__probabilities,
action__label, action__confidence, action__probabilities,
suggestion_type__label, suggestion_type__confidence, suggestion_type__probabilities
```

Set `workflow="automotive_assistant"` and `n_questions=6`. Match the reference pipeline's **JSON-encoded string** representation for `state`, `questions`, `gold`, `factors`, `label_agreement`, and each flattened `__probabilities` column in Parquet; use the same representation in JSONL. `__confidence` is numeric. `questions` has precisely the six named keys, each with `type`, `instructions`, and the complete ordered `criteria` dictionary. `gold` has exactly the same six keys, each with `type="choice"`, `label`, `confidence`, and `probabilities`.

`factors` and `label_agreement` are **not model inputs**. Do not insert intended skill or a trivial answer code into a scenario identifier or factors. Do not fabricate teacher-sample agreement: if no independent teacher samples were collected, set `label_agreement` to JSON `null`, document why in metadata, and verify downstream consumers accept this value; otherwise use genuine computed agreement. The notebook's training preprocessing reads `state`, `questions`, and `gold`; `description` must not be added to any of them or passed to `build_sequence`.

### Probability contract

For each question:

```text
keys(probabilities) == keys(questions[q].criteria)
0 <= p[label] <= 1 for every label
abs(sum(p.values()) - 1) <= 1e-6
argmax(probabilities) == gold.label
confidence == max(probabilities.values())
```

Use a unique argmax; no all-one-hot dataset. Allocate limited, evidence-related uncertainty to plausible alternatives, not arbitrary random noise or unsupported classes. The flattened label, confidence, and distribution must exactly match `gold`. If distributions are generated by a heuristic rather than calibrated teacher judgments, disclose this in metadata and do not describe them as calibrated probabilities.

## 5. Split design and validation

Assign **scenario families and their contrastive variants to one split before generating rows**. If the same label must appear in multiple splits, create independently designed, meaningfully different families for each split; do not clone a family and merely change an irrelevant contextual fact. Normalize sentence order and compare both exact knowledge sets and semantic signatures across splits. Keep contrastive pairs within a split. Reject duplicates and near-duplicates rather than padding to 50,000.

Aim for broad, evidence-grounded urgency coverage, diverse negatives, and substantially more `suggest` than `act`. Do not force label counts. Coverage is a **reporting goal subordinate to validity**. Report unachievable labels as blockers, not as successful coverage.

Before packaging, perform and record automated checks for:

1. Exactly 50,000 rows; exact split sizes; unique IDs; identical schema and content across Parquet and corresponding JSONL; all files readable.
2. Exactly six choice questions and complete unchanged ordered criteria in every row; no `urgency__score`.
3. Complete probability keys, numeric bounds, sum tolerance, unique gold argmax, confidence equality, and agreement between nested and flattened gold.
4. All cross-decision rules and safety constraints; `urgency=none` only for no intervention.
5. Knowledge restricted to the catalog and allowed sentence forms; no raw intensity/density floats; categorical traffic only; no contradictory observations.
6. Evidence for **every** non-`none` action and suggestion, including necessary action preconditions; no action selected to satisfy coverage. Flag ambiguous labels rather than silently accepting them.
7. Skill inferred from the decisive knowledge, never from an explicit input scope. No answer leakage via state, IDs, metadata, or training inputs.
8. Split-level scenario coverage and action/suggestion coverage **where actually feasible**; suggestion-to-act ratio; achieved label histograms; cross-split exact and semantic near-duplicate rates.
9. Non-empty 2–5 sentence descriptions explaining the observed evidence, urgency, tone, intervention, skill, action, and suggestion, including why an unused action/suggestion is `none`. No unsupported facts or label-only boilerplate. This requires both deterministic checks and **manual review of representative samples**; do not claim automated proof of causal grounding from string matching.
10. ZIP integrity and SHA-256 hashes for each included payload. Define whether `manifest.json` excludes itself from the manifest and verify accordingly.

Write actual outcomes, failures, warnings, distribution counts, duplicate metrics, and sampling method into `statistics.json`/`README.md`. Do not set a validation flag to `true` unless the corresponding check was performed and passed. If any mandatory check fails, **do not publish the ZIP as compliant**; report the blocker.

## 6. Known feasibility blockers to resolve before generation

The current allowed knowledge catalog has **no air-conditioning on/off state, radio on/off state, or navigation active/inactive state**. It also has no explicit user request for starting/stopping these services. Consequently, `enable_air_conditioning`, `disable_air_conditioning`, `start_radio`, `stop_radio`, `start_navigation`, and `stop_navigation` cannot generally be justified as uniquely identifiable direct actions from the permitted knowledge alone. Cabin temperature does not establish AC state; an approaching exit does not establish navigation state; driver emotion does not authorize radio playback.

The simultaneous requirements **“all required action labels represented in every split”** and **“every action grounded exclusively in allowed knowledge”** therefore cannot both be guaranteed without changing the contract. Do **not** manufacture those actions or silently add forbidden knowledge. Request an explicit contract revision that either (a) adds observable service-status and user-intent fields with sentence templates, or (b) exempts unidentifiable actions from per-split coverage. The same principle applies to any other action whose necessary precondition is not observable.

Likewise, `conversation` is listed as a skill but prohibited by all three intervention consistency branches. It must remain an available criterion but cannot appear as a valid target unless the consistency contract is changed. These blockers take precedence over the 50,000-row delivery target.