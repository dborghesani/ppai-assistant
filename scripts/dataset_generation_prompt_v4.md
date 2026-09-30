# Automotive typed decision dataset generation

Create a synthetic automotive decision-making dataset for fine-tuning Laya.
The dataset must be structurally compatible with the LocalLLaMA/typed-decisions dataset and with the Laya fine-tuning notebook:
- Dataset reference:
https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- Fine-tuning notebook:
https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
Generate exactly 50,000 entries.

## OUTPUT FILES

Create a ZIP archive containing:
- automotive_assistant.parquet
- train.jsonl
- validation.jsonl
- test.jsonl
- metadata.json
- statistics.json
- manifest.json
- README.md
Use the following split:
- train: 40,000 entries
- validation: 5,000 entries
- test: 5,000 entries
Use a deterministic random seed and record it in metadata.json.

## LABELING PRINCIPLE

Every gold label must be produced by one deterministic labeling function of the
serialized knowledge:

```
gold = label(state["knowledge"])
```

This function reads only the knowledge sentences (or the equivalent parsed
facts). It must not read hidden scenario-generation variables, scenario-family
IDs, author intent, or any information omitted from the serialized state.
Because labels are a pure function of the knowledge, two rows with identical
knowledge always receive identical labels, and validation must be able to
re-derive every gold label by re-parsing `state["knowledge"]` and re-running
the same labeling function. Generate the state first, derive the six labels
from it, then compute probabilities, and only then write `description`; none
of these later steps may change an already-decided label.

## TYPED DECISIONS

Every entry must contain exactly six typed decisions. Use these names exactly:
1. urgency
2. tone
3. intervention_type
4. skill
5. action
6. suggestion_type

All six decisions must use `type="choice"`.

The following is the complete self-contained decision contract. Generate only
the dataset records described here.

### URGENCY LABELS

| Label | Description |
| --- | --- |
| `none` | No intervention is needed. |
| `low` | Minor issue or optional assistance. |
| `medium` | Intervention is advisable soon. |
| `high` | Prompt intervention is strongly advisable. |
| `critical` | Immediate attention is required to reduce a serious safety risk. |

### TONE LABELS

| Label | Description |
| --- | --- |
| `calm` | Use a calm and reassuring tone. |
| `enthusiastic` | Use a lively and encouraging tone. |
| `serious` | Use a serious and direct tone. |
| `empathetic` | Use a warm and understanding tone. |
| `discreet` | Use a discreet and non-intrusive tone. |

### INTERVENTION TYPE LABELS

| Label | Description |
| --- | --- |
| `none` | Do not intervene. |
| `suggest` | Communicate a contextual recommendation without directly controlling a function. |
| `act` | Execute one supported reversible vehicle or comfort action. |

### SKILL LABELS

| Label | Description |
| --- | --- |
| `none` | No skill is required. |
| `conversation` | General conversation, social interaction, or dialogue management. |
| `driving` | Driving, navigation, vehicle state, road safety or coaching. |
| `wellbeing` | Fatigue, attention, emotion or comfort. |

### ACTION LABELS

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

### SUGGESTION TYPE LABELS

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

Use the corresponding table above as the complete `criteria` dictionary for
each question. Do not invent labels, rename labels, or change descriptions.
The labels are stable tokens and the descriptions are the natural-language
criteria shown to Laya.

### URGENCY
Define how much urgent the situation is.
Use the five urgency labels in the contract above.
Do not generate an urgency score.

### TONE
Define the communication tone appropriate for the situation.
Use the five tone labels in the contract above.
Tone is a response-style decision, not a voice identity or a TTS model name.

### INTERVENTION TYPE
Decide what the assistant should do. 
Use the three intervention labels in the contract above.

### SKILL
Define a scope for the potential skills the assistant should have to face the situation.
Use the four skill labels in the contract above. The `skill` label must be
inferred from the textual knowledge and the complete scenario.

### ACTION
Defines what the assistant should actuate if an action is required.
Use the action labels in the contract above, including `find_rest_area`.

### SUGGESTION
Define what the assistant should suggest (vocally) to the driver if a suggestion is requred.
Suggestion type represents the semantic intent of a suggestion, not the final driver-facing message.
Use the suggestion labels in the contract above. The selected value is a
semantic intent, not the final driver-facing wording.


### INTERVENTION CONSISTENCY RULES
These rules are mandatory:
If intervention_type is none:
- skill must be none
- action must be none
- suggestion_type must be none
If intervention_type is act:
- action must not be none
- suggestion_type must be none
- skill must be driving or wellbeing
If intervention_type is suggest:
- action must be none
- suggestion_type must not be none
- skill must be driving or wellbeing
Do not create entries that violate these rules.

`urgency=none` if and only if `intervention_type=none`; a minor optional
action or suggestion still carries `urgency=low` or higher. As a direct
consequence of these rules, `skill=conversation` can never be a correct gold
label: it must remain in the `criteria` dictionary with a small non-zero
probability mass, but it must never be the selected target.

### LABEL IDENTIFIABILITY AND EVIDENCE

Derive every target exclusively from the information observable in
`state["knowledge"]`. Do not use hidden scenario-generation variables, scenario
IDs, author intent, or facts omitted from the serialized state.

For each row, check that the knowledge supports the complete set of target
decisions: urgency, tone, intervention_type, skill, action, and
suggestion_type. Each selected action or recommendation must address the
condition evidenced by the knowledge and be appropriate in the full context.
Do not select a label merely to increase its frequency or satisfy a coverage
target.

The intended labels must be identifiable from the serialized knowledge. If
materially different labels would be equally reasonable given the same
knowledge, either add an allowed knowledge fact that distinguishes them or
choose the least specific valid decision. Do not use arbitrary soft-label
probabilities to disguise an ambiguous target.

Apply this test to every domain and signal, including combinations of signals:
for any two examples with the same or nearly identical knowledge, their target
decisions should be the same unless a meaningful knowledge difference explains
the change. Conversely, when one relevant fact changes and that change affects
a decision, the target should reflect it. The state, not an unstated
interpretation of a particular signal, is the complete evidence available to
the model.

### SKILL INFERENCE RULES

The model must infer `skill` from knowledge and must not receive the answer as
an input feature. Do not repeat the intended skill in the state, metadata, or
scenario identifier.

Choose `wellbeing` when the decisive evidence concerns fatigue, attention,
emotion, physical state, driver activity, or driver comfort. Choose `driving`
when the decisive evidence concerns speed, navigation, vehicle state, road
conditions, traffic, visibility, lighting, lane behavior, or driving coaching.
Choose `none` only when no intervention is needed or when the available
knowledge does not support either skill. Shared evidence must be resolved by
the event's decisive condition.

### CRITICAL SAFETY RULE
Do not associate critical safety situations with autonomous safety-critical actions.
Situations such as:
- driver sleeping while the vehicle is moving
- a door or trunk open while driving
- a nearby dangerous object
- severe distraction at high speed
- very high fatigue combined with low attention
- very poor visibility combined with an icy or snowy road
- extreme driving tension combined with speeding
must normally produce:
- urgency: critical
- intervention_type: suggest
- action: none
- an appropriate suggestion_type
ACT must be limited to supported, reversible and contextually justified actions.

### DECISIVE CONDITION PRECEDENCE

A single scenario may trigger more than one candidate rule at once (for
example, high fatigue while the vehicle is also speeding). Resolve every such
case deterministically:

1. Select the triggered candidate with the highest urgency.
2. Break remaining ties using this fixed priority order (highest first):
	 sleeping while moving, open door or trunk while moving, dangerous objects,
	 severe distraction, fatigue and attention, visibility and road condition,
	 speeding, driving tension and emotion, lane and turn-signal behavior,
	 lighting, highway exit preparation, vehicle security while stationary,
	 weather preparation, cabin comfort, navigation and infotainment.
3. Non-decisive triggered facts may remain in the knowledge as distractors,
	 but they must not change the selected labels, and the description must
	 briefly note why they were not decisive.

Use the same priority order, unchanged, across every split and every
generation run.

## KNOWLEDGE REPRESENTATION

The Laya state must contain English textual knowledge, not raw telemetry
objects. The following is the complete allowed knowledge catalog. No other
field may appear in `state["knowledge"]`.

### ALLOWED KNOWLEDGE FIELDS

`driving` fields:
- `VehicleMotion.speed` (numeric km/h, when available)
- `VehicleState.door_open_front_left`
- `VehicleState.door_open_front_right`
- `VehicleState.door_open_rear_left`
- `VehicleState.door_open_rear_right`
- `VehicleState.doors_locked`
- `VehicleState.lights_on_sidelights`
- `VehicleState.lights_on_low_beams`
- `VehicleState.lights_on_high_beams`
- `VehicleState.lights_on_fog_lights`
- `VehicleState.turn_signal`
- `VehicleState.trunk_open`
- `VehicleState.engine_on`
- `VehicleState.internal_temperature` (numeric degrees Celsius)
- `LaneTracing.highway_exit`
- `LaneTracing.lane_crossing_left`
- `LaneTracing.lane_crossing_right`
- `LaneTracing.driving_lane`

`wellbeing` fields:
- `DetectedObjects.people_around`
- `DetectedObjects.vehicles_around`
- `DetectedObjects.dangerous_objects_around`
- `DriverPhysicalState.activity`
- `DriverPhysicalState.attention_level`
- `DriverPhysicalState.fatigue_level`
- `DriverEmotionState.angry`
- `DriverEmotionState.disgust`
- `DriverEmotionState.fear`
- `DriverEmotionState.happy`
- `DriverEmotionState.sad`
- `DriverEmotionState.surprise`
- `DriverEmotionState.neutral`

Shared `driving` and `wellbeing` fields:
- `DriverDrivingStyle.driving_tension`
- `EnvironmentState.external_temperature` (numeric degrees Celsius)
- `EnvironmentState.weather`
- `EnvironmentState.forecast_weather`
- `EnvironmentState.time_of_day`
- `EnvironmentState.road_condition`
- `EnvironmentState.road_type`
- `EnvironmentState.risk_level`
- `EnvironmentState.visibility` (numeric percent only if represented as a physical measurement; otherwise use its bucket)
- `EnvironmentState.traffic`
- `TrafficSigns.speed_limit` (numeric km/h when known)

Render values with the following semantic mapping:
- booleans: use the field-aware states such as open/closed, locked/unlocked,
	on/off, active/inactive;
- counts: use the exact sentences `No people are currently detected around
	the vehicle.`, `People density around the vehicle is <low|medium|high>.`,
	`No vehicles are currently detected around the vehicle.`, `Traffic around
	the vehicle is <low|medium|high>.`, `No dangerous objects are currently
	detected around the vehicle.`, or `Dangerous object density around the
	vehicle is <low|medium|high>.`;
- intensities: use the field-specific sentence and the buckets `low`, `medium`,
	`high`, `very high`, or the exact zero-state sentence. Never expose the
	source float;
- categories: use the exact field-specific sentences listed below;
- ordinary numeric fields: use `Display name is <number> and is <stable|increasing|decreasing>.`
	when a trend is available. Do not add units unless they are explicitly part
	of the required sentence template.

Use these exact field-specific sentence templates:
- highway exit: `The vehicle is not approaching a highway exit.`, `The vehicle
	is approaching a highway exit on the right.`, or `The vehicle is approaching
	a highway exit on the left.`
- traffic: `No traffic is currently detected around the vehicle.`, `Current
	traffic is light.`, `Current traffic is medium.`, or `Current traffic is
	heavy.`
- activity: `No distracting driver activity is detected.`, `The driver is
	about to exit the vehicle.`, `Normal driving activity is detected.`, `The
	driver is talking, which may distract from driving.`, `The driver is eating,
	which distracts from driving.`, `The driver is using a phone, which seriously
	distracts from driving.`, or `The driver appears to be asleep and unable to
	drive safely.`
- weather: `The current weather is sunny/cloudy/rainy/snowy/foggy.`
- forecast: `The weather forecast predicts sunny/cloudy/rainy/snowy/foggy
	conditions.`
- risk: `There is no current risk in this area.` or `The current risk level in
	this area is low/medium/high.`
- road type: `The vehicle is currently driving on an urban/rural/highway/
	residential road.`
- visibility: `Driving visibility is optimal.` or `Driving visibility is
	low/medium/high/very high.`

Raw numeric values must not appear for intensity, density, or categorical
fields. Numeric physical measurements may remain numeric when listed as
numeric fields. The current knowledge wording for these fields does not add
units automatically; reproduce the wording exactly rather than inventing a
different sentence.

Example knowledge:

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

KNOWLEDGE FIELD FILTERING

Use only the fields in the allowed knowledge catalog above. Each field is
already assigned to `driving`, `wellbeing`, or both in that catalog; no source
lookup or external schema is required.

Do not place the intended skill, source field scope, or routing metadata in the
state. The skill must remain an output inference problem.

Use only this catalog. Do not introduce legacy skill names such as
`driver_health` or `navigation_and_coaching`.

## LAYA INPUT CONTRACT

Every `state` value must have exactly this shape:

```json
{
	"knowledge": [
		"Driving visibility is low.",
		"The current weather is foggy."
	]
}
```

Do not include an extra `event` key, raw telemetry object, dataclass name, field
path, intended skill, or routing metadata in the state.

Every `questions` value must contain exactly these six keys:

```text
urgency
tone
intervention_type
skill
action
suggestion_type
```

Each question must use `type="choice"` and the complete criteria table from
this prompt. The question names, labels, descriptions, and state shape must be
identical in every split.

AUTOMOTIVE SCENARIOS

Generate realistic, diverse and internally consistent combinations covering:

1. Driver emotions

- angry
- disgust
- fear
- happy
- sad
- surprise
- neutral

Include different semantic intensities and negative examples where an emotion does not require intervention.

2. In-cabin activity

- idle
- talking
- driving
- eating
- on the phone
- about to exit
- sleeping

The significance of the activity must depend on whether the vehicle is stationary or moving, vehicle speed, traffic and road type.

3. Driving tension

- low
- medium
- high
- very high

Treat driving tension as nervous, tense or aggressive driving behavior, not physical violence.

4. Fatigue and attention

Combine fatigue and attention with:

- time of day
- road type
- traffic
- speed
- visibility
- driver activity

5. Traffic

- `No traffic`
- `Light`
- `Medium`
- `Heavy`

6. Visibility

Generate the exact knowledge buckets `optimal`, `low`, `medium`, `high`, and
`very high`.

7. Driving coaching

Include:

- lane changes with and without the correct turn signal
- inappropriate or missing lighting
- unnecessary high beams
- speed below, at and above the verified speed limit
- unknown speed limit cases
- driving style not adapted to road or traffic conditions

8. Road conditions

- Dry
- Wet
- Icy
- Snowy
- Gravel

9. Current and forecast weather

- Sunny
- Cloudy
- Rainy
- Snowy
- Foggy

Include contextual suggestions such as preparing for forecast rain when the driver is about to exit the vehicle.

10. Time of day

- Morning
- Afternoon
- Evening
- Night

Combine time of day with fatigue, attention, visibility and lighting.

11. Road type

- Urban
- Rural
- Highway
- Residential

12. Area risk

- no risk
- low
- medium
- high risk

13. Vehicle security and state

Include:

- locked and unlocked doors
- open and closed individual doors
- open and closed trunk
- vehicle stationary versus moving
- driver about to exit
- area risk

14. Safety

Include:

- dangerous objects
- door or trunk open while moving
- severe distraction
- sleeping while moving
- dangerous combinations of weather, visibility, speed and road condition

15. Cabin comfort

Combine:

- cabin temperature
- external temperature
- air-conditioning state
- driver wellbeing
- activity and vehicle state

16. Navigation and infotainment

Include contextually valid situations for starting or stopping navigation and radio.

## STAKEHOLDER REFERENCE USE CASES

The following use cases were supplied by stakeholders as illustrative examples
of what the assistant should eventually be able to handle. Treat them as
general inspiration for scenario coverage, not as an exhaustive list and not as
a template to generate cases around exclusively: do not let these examples
dominate the dataset, and do not restrict scenario generation to only the
situations described below. The full scenario coverage in the sections above
still applies. That said, the generated dataset should, taken as a whole,
plausibly support every use case listed here; if a use case cannot be
represented with the allowed knowledge catalog and labeling contract as
written, record it as a known gap (see "ASSUMPTIONS AND KNOWN GAPS") rather
than inventing a new knowledge field or an unsupported label to force it in.

Driver health skill:

- if the internal temperature is too high or too low, activate the vehicle
	temperature control accordingly
- if some dangerous objects are detected around, warn the driver (and lock the
	doors)
- suggest the driver rest if the fatigue level is too high or the attention
	level is too low, especially at night or in heavy traffic
- alert the driver if they are doing something dangerous while driving (e.g.
	talking, using the phone, eating, sleeping)
- react to the driver's emotional state: neutral is the default for safe
	driving, but strong sadness, anger or anxiety can trigger the assistant to
	check in with the driver and, if appropriate, offer some relaxing
	background audio

Navigation and driving coaching skill:

- if the driving style is too aggressive, warn the driver
- if the vehicle is turning left or right and the corresponding turn signal is
	not active (or, conversely, if the vehicle has been going straight for a
	while and a turn signal is still active), suggest using the turn signal
- if the vehicle is not in the correct lane (for example the right-most lane
	on a highway, or the correct lane for an upcoming exit), suggest moving to
	the correct lane
- if the vehicle is going faster than the known speed limit for the current
	location, warn the driver and suggest slowing down
- if the doors are unlocked while driving through a dangerous area, warn the
	driver and lock the doors
- if the trunk is open while the vehicle is moving, warn the driver
- warn the driver about particularly dangerous road conditions
- turn on the fog lights when fog is detected and visibility is low
- warn the driver if the low beam headlights are off in the evening or at
	night
- when the vehicle stops, the driver is about to exit, and the forecast
	weather is rainy, suggest bringing an umbrella
- remind the driver to lock the vehicle's doors when about to exit
- when the road presents an elevated risk (for example from accident history
	or mortality rate), warn the driver

Only represent these situations using the allowed knowledge catalog above:
for example, "last known speed limit" and "area risk" must be rendered through
`TrafficSigns.speed_limit` and `EnvironmentState.risk_level` respectively, not
through a new GPS, map, or accident-history field.

MULTI-CONDITION SUGGESTIONS

Suggestions must predominantly depend on multiple contextual conditions.

Avoid simplistic mappings such as:

fatigue -> take_break

Prefer:

very high fatigue
+ low attention
+ Night
+ Highway
+ high traffic
-> critical
-> suggest
-> wellbeing
-> take_break

Generate boundary cases and contrastive examples where changing one relevant condition changes urgency or intervention.

For every contrastive pair, make sure the changed knowledge fact is sufficient
to explain the changed target. Keep unrelated decisions the same unless the
changed fact logically affects them. Do not generate near-duplicate rows with
different gold labels unless the knowledge difference explains the label
difference. Apply this consistently to every signal and domain, not only to
the examples listed above.

Examples:

- high fatigue on a stationary vehicle should not be treated like high fatigue at highway speed
- sadness at medium intensity with good attention may be low urgency
- anger combined with high driving tension and heavy traffic may be high urgency
- fog lights should not be enabled only because the weather label is Foggy if visibility is still good
- an unlocked vehicle in a no-risk area while driving should not always trigger the same decision as an unlocked stationary vehicle in a high-risk area

### DEBUG DESCRIPTION

Every dataset row must contain a top-level `description` field.

`description` is provided exclusively for debugging, inspection, and dataset
quality analysis. It is not part of the Laya input and must never be included
in `state`, `questions`, or any model input during training or inference.

The description must briefly explain, in natural English, why the observable
knowledge supports the selected decisions.

It must explicitly justify all six decisions:

- urgency
- tone
- intervention_type
- skill
- action
- suggestion_type

The explanation must be derived exclusively from facts observable in
`state["knowledge"]` and from the decision contract. Do not use hidden
generation variables, scenario identifiers, latent factors, intended labels,
or information that is not present in the serialized knowledge.

The description should explain the causal reasoning behind the decisions,
rather than merely restating the selected labels.

For example:

"The driver shows very high fatigue and low attention while driving at highway
speed at night. This represents an immediate safety concern, so urgency is
critical and a serious tone is appropriate. Because the risk concerns the
driver's physical condition and attention, the relevant skill is wellbeing.
The assistant should suggest rather than autonomously control the vehicle, so
intervention_type is suggest and action is none. The appropriate suggestion is
take_break because the driver should stop at the next safe opportunity."

Avoid descriptions such as:

"Urgency is critical, tone is serious, intervention is suggest, skill is
wellbeing, action is none, and suggestion is take_break."

Such descriptions only repeat the labels and do not explain why they follow
from the knowledge.

For `none` decisions, explain briefly why the observed conditions do not
justify intervention. For example:

"The driver has good attention, low fatigue, and no relevant safety or comfort
issue is present. No intervention is therefore warranted, so urgency,
intervention, skill, action, and suggestion remain none. A discreet tone is
appropriate because no active communication is required."

For `act` decisions, explicitly identify the observable condition that makes
the selected reversible action appropriate.

For `suggest` decisions, explicitly identify the combination of observed
conditions that motivates the recommendation.

The description must remain concise, preferably 2-5 sentences.

`description` is diagnostic metadata only. It must not be used as a training
feature, must not influence the state representation, and must not be included
when constructing Laya training sequences.

Generate description only after state["knowledge"] and all six gold decisions have been finalized. The description must explain an already evidence-grounded decision and must never be used to determine, modify, or validate the target labels themselves.

## ASSUMPTIONS AND KNOWN GAPS

The allowed knowledge catalog cannot represent every state that a real action
might depend on. Resolve the following gaps as described here, and record them
verbatim in `metadata.json → assumptions`; do not silently invent a knowledge
field or fabricate a scenario to avoid reporting a gap:

- Air-conditioning on/off state is not observable. Ground
	`enable_air_conditioning` / `disable_air_conditioning` only in disjoint
	internal- and external-temperature ranges, never in driver mood, activity,
	or any other unrelated signal.
- Radio and navigation on/off state are not observable. Ground
	`stop_navigation` in end-of-trip evidence (stationary, engine off, driver
	about to exit) and `start_navigation` in a stationary, ready vehicle with no
	other decisive trigger (engine on, doors locked, trunk closed, no
	distracting activity). Ground `start_radio` / `stop_radio` only in the
	explicit combination of attention, traffic and emotion described in the
	suggestion and action rules above, never in temperature, location, or
	fatigue alone.
- As a consequence of the intervention consistency rules, `skill=conversation`
	is unreachable as a gold label; keep it in the criteria dictionary only.
- Dangerous objects are catalogued as a `wellbeing` source field but are
	labeled as a `driving` hazard (`adapt_driving_to_conditions`), because the
	decisive condition is external to the driver, not a wellbeing concern.
- Where this prompt defines a sentence template, threshold, or priority order
	that was not explicit in the original brief, treat it as authoritative and
	keep it fixed and identical across every split and every generation run.

Each row must contain at least:
 
- id
- workflow
- split
- state
- questions
- gold
- factors
- description
- n_questions
- label_agreement

And flattened columns:

- urgency__label
- urgency__confidence
- urgency__probabilities
- tone__label
- tone__confidence
- tone__probabilities
- intervention_type__label
- intervention_type__confidence
- intervention_type__probabilities
- skill__label
- skill__confidence
- skill__probabilities
- action__label
- action__confidence
- action__probabilities
- suggestion_type__label
- suggestion_type__confidence
- suggestion_type__probabilities

Set:

- workflow = automotive_assistant
- n_questions = 6

Store state, questions, gold, factors and probability dictionaries in the same representation expected by the typed-decisions fine-tuning pipeline.

PROBABILITIES

Every choice decision must include a complete probability distribution across all its allowed labels.

For every probability distribution:

- include every allowed label
- use values between 0 and 1
- ensure the values sum to 1 within floating-point tolerance
- ensure the target label has the highest probability
- ensure confidence equals the highest probability

Do not use one-hot probabilities exclusively. Use plausible soft labels with limited uncertainty around the selected class.

DISTRIBUTION

Aim for broad urgency coverage, but do not force exact class counts by
mislabeling or weakening scenario evidence. Safety and evidence-grounding take
priority over a perfectly balanced histogram. Report the achieved class
distribution.

Avoid excessive duplication.

Ensure that:

- action and suggestion labels are distributed across splits when supported by
	valid, explicit knowledge; never fabricate a scenario to satisfy label
	coverage
- each relevant emotion, activity, road type, road condition, weather, time of day, traffic level and area-risk level is represented in every split
- suggestion examples substantially outnumber direct ACT examples
- negative and no-intervention examples are diverse
- every non-`none` action and suggestion has traceable supporting knowledge
- actions are not correlated with unrelated event types due to balancing

SPLIT QUALITY

Do not randomly split near-duplicate generated records.

Assign scenario templates or scenario families to splits first, then generate their variations.

Avoid placing identical or near-identical scenarios in different splits.

VALIDATION

Before creating the ZIP, perform and report automated validation for:

- exactly 50,000 rows
- exact split sizes
- unique IDs
- six questions per row
- urgency always type=choice
- no urgency__score column
- full probability distributions
- probability sums equal to 1 within tolerance
- target label has maximum probability
- intervention/action/suggestion consistency
- every critical safety situation yields urgency=critical, intervention_type=suggest, action=none and a serious tone; no `act` co-occurs with urgency=critical
- gold labels are 100% re-derivable by re-parsing state["knowledge"] and re-running the same labeling function
- identical or normalized-equal knowledge never yields different gold labels across rows
- every non-`none` action and suggestion is supported by explicit knowledge
- no action is assigned solely to satisfy class-balance or coverage targets
- ambiguous scenarios use `none` rather than arbitrary specific labels
- skill labels are inferable from knowledge without an explicit scope feature; no knowledge sentence contains the words "skill" or "wellbeing"
- no raw float perception values in textual knowledge
- categorical traffic values only
- all required action labels represented in every split
- all required suggestion labels represented in every split
- all Parquet and JSONL files readable
- ZIP integrity
- SHA-256 manifest correctness
- regenerating with the same seed reproduces byte-identical data files
- cross-split near-duplicate rate (Jaccard similarity of knowledge sets ≥ 0.9) stays below 1% of test rows
- every row contains a non-empty description
- every description explains all six selected decisions
- descriptions are grounded exclusively in state["knowledge"]
- descriptions do not introduce facts absent from state["knowledge"]
- descriptions explain the rationale rather than merely restating labels
- description is never embedded in state or questions
