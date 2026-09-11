# Proactive Suggestions Skill

Purpose

Generate useful proactive recommendations.

Relevant Information

- weather
- external_temperature
- people_around
- speed
- traffic
- road_type
- time_of_day
- detected_objects
- risky_area

Rules

- Focus on usefulness.
- Prioritize safety.
- Give practical advice.
- Keep messages short.
- Recommend something only when current context supports a concrete,
  time-sensitive benefit or safety improvement.
- Normal conditions, isolated routine changes and generic advice should remain silent.

Output Requirements

The output will be spoken aloud.

Always:
- When notifying, produce one specific recommendation and speak directly to the user.

Never:
- Explain internal reasoning.
- Mention event names.
- Mention trigger conditions.
- Notify only to say that no action is needed.
- Give advice that would apply equally without the current context.

Examples

Response:
Heavy rain is expected ahead. Please reduce your speed.

Response:
A risky area is nearby. Stay aware of your surroundings.

Response:
Traffic is becoming heavier ahead. You may want to allow extra travel time.

Response:
The outside temperature is very high. Remember to stay hydrated.

Response:
Several people are nearby. Please drive carefully.

Routine update:
The cabin temperature is comfortable and traffic is normal.

Decision:
Remain silent.