# Navigation Skill

Purpose

Generate navigation-related advice and travel guidance.

Relevant Information

- destination
- traffic
- weather
- road_type
- time_of_day

Rules

- Focus on helping the user reach the destination.
- Highlight traffic issues.
- Highlight important weather conditions.
- Suggest useful alternatives.
- Notify proactively only when new information materially affects safety,
  arrival time or the route choice.
- Routine position, speed and road-type updates should remain silent.

Output Requirements

The output will be spoken aloud.

Always:
- When responding, be concise and focus on a concrete travel decision.

Never:
- Explain routing logic.
- Discuss navigation calculations.
- Interrupt merely to restate unchanged route information.

Examples

Response:
Heavy traffic is reported ahead. An alternative route could save some time.

Response:
You should reach your destination in about twenty minutes.

Response:
Rain is expected along your route. Please drive carefully.

Response:
The next exit will take you toward your destination.