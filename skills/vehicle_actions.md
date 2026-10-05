# Vehicle Actions

Use this catalog to map a driver's direct request to one supported `ActionType`.
This reference describes the assistant's simulated vehicle controls, not physical
vehicle capabilities. Select only actions present in `action_options`.

## Climate and airflow

- `enable_heating` / `disable_heating`: toggle cabin heating. "Warmer", "warm
  the cabin" and "raise the temperature" mean enable heating. "Turn off the
  heating" means disable heating, not disable seat heating.
- `increase_fan_speed` / `decrease_fan_speed`: change airflow intensity by one
  level, clamped to levels 0-7 (default 3). "More/less airflow", "fan up/down"
  and "blow harder/softer" refer to this action, not cabin temperature.
- `enable_air_conditioning` / `disable_air_conditioning`: toggle AC on/off. This
  is separate from fan speed. "Cooler", "cool the cabin" and "lower the
  temperature" mean enable air conditioning. Neither AC nor heating changes
  the measured cabin temperature or the driver's preferred temperature.
- `enable_air_recirculation` / `disable_air_recirculation`: toggle cabin air
  recirculation.
- `enable_seat_heating` / `disable_seat_heating`: toggle seat heating.

For relative increases or decreases, use one step and do not ask for a target.
There is no action for setting an exact temperature, fan level or volume value.

## Windows, roof and access

- `open_windows` / `close_windows`: open or close all four simulated windows.
- `open_sunroof` / `close_sunroof`: toggle the simulated sunroof.
- `lock_doors` / `unlock_doors`: toggle the simulated door-lock state.

These are binary UI states; partial window positions and individual-window
commands are not supported.

## Audio and navigation

- `start_radio` / `stop_radio`: toggle radio power.
- `increase_audio_volume` / `decrease_audio_volume`: adjust volume by 5 points,
  clamped to 0-100 (default 20).
- `start_navigation` / `stop_navigation`: toggle the navigation-active indicator.
  This does not select a destination or calculate a route.

## Lighting

- `enable_sidelights` / `disable_sidelights`: toggle sidelights.
- `enable_low_beam_headlights` / `disable_low_beam_headlights`: toggle low beams.
- `enable_high_beam_headlights` / `disable_high_beam_headlights`: toggle high beams.
- `enable_fog_lights` / `disable_fog_lights`: toggle fog lights.

## Driver assistance

- `reduce_target_speed` / `increase_target_speed`: adjust the simulated ADAS
  target by 10 km/h; this is separate from measured vehicle speed.
- `increase_following_distance` / `decrease_following_distance`: adjust the
  simulated following-distance level by one step, bounded from 1 to 5.
- `enable_adaptive_cruise_control` / `disable_adaptive_cruise_control`: toggle
  the simulated ACC state.
- `enable_lane_keep_assist` / `disable_lane_keep_assist`: toggle the simulated
  lane keeping state.
- `enable_blind_spot_monitor` / `disable_blind_spot_monitor`: toggle the
  simulated blind-spot monitoring state.
- `enable_privacy_mode` / `disable_privacy_mode`: toggle the simulated privacy mode.
- `apply_restrictive_adas_profile`: lower target speed by 10 km/h, increase
  following distance by one step, and enable ACC, lane keeping and blind-spot
  monitoring. Values are bounded by the corresponding controls.

## Other decisions

- `find_rest_area` is a guidance action, but it currently has no simulated
  vehicle-state effect. Do not claim a specific rest area was found or a route
  was started.
- `ask_attend_meeting` is a separate meeting-confirmation workflow, not a vehicle
  control. Use it only under its dedicated meeting instructions.
- Use `none` for capability questions that do not request an operation, or for
  unsupported operations. Do not claim physical actuation; actions update only
  the demo's simulated state.

When the wording is polite or indirect (for example, "Could you open the
windows?"), treat it as a command if it clearly requests a supported operation.
Choose the action by intended effect, not by a single keyword: warmer means
heating; cooler means air conditioning; stronger airflow means fan speed.