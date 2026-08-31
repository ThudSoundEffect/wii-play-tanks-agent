# Wii Play Tanks RL Agent

A reinforcement learning agent that plays Wii Play Tanks live through the Dolphin emulator by reading game state from emulator memory and driving input directly.

## Status: Work in progress (training not yet complete)

This project is actively being trained and debugged. It is **not** a finished, tuned agent yet. Further information is reflective of the current state, not a final result.

## Environment (`tank_env.py`)

- Launches and hooks into a running Dolphin instance via `dolphin-memory-engine`, reading lives, bullets, level, position, and destroyed enemies counters directly from emulated RAM each step.
- Captures the game window via `mss` screen capture and drives input (movement, mouse aim, fire) via `pydirectinput`.
- Action space: `dpad` (8 directional movement), `pointer` (24 discrete aim bins), `buttons` (fire).
- **Curriculum aware reset**: on `reset()`, the environment samples a curriculum stage probabilistically. Sampling decays exponentially `0.5^(max_stage - stage)` so within the currently unlocked range of stages, harder stages are sampled more heavily than earlier ones. The system is in place to prevent overtraining on early stages that the agent has passed, while still allowing some exposure to earlier levels but focusing training on hardest unlocked levels.
## Observation & Reward Wrapper (`tank_env_wrapper.py`)

Where most of the game specific logic lives, on top of the raw env:

- **Radar encoding**: agent and aim radar rays are cast and encoded as `[normalized_distance, one_hot(detect_type)]` per ray (miss, wall, threat type).
- **Aim forecasting**: for each of the 24 aim bins, a ray is marched forward against a wall mask for the captured frame. On a wall hit, the ray reflects and continues, so the system can predict ricochet trajectories. Each bin fires 3 sub rays (±5°) rather than one, to smooth over single pixel wall mask noise. Per bin, this produces: predicted hit or miss, a lead shot flag for hitting a moving threat's predicted position, and a self danger flag for trajectories that end near the agent.
- **`best_aim_bin`**: the bin with the best predicted outcome. Used both as a feature fed into the observation and as the supervised target for the auxiliary aim prediction loss.
- **Action masking**: exposes `action_masks()` (used by `sb3_contrib`'s `ActionMasker`) to disable invalid game actions i.e. firing is masked out once the per curriculum stage bullet limit has been reached
- **Reward shaping**, grouped into logged categories:
  - `core`: enemy destroyed, level advance, curriculum cap bonus
  - `shaping`: aim accuracy, radar based penalties
  - `penalty`: difference from the forecasted best aim bin, per step penalty
  - `guardrail`: life lost

All weights are staged per curriculum level in `config.py`, ramping up penalties as the agent progresses.

## Curriculum Gating (`curriculum.py`)

- Tracks rolling success rate per stage and only promotes the agent to the next stage once success rate clears a threshold for a **streak** of consecutive evaluation windows.
- Model checkpoints save alongside curriculum state, so resuming training resumes at the correct stage.

## Policy & Training

- **Feature extractor** (`feature_extractor.py`): multibranch CNN: a shared CNN + status encoder feeding into three specialized subheads (movement, aiming, actions), and a small auxiliary head that predicts `best_aim_bin` from the pointer embedding.
- **Custom PPO** (`custom_ppo.py`): extends `sb3_contrib.MaskablePPO` to add the auxiliary aim prediction loss into the training step to give the aim subnetwork a denser gradient signal than standard PPO.

## Current focus

Debugging and stabilizing training dynamics. 

Recent work has included:
- fixing a ray marching bug that was causing aim forecasts to collapse at the agent's own position
- correcting a position normalization bug
- tuning reward scaling/entropy coefficients so the value function and policy converge properly

Full curriculum training runs are ongoing.

## Not yet done

- Full training run to convergence
- Evaluation on game levels beyond early curriculum levels
- Cleanup and documentation of the codebase
