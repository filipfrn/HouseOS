# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

From-scratch Q-learning implementation in Python. An agent manages a building's energy use over a 24-hour day, minimizing electricity cost and carbon emissions. Personal learning project — no RL frameworks, no Gym, no pandas.

## Environment

- Python 3.12.4, venv at `venv\`
- Activate: `.\venv\Scripts\Activate.ps1`
- Dependencies: numpy, matplotlib, jupyter
- Install: `pip install numpy matplotlib jupyter`

## Stack Constraints

- No external RL libraries — implement everything from scratch
- No Gym, no stable-baselines, no pandas
- Q-table is a plain numpy array: `np.zeros((24, 11, 3))`

## Project Structure

```
environment.py      # EnergyEnv class
agent.py            # Q-learning agent
train.py            # Training loop
visualize.py        # Plots
notebook.ipynb      # Scratch / exploration
```

## The MDP

- **State:** `(hour, battery_level)` — hour: 0–23, battery: 0–10 → 240 states
- **Actions:** 0 = draw from grid, 1 = use battery, 2 = shed load
- **Reward:** negative electricity cost minus carbon penalty (always negative — minimization)
- **Episode:** one full 24-hour day (24 steps)
- **Optional extension:** solar PV as a 4th action, available hours 8–18

## EnergyEnv

Three hardcoded 24-element numpy arrays: `price_curve` (€/kWh, cheap at night ~0.07, peak 18–20h ~0.28), `carbon_curve` (gCO2/kWh), `demand_curve` (kWh/hour).

- `reset()` → starting state `(hour=0, battery=5)`
- `step(action)` → `(next_state, reward, done)`
- Battery bounds: clamp to [0, 10]; penalize if agent tries to use empty battery

## Agent

- `alpha = 0.1`, `gamma = 0.95`, epsilon decays 1.0 → 0.01 over 2000 episodes
- Epsilon-greedy action selection
- Standard Q-learning update (off-policy TD control, Sutton & Barto Ch. 6.5)

## Conventions

- Test environment manually (step through 24 actions by hand) before wiring up the agent
- Keep environment and agent in separate files
- Explain reasoning behind non-obvious implementation choices (learner context: first RL project, reference is S&B Ch. 3, 6.1, 6.2, 6.5)
