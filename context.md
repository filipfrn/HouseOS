# RL Energy Grid — Project Context

## What This Is
A from-scratch Q-learning implementation in Python. An agent manages a building's energy use over a 24-hour day, minimizing electricity cost and carbon emissions. Personal learning project — not academic, no external framework.

## Stack
- Python, numpy, matplotlib, jupyter
- No RL frameworks, no Gym, no pandas
- Q-table is a numpy array of shape (24, 11, 3)

## Project Structure (planned)
```
rl-energy-grid/
├── environment.py      # EnergyEnv class
├── agent.py            # Q-learning agent
├── train.py            # Training loop
├── visualize.py        # Plots
└── notebook.ipynb      # Scratch / exploration
```

## The MDP
- **State:** (hour, battery_level) — hour: 0–23, battery: 0–10 → 240 states
- **Actions:** 0 = draw from grid, 1 = use battery, 2 = shed load
- **Reward:** negative electricity cost minus carbon penalty
- **Episode:** one full 24-hour day (24 steps)
- **Optional extension:** solar PV as a 4th action, available hours 8–18

## Environment
`EnergyEnv` class with:
- `reset()` → returns starting state (hour=0, battery=5)
- `step(action)` → returns (next_state, reward, done)

Three hardcoded 24-element numpy arrays:
- `price_curve` — €/kWh, cheap at night (~0.07), expensive at peak 18–20h (~0.28), German day-ahead style
- `carbon_curve` — gCO2/kWh, high during peak demand hours
- `demand_curve` — kWh/hour, high in morning and evening

Battery rules: never below 0, never above 10. Penalize if agent tries to use empty battery.

## Agent
- Q-table: `np.zeros((24, 11, 3))`
- `alpha = 0.1`, `gamma = 0.95`, `epsilon` decays from 1.0 → 0.01
- Epsilon-greedy action selection
- Standard Q-learning update rule (off-policy TD control, S&B Chapter 6.5)
- 2000 training episodes

## Conventions
- No external RL libraries — implement everything from scratch
- Test environment manually (step through 24 actions by hand) before adding agent
- Reward is always negative (minimization problem)
- Keep environment and agent in separate files

## Learner Context
Beginner — learning RL for the first time. Reference material: Sutton & Barto Chapters 3 and 6.1, 6.2, 6.5. Explain reasoning when making non-obvious implementation choices.