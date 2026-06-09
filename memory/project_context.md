---
name: RL Energy Grid project context
description: What the project is, its MDP design, stack constraints, and learner context
type: project
---

From-scratch Q-learning project (no RL frameworks). Agent manages a building's energy use over 24 hours, minimizing electricity cost and carbon emissions.

MDP: state=(hour 0-23, battery 0-10) → 240 states, 3 actions (grid/battery/shed), reward is always negative (minimization), 2000 training episodes.

Stack: Python 3.12.4, numpy, matplotlib, jupyter. No Gym, no pandas, no RL libraries.

Planned files: environment.py, agent.py, train.py, visualize.py, notebook.ipynb.

**Why:** Personal learning project — user is a beginner learning RL for the first time. Reference: Sutton & Barto Ch. 3, 6.1, 6.2, 6.5.

**How to apply:** Explain reasoning behind non-obvious implementation choices. Keep explanations beginner-friendly but don't oversimplify. Always implement from scratch — never suggest RL libraries.
