# Reinforcement Learning & MDP Summary

## What is RL and MDP?

An agent interacts with an environment over time — it observes a state, takes an action, receives a reward, and moves to a new state. The MDP (Markov Decision Process) is the formal framework formalizing this loop.

---

## The Three Components

**Reward** — the purpose of the task. Best to define first, because if it's unclear, the problem is unclear. Can be delayed, sparse, and subjective (e.g. a scheduling assistant where the user rates meeting placements).

**State** — everything the agent needs to make a good decision right now. Key question: *what would I want to know if I were the agent?* Doesn't need to capture everything, just what's relevant to the next action.

**Actions** — the levers the agent has to influence the environment. The right abstraction level depends on the goal. Can be structured and variable-sized (e.g. molecular discovery with bond placements).

---

## MDP Assumptions

| Assumption | Breaks when... |
|---|---|
| **Markov property** | History matters (e.g. patient drug history) |
| **Full observability** | State is hidden (e.g. poker, robotics) |
| **Fixed transitions** | Environment dynamics shift over time (e.g. financial markets) |
| **Fixed reward** | Goals shift over time |
| **Discrete time steps** | Process is continuous |

---

## Key Insights

- **Start with reward** when designing an MDP — it anchors the purpose of the task
- **Stochasticity ≠ partial observability** — uncertain outcomes are fine in MDP; hidden state is not
- **The agent-environment boundary** is a modeling choice, not an objective fact — draw it where learning is actually needed
- **Partial observability and non-stationarity** are the two most common real-world limitations of the base MDP framework — extensions like POMDPs exist to address them

## State-value function
- **V(s) = E[G(t) | S(t) = s]**
- V(s) is the expected total discounted reward an agent will accumulate from state s onwards, following a given policy.

## Return G(t)
The total discounted reward from timestep t onwards in one specific episode:
 
```
G(t) = R(t+1) + γR(t+2) + γ²R(t+3) + ...
```
 
Or recursively:
 
```
G(t) = R(t+1) + γG(t+1)
```
 
- G(t) is a **concrete number** from one specific run
- It is **random** — different actions lead to different G(t)
- In episodic tasks the sum terminates naturally at T
- In continuing tasks gamma < 1 ensures the sum stays finite
---
 
## Value Functions
 
### State Value Function V(s)
The expected total discounted reward from state s onwards, following the current policy:
 
```
V(s) = E[G(t) | S(t) = s]
```
 
V(s) is the average G(t) across infinitely many episodes starting from s. You never compute it directly — your Q-table approximates it through experience.
 
### Action Value Function Q(s,a)
The expected total discounted reward from state s, taking action a first:
 
```
Q(s,a) = E[G(t) | S(t) = s, A(t) = a]
```
 
Q gives finer resolution than V — not just "how good is this state" but "how good is this specific action in this state."
 
### Relationship between V and Q
Once you have the optimal Q-table:
 
```
V(s) = max_a Q(s,a)
```
 
---
 
## Discounting (gamma)
- gamma ∈ [0, 1) controls how much future rewards are valued relative to immediate ones
- gamma = 0 → agent only cares about immediate reward
- gamma = 1 → agent treats all future rewards equally (valid for episodic tasks)
- gamma = 0.95 → slight preference for sooner rewards, good default for energy scheduling
---
 
## Monte Carlo vs TD Learning
 
| | Monte Carlo | TD(0) |
|---|---|---|
| When it updates | End of episode | After every step |
| What it uses | Actual return G(t) | Estimated return R(t+1) + γV(S(t+1)) |
| Needs full episode | Yes | No |
| Bootstraps | No | Yes |
| Bias | None (uses real return) | Some (uses estimate) |
| Variance | High | Low |
 
**Bootstrapping** means updating an estimate using another estimate — using your current guess of V(S(t+1)) instead of waiting for the real future return. This is what makes TD learn after every step rather than waiting until the end.
 
---
 
## TD(0) Update Rule
 
```
V(St) ← V(St) + α[R(t+1) + γV(S(t+1)) − V(St)]
```
 
- **R(t+1) + γV(S(t+1))** — the TD target. Approximation of G(t) using bootstrapping
- **R(t+1) + γV(S(t+1)) − V(St)** — the TD error δt. How wrong your current estimate was
- **α** — learning rate. Controls size of the update nudge
- Update happens at t+1 looking back at t — you need to arrive at S(t+1) before you can update V(St)
### Monte Carlo error as sum of TD errors
When V is fixed (as in Monte Carlo):
 
```
G(t) - V(St) = Σ γ^(k-t) δk,   for k = t to T-1
```
 
The total Monte Carlo error equals the discounted sum of all TD errors across the episode. When V updates during the episode (TD), this only holds approximately — the smaller alpha is, the better the approximation.
 
---
 
## Q-Learning
 
### The Q-table
A numpy array of shape (states × actions) storing the expected return for every state-action pair. In the energy project: shape (24, 11, 3).
 
Initialized to zeros. Updated after every step. Converges asymptotically to Q* — the optimal action-value function.
 
### Update Rule
 
```
Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) − Q(S,A)]
```
 
- **R** — immediate reward (the agent learns from this, including bad exploratory actions)
- **max_a Q(S',a)** — assumes optimal behavior from the next state onwards
- The `max` is what makes Q-learning off-policy
### Epsilon-Greedy Action Selection
Balances exploration and exploitation:
- With probability ε → pick random action (explore)
- With probability 1-ε → pick action with highest Q-value (exploit)
- ε decays from 1.0 → 0.01 over training: explore early, exploit later
---
 
## On-Policy vs Off-Policy
 
| | On-Policy (SARSA) | Off-Policy (Q-learning) |
|---|---|---|
| Learns about | The policy currently being followed | The optimal policy |
| Next action in update | Actual action taken A' | Best possible action max Q(S',a) |
| Effect of exploration | Contaminates the learned values | Ignored in the update |
| Converges to | Q for current policy | Q* optimal policy |
 
**Key insight:** Q-learning learns from mistakes via R (the immediate reward always reflects what actually happened). The `max` only affects the future estimate — not whether the current bad action gets penalized.
 
---
 
## Key Definitions
 
**Bootstrapping:** updating an estimate using another estimate rather than waiting for the true outcome.
 
**TD error (δt):** `R(t+1) + γV(S(t+1)) − V(St)` — how wrong your current estimate was given what you just observed.
 
**TD target:** `R(t+1) + γV(S(t+1))` — your improved estimate of what V(St) should be.
 
**Converge asymptotically:** approaching the true value forever, getting arbitrarily close but only reaching it at t = ∞. In practice you stop when close enough.
 
**Off-policy:** learning about a different policy (optimal) than the one used to generate behavior (epsilon-greedy).