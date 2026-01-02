# Episode Reward Calculation - Quick Reference

## The ONLY Acceptable Pattern

```python
episode_rewards = []

for episode in range(num_episodes):
    episode_reward = 0.0  # ← RESET at start of EACH episode

    for t in range(max_steps):
        timestep_reward = compute_reward(...)
        episode_reward += timestep_reward  # ← ACCUMULATE within episode

    episode_rewards.append(episode_reward)  # ← ONE value per episode
    # episode_reward automatically resets in next iteration
```

## Variable Definitions

| Variable | Type | Scope | Description |
|----------|------|-------|-------------|
| `timestep_reward` | float | Single timestep | Reward at time t (r_t) |
| `episode_reward` | float | Single episode | Σ(r_t) for current episode |
| `episode_rewards` | list | All training | One value per episode |

## Formula

```
episode_reward[e] = Σ(timestep_reward[t]) for t ∈ [0, T-1]
```

Where:
- `e` = episode index
- `t` = timestep index within episode
- `T` = number of timesteps in episode

**CRITICAL**: `episode_reward[e]` is **independent** of `episode_reward[e-1]`

## DO ✅

```python
# Initialize storage
episode_rewards = []

# Training loop
for episode in range(num_episodes):
    episode_reward = 0.0  # Reset
    
    for step in range(max_steps):
        r = get_reward()
        episode_reward += r  # Accumulate
    
    episode_rewards.append(episode_reward)  # Store
```

## DON'T ❌

```python
# WRONG: Global accumulation
cumulative_reward = 0.0  # ❌

for episode in range(num_episodes):
    for step in range(max_steps):
        r = get_reward()
        cumulative_reward += r  # ❌ Never resets!
```

```python
# WRONG: Appending timestep rewards
episode_rewards = []  # Will be wrong length!

for episode in range(num_episodes):
    for step in range(max_steps):
        r = get_reward()
        episode_rewards.append(r)  # ❌ One per step, not per episode!
```

## Output Format

### CSV Structure
```
episode,agent1,agent2,agent3
0,123.45,234.56,345.67
1,125.32,236.41,341.23
2,121.98,232.11,348.92
```

- **Rows**: One per episode
- **Columns**: One per agent (plus episode number)
- **Values**: Total reward for that episode

### Length Check
```python
assert len(episode_rewards) == num_episodes  # ✅
assert len(episode_rewards) == num_episodes * max_steps  # ❌
```

## Plotting

### Correct
```python
plt.plot(episodes, episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Episode Reward')  # NOT "Cumulative Reward"
```

### Incorrect
```python
cumulative = np.cumsum(episode_rewards)  # ❌
plt.plot(episodes, cumulative)  # ❌ This is WRONG
```

## Interpretation

**Episode Reward Plot**:
- ✅ Can go up or down
- ✅ Shows true learning progress
- ✅ Fluctuations are normal
- ✅ Look for upward trend

**NOT**:
- ❌ Should not always increase
- ❌ Is not cumulative across episodes
- ❌ Does not carry over between episodes

## Verification

```python
# Manual check for episode 0
df = pd.read_csv('results/evolution/episode_0.csv')
manual_sum = df['reward_agent'].sum()

rewards = pd.read_csv('results/logs/episode_rewards.csv')
stored = rewards.loc[0, 'agent']

assert abs(manual_sum - stored) < 1e-6  # Should match
```

## Files in This Project

| File | Purpose |
|------|---------|  
| `core/simulation.py` | Implementation |
| `results/logs/episode_rewards.csv` | Output data |
| `analysis/E_accumulated_reward.py` | Complete Analysis, Validation & Visualization |
| `docs/EPISODE_REWARDS_GUIDE.md` | Full documentation |## Commands

```bash
# Run training
python main.py

# Comprehensive analysis (includes validation, statistics, and all plots)
python analysis/E_accumulated_reward.py
```

## One-Line Summary

**Episode reward = sum of timestep rewards within ONE episode only, reset to 0 at start of each episode.**
