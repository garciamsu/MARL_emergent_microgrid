# Distributed Multi-Agent Reinforcement Learning Environment

This repository provides a Python-based simulation framework for studying **distributed coordination of renewable energy resources** using **multi-agent reinforcement learning (MARL)**.  
The system models the interaction between solar generation, wind generation, battery storage, controllable loads, and the main utility grid, applying Q-learning with discretized state-action spaces.

The codebase is designed for **engineers and researchers familiar with Python and reinforcement learning**, aiming to facilitate reproducibility and extensibility.

---

## Features

- Multi-agent simulation of energy systems under dynamic conditions
- Modular architecture with specialized agent classes
- Discretization of state variables into configurable bins
- Q-learning implementation with per-agent policies
- Automatic generation of:
  - Evolution logs (`CSV`)
  - Episode metrics (`Excel`)
  - Q-table snapshots (`Excel`)
  - High-resolution and vector plots (`SVG`)
- Analysis and visualization utilities

---

## Installation

**Recommended platform:** Linux Ubuntu 22.04.5 LTS  
**Python version:** 3.11.7  
**Environment manager:** Anaconda (conda 24.9.1)

### Create a conda environment

```bash
conda create -n marl_env python=3.11.7
```

### Activate the environment

```bash
conda activate marl_env
```

### Install required packages

```bash
pip install -r requirements.txt
```

*(Make sure `requirements.txt` includes `numpy`, `pandas`, `matplotlib`, `openpyxl`, etc.)*

---

## Running the Simulation

The main entry point is `main.py`.

```bash
python main.py
```

**Parameters** such as:

- Number of episodes
- Exploration factor (`epsilon`)
- Input dataset (`csv_filename`)

are set in the `Simulation` class instantiation.  
**Details of parameter values are described in the accompanying scientific article.**

Running the simulation will:

1. Initialize the environment and all agents
2. Train agents via Q-learning
3. Generate results in the `results/` directory
4. Output performance metrics to the console

---

## Folder Structure

After a simulation run, your project will look like this:

```
results/
├── evolution/
│   └── learning_<episode>.csv          # Detailed logs per episode
├── q_tables/
│   └── qtable_<agent>_ep<episode>.xlsx # Q-table snapshots
├── plots/
│   ├── IAE_over_episodes.svg
│   ├── Var_dif_over_episodes.svg
│   ├── Q_Norm_<agent>.svg
│   └── env_plot.svg
└── metrics_episode.xlsx                # Episode summary metrics
```

---

## Metrics Overview

To **evaluate coordination effectiveness**, the system computes:

- **Energy Balance (ΔP):** Instantaneous difference between generation and demand
- **ISE:** Integral Square Error
- **IAE:** Integral Absolute Error
- **REP:** Renewable Energy Penetration (%)
- **GEP:** Grid Energy Penetration (%)

To **evaluate learning**, it also calculates:

- **Average Reward per Episode**
- **Average Cumulative Reward**

**Additional stability metrics** are implemented and will be detailed in the publication.

---

## Extending the Framework

You can create new agents or customize the environment:

### Adding a New Agent

1. **Subclass `BaseAgent`**
2. Implement:
   - `get_discretized_state()`: Define state representation
   - `initialize_q_table()`: Configure state-action space
   - `calculate_reward()`: Design the reward function

### Customizing the Environment

- Subclass `MultiAgentEnv` to load different datasets or apply new discretization schemes.

This design supports **flexible experimentation without altering core components.**

---

## Utilities

The module `analysis_tools.py` provides helper functions:

| Function | Purpose |
|---|---|
| `load_latest_evolution_csv()` | Load the most recent simulation log |
| `plot_metric()` | Generate metric plots |
| `compute_q_diff_norm()` | Compute L2 norm between Q-tables |
| `check_stability()` | Evaluate stability over episodes |
| `process_evolution_data()` | Prepare logs for visualization |
| `plot_coordination()` | Generate multi-panel plots of agent behavior |
| `clear_results_directories()` | **Clean all files under `results/`** |

> **Note:** `clear_results_directories()` removes previous outputs before a new simulation run.

---

## Outputs and Visualization

By default, the simulation generates:

- Time series plots of power and energy balance
- SVG graphics of learning progress (`IAE`, `Var_dif`, `Q Norms`)
- Per-agent Q-tables and rewards

Visual outputs help validate whether agents learn effective coordination strategies.

---

## Notes

- Input datasets should be placed in `assets/datasets/` in CSV format.
- Default time resolution: **1 hour per time step**
- For parameter explanations and case studies, see the accompanying article.

---

## License

This project is released for **academic research** purposes.  
Please **cite appropriately** if used in publications.

