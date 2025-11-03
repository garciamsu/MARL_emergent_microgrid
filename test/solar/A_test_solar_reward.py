"""
Solar reward testing script.
Calculates rewards using the application's DefaultSolarReward function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from core.rewards import DefaultSolarReward

# Ensure the test/utils directory is added to sys.path for imports
_utils_dir = Path(__file__).resolve().parents[1] / 'utils'
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))
from reward_debug import build_reward_from_config


class _AgentStub:
    def __init__(self, action: int):
        self.action = action


class _EnvStub:
    """Stub minimal del entorno para pruebas de recompensas.

    Proporciona los atributos que `DefaultSolarReward.compute` espera:
    `renewable_power_idx` y `demand_power_idx`.
    """

    def __init__(self, renewable_power_idx: int, demand_power_idx: int):
        self.renewable_power_idx = renewable_power_idx
        self.demand_power_idx = demand_power_idx


def run_test(input_path: Path, output_path: Path):
    """
    Runs the reward calculation test for the solar agent using input data.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file with rewards.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Solar CSV usa comas
    data_frame = pd.read_csv(input_path, delimiter=';')
    rewards = []

    reward_fn = build_reward_from_config('solar', DefaultSolarReward)
    for _, row in data_frame.iterrows():
        agent = _AgentStub(action=int(row["action"]))
        # state_tuple: (solar_idx, demand_idx, renewable_idx)
        state_tuple = (
            int(row["solar_idx"]),
            int(row["demand_idx"]),
            int(row["renewable_idx"])  # treat total as renewable aggregate
        )
        # Construir un stub de entorno con los índices del CSV y calcular la recompensa
        env = _EnvStub(renewable_power_idx=int(row["renewable_idx"]),
                       demand_power_idx=int(row["demand_idx"]))
        reward = reward_fn.compute(agent, env=env, state_tuple=state_tuple)
        rewards.append(reward)

        # Log the reward calculation for debugging
        print(f"State: {state_tuple}, Action: {agent.action}, Reward: {reward}")

    data_frame["reward"] = rewards
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Solar reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = Path(__file__).parent / 'input' / 'Solar_Agent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'output' / 'reward_solar.csv'

    try:
        run_test(input_file, output_file)
    except FileNotFoundError as file_err:
        print(f"File error: {file_err}")
    except ValueError as value_err:
        print(f"Value error: {value_err}")
