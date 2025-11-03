"""
Battery reward testing script.
Calculates rewards using the application's DefaultBatteryReward function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Adjusted import order to resolve linting issues
import pandas as pd
from core.rewards import DefaultBatteryReward

# Ensure the test/utils directory is added to sys.path for imports
_utils_dir = Path(__file__).resolve().parents[1] / 'utils'
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

# Import debug helpers from test/utils after ensuring the directory is on sys.path
from reward_debug import build_reward_from_config, explain_and_compute

# NOTE: La lógica de ejecución paso a paso ha sido eliminada; el script
# siempre ejecuta en modo continuo.

class _AgentStub:
    def __init__(self, action: int, soc_max: float = 1.0):
        self.action = action
        self.soc_max = soc_max


class _EnvStub:
    """Minimal environment stub that provides the attributes expected by
    DefaultBatteryReward.compute: `renewable_power_idx` and `demand_power_idx`.
    """

    def __init__(self, renewable_power_idx: int, demand_power_idx: int):
        self.renewable_power_idx = renewable_power_idx
        self.demand_power_idx = demand_power_idx

def run_test(input_path: Path, output_path: Path):
    """
    Runs the reward calculation test for the battery agent using input data.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file with rewards.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data_frame = pd.read_csv(input_path, delimiter=';')
    rewards = []

    reward_fn = build_reward_from_config('battery', DefaultBatteryReward)

    for _, row in data_frame.iterrows():
        # Map discrete soc_idx
        soc_idx = float(row["battery_soc_idx"])
        agent = _AgentStub(action=int(row["action"]))
        # state_tuple: (soc, demand_idx, total_idx)
        state_tuple = (
            soc_idx,
            int(row["demand_power_idx"]),
            int(row["renewable_idx"])
        )
        # Construir un stub de entorno con los índices relevantes desde el CSV
        env = _EnvStub(
            renewable_power_idx=int(row.get("renewable_idx", row.get("renewable_idx", 0))),
            demand_power_idx=int(row.get("demand_power_idx", row.get("demand_idx", 0)))
        )
        # Direct reward calculation
        reward = reward_fn.compute(agent, env=env, state_tuple=state_tuple)
        rewards.append(reward)

        # Log the reward calculation for debugging
        # print(f"State: {state_tuple}, Action: {agent.action}, Reward: {reward}")

    data_frame["reward"] = rewards
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Battery reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = Path(__file__).parent / 'input' / 'Battery_Agent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'output' / 'reward_battery.csv'

    try:
        run_test(input_file, output_file)
    except FileNotFoundError as file_err:
        print(f"File error: {file_err}")
    except ValueError as value_err:
        print(f"Value error: {value_err}")
