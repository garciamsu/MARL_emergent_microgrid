"""Wind reward testing script.

Calculates rewards using the application's DefaultWindReward function.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from core.rewards import DefaultWindReward

# Ensure the test/utils directory is added to sys.path for imports
_utils_dir = Path(__file__).resolve().parents[1] / "utils"
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))
from reward_debug import build_reward_from_config


class _AgentStub:
    def __init__(self, action: int):
        self.action = action


def run_test(input_path: Path, output_path: Path):
    """Run the wind reward computation using an input CSV and save the output.

    The input CSV is expected to use ';' as delimiter and contain action/state
    columns that map to the reward function. The output CSV is written with
    UTF-8 encoding.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data_frame = pd.read_csv(input_path, delimiter=";")
    rewards = []

    reward_fn = build_reward_from_config("wind", DefaultWindReward)
    for _, row in data_frame.iterrows():
        agent = _AgentStub(action=int(row["action"]))
        state_tuple = (
            int(row.get("wind_idx", row.get("wind_potential_idx", 0))),
            int(row.get("demand_idx", row.get("demand_power_idx", 0))),
            int(row.get("renewable_potential_idx", 0)),
        )
        reward = reward_fn.compute(agent, env=None, state_tuple=state_tuple)
        rewards.append(reward)
        print(f"State: {state_tuple}, Action: {agent.action}, Reward: {reward}")

    data_frame["reward"] = rewards
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wind reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = Path(__file__).parent / "input" / "Wind_Agent_Reward_Table.csv"
    output_file = Path(__file__).parent / "output" / "reward_wind.csv"

    try:
        run_test(input_file, output_file)
    except FileNotFoundError as file_err:
        print(f"File error: {file_err}")
    except ValueError as value_err:
        print(f"Value error: {value_err}")
