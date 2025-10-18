"""
Solar reward testing script.
Calculates rewards using the application's DefaultSolarReward function.
"""
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.rewards import DefaultSolarReward

class _AgentStub:
    def __init__(self, action: int):
        self.action = action


def run_test(input_path: Path, output_path: Path):
    """
    Runs the reward calculation test for the solar agent using input data.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file with rewards.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Solar CSV usa comas
    data_frame = pd.read_csv(input_path, delimiter=',')
    rewards = []

    reward_fn = DefaultSolarReward()
    for _, row in data_frame.iterrows():
        agent = _AgentStub(action=int(row["action"]))
        # state_tuple: (solar_idx, demand_idx, renewable_idx)
        state_tuple = (
            int(row["solar_potential_idx"]),
            int(row["demand_power_idx"]),
            int(row["total_power_idx"])  # treat total as renewable aggregate
        )
        reward = reward_fn.compute(agent, env=None, state_tuple=state_tuple)
        rewards.append(reward)

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
