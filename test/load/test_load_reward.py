"""
Load reward testing script.
Calculates rewards using the application's DefaultLoadReward function.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from reward_debug import explain_and_compute, build_reward_from_config
from core.rewards import DefaultLoadReward

# Ensure the test/utils directory is added to sys.path for imports
_utils_dir = Path(__file__).resolve().parents[1] / 'utils'
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

# Modo de ejecución: True => paso a paso (espera BARRA ESPACIADORA),
# False => ejecución continua
STEP_BY_STEP = False
# Precio constante por defecto para env.price
DEFAULT_PRICE = 1.0

class _AgentStub:
    def __init__(self, action: int, comfort_threshold: float = 1):
        self.action = action
        self.comfort_threshold = comfort_threshold


def run_test(input_path: Path, output_path: Path):
    """
    Runs the reward calculation test for the load agent using input data.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file with rewards.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data_frame = pd.read_csv(input_path, delimiter=';')
    rewards = []

    reward_fn = build_reward_from_config('load', DefaultLoadReward)
    # Fake env with price attribute for reward function needs
    class _Env:
        def __init__(self, price):
            self.price = price

    for _, row in data_frame.iterrows():
        agent = _AgentStub(action=int(row["action"]))
        env = _Env(price=DEFAULT_PRICE)
        # state_tuple: (soc_idx, demand_idx, renewable_idx)
        state_tuple = (
            int(row["load_soc_idx"]),
            int(row["demand_power_idx"]),
            int(row["renewable_potential_idx"])
        )
        reward = explain_and_compute(
            reward_fn, agent, env=env, state_tuple=state_tuple, step=STEP_BY_STEP
        )
        rewards.append(reward)

        # Log the reward calculation for debugging
        print(f"State: {state_tuple}, Action: {agent.action}, Reward: {reward}")

    data_frame["reward"] = rewards
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Load reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent / 'input'
    cand1 = base / 'Load_Agent_Reward_Table.csv'
    cand2 = base / 'LoadAgent_Reward_Table.csv'
    input_file = cand1 if cand1.exists() else cand2
    output_file = Path(__file__).parent / 'output' / 'reward_load.csv'

    try:
        run_test(input_file, output_file)
    except FileNotFoundError as file_err:
        print(f"File error: {file_err}")
    except ValueError as value_err:
        print(f"Value error: {value_err}")
