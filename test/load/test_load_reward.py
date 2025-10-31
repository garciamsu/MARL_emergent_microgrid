"""
Load reward testing script.
Calculates rewards using the application's DefaultLoadReward function.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reordered imports
from core.rewards import DefaultLoadReward
import pandas as pd
from configs.loader import load_config

# Ensure the test/utils directory is added to sys.path for imports
_utils_dir = Path(__file__).resolve().parents[1] / 'utils'
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))
from reward_debug import build_reward_from_config

# Ensure the test/utils directory is added to sys.path for imports
_utils_dir = Path(__file__).resolve().parents[1] / 'utils'
if str(_utils_dir) not in sys.path:
    sys.path.insert(0, str(_utils_dir))

# Precio constante por defecto para env.price
DEFAULT_PRICE = 7.5

class _AgentStub:
    def __init__(self, action: int, comfort_threshold: float | None = None):
        self.action = action
        # Load default config value for comfort_threshold if not provided
        if comfort_threshold is None:
            try:
                cfg = load_config()
                self.comfort_threshold = cfg.get('agents', {}).get('load', {}).get('limits', {}).get('comfort_threshold', 1)
            except Exception:
                # If config cannot be loaded, fallback to 1
                self.comfort_threshold = 1
        else:
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

    # Adjusted reward function initialization
    reward_fn = build_reward_from_config('load', DefaultLoadReward)

    # Minimal env stub used by DefaultLoadReward (expects env.price)
    class _EnvStub:
        def __init__(self, price: float = DEFAULT_PRICE):
            self.price = price

    env_stub = _EnvStub()

    for _, row in data_frame.iterrows():
        agent = _AgentStub(action=int(row["action"]))

        # Support multiple possible column names to be robust with different CSVs
        def _to_int(val, default=0):
            try:
                return int(val)
            except (TypeError, ValueError):
                return default

        soc_idx = _to_int(
            row.get("soc_idx", row.get("battery_soc_idx", row.get("soc", 0)))
        )
        # 'comfort_idx' in some CSVs is categorical (e.g. 'acceptable'), so
        # fallback to 0 if it cannot be converted to int.
        demand_idx = _to_int(
            row.get("demand_idx", 0)
        )
        renewable_idx = _to_int(
            row.get(
                "renewable_idx", 0
            )
        )

        state_tuple = (soc_idx, demand_idx, renewable_idx)

        # Compute reward using a small env stub (provides price)
        reward = reward_fn.compute(agent, env=env_stub, state_tuple=state_tuple)
        rewards.append(reward)

        # Log the reward calculation for debugging
        print(f"State: {state_tuple}, Action: {agent.action}, Reward: {reward}")

    data_frame["price"] = DEFAULT_PRICE 
    data_frame["comfort_threshold"] = agent.comfort_threshold
    data_frame["reward"] = rewards
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Load reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    base = Path(__file__).parent / 'input'

    input_file = base / 'LoadAgent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'output' / 'reward_load.csv'

    try:
        run_test(input_file, output_file)
    except FileNotFoundError as file_err:
        print(f"File error: {file_err}")
    except ValueError as value_err:
        print(f"Value error: {value_err}")
