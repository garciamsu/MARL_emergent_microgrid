import math
from pathlib import Path
import pandas as pd
import os

class LoadAgent:
    def __init__(self, action: int, comfort_idx: str):
        """
        Initialize the LoadAgent with selected action and comfort condition.

        Parameters:
            action (int): 0 for OFF, 1 for ON.
            comfort_idx (str): 'acceptable' or 'expensive'.
        """
        self.action = action
        self.comfort_idx = comfort_idx

    def calculate_reward(self, battery_soc_idx: int, renewable_potential_idx: int) -> float:
        """
        Computes reward based on the system state and agent action.

        Encourages smart load usage: consume when local energy is available or market is cheap.
        Avoids usage when grid is under deficit or electricity is expensive.

        Parameters:
            battery_soc_idx (int): Discretized battery SoC index.
            renewable_potential_idx (int): Discretized renewable generation index.

        Returns:
            float: Reward value.
        """
        # Reward parameters
        sigma = 2     # penalty for expensive ON
        mu = 2        # reward for cheap ON with no local energy
        kappa = 2     # reward for ON with battery or renewable

        psi = 2       # reward for OFF in expensive condition
        nu = 2        # penalty for OFF in cheap condition with no local energy
        beta = 2      # penalty for OFF when there is battery or renewable

        if self.action == 1:  # Turn ON
            if battery_soc_idx == 0 and renewable_potential_idx == 0:
                return -sigma if self.comfort_idx == 'expensive' else mu
            else:
                return kappa
        else:  # Turn OFF
            if battery_soc_idx == 0 and renewable_potential_idx == 0:
                return psi if self.comfort_idx == 'expensive' else -nu
            else:
                return -beta


def run_test(input_path, output_path):
    df = pd.read_csv(input_path, delimiter=';')

    rewards = []
    for _, row in df.iterrows():
        agent = LoadAgent(
            action=row["action"],
            comfort_idx=row["comfort_idx"]
        )
        reward = agent.calculate_reward(
            battery_soc_idx=row["battery_soc_idx"],
            renewable_potential_idx=row["renewable_potential_idx"]
        )
        rewards.append(reward)

    df["reward"] = rewards
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Load reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = Path(__file__).parent / 'data' / 'LoadAgent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'reports' / 'reward_load.csv'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    run_test(input_file, output_file)
