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
        self.market_price = 5.3

    def calculate_reward(self, battery_soc_idx: int, renewable_potential_idx: int) -> float:
        """
        Computes reward based on full state context and agent's action.
        Encourages smart load usage: consume when local energy is available or market is cheap.
        Avoids usage when grid is under deficit or electricity is expensive.

        :param battery_soc_idx: Discretized battery SoC index.
        :param renewable_potential_idx: Discretized renewable generation index.
        :param comfort_idx: 'acceptable' or 'expensive'.
        :return: Reward value based on symbolic context.
        """

        # Updated reward parameters
        sigma = 9     
        mu = 4        
        kappa = 6    

        psi = 9
        nu = 5       
        beta = 3   

        if self.action == 1:  # Turn ON
            # Turn on controllable load without battery and without renewable energy
            if battery_soc_idx == 0 and renewable_potential_idx == 0:
                # expensive
                if self.comfort_idx == 'expensive':
                    return -sigma * self.market_price
                else:
                    # Cheap
                    return mu / self.market_price
            else:
                # Use of battery or renewables
                return kappa * max(1, battery_soc_idx) * max(1, renewable_potential_idx)

        else:  # Turn OFF
            #Turn off controllable load without battery and without renewable energy
            if battery_soc_idx == 0 and renewable_potential_idx == 0:
                # expensive 
                if self.comfort_idx == 'expensive':
                    return psi * self.market_price
                else:
                    # Cheap
                    return - nu / self.market_price
            else:
                # Use of battery or renewables
                return - beta * max(1, battery_soc_idx) * max(1, renewable_potential_idx)


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
