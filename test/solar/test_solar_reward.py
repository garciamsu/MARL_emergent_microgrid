from pathlib import Path
import math
import pandas as pd
import numpy as np
import os

class SolarAgent:
    def __init__(self, action):
        self.action = action

    def calculate_reward(
        self,
        solar_potential_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the solar agent based on its action, solar potential,
        total system power output, and current demand.

        Parameters:
            solar_potential_idx (int): Discretized index of available solar resource.
            total_power_idx (int): Discretized index of total system power (all agents).
            demand_power_idx (int): Discretized index of system demand.

        Returns:
            float: Reward signal guiding the solar agent's learning.
        """
        
        # Reward adjustment parameters
        sigma = 15
        kappa = 3
        mu = 12
        nu = 1
        beta = 5
        xi = 8
        
        power_gap = total_power_idx - demand_power_idx

        # Action = produce
        if self.action == 1:
            if solar_potential_idx == 0:
                # ⚠️ Trying to produce without sun → strong penalty
                return -sigma * math.log(demand_power_idx + 1)
            elif solar_potential_idx > 0 and power_gap >= 0:
                # ✅ Producing when demand is already covered → moderate reward
                return kappa * solar_potential_idx
            else:
                # ✅ Producing when there's energy deficit → higher reward
                return mu * np.tanh(abs(power_gap))

        # Action = idle
        else:
            if solar_potential_idx == 0:
                # ✅ No sun, no action → small positive reinforcement
                return nu
            elif solar_potential_idx > 0 and power_gap >= 0:
                # ⚠️ Wasting available sun when demand is covered → small penalty
                return -beta * solar_potential_idx
            else:
                # ⚠️ Not helping in a deficit despite available sun → strong penalty
                return max(-xi * abs(power_gap), -50)

def run_test(input_path, output_path):
    df = pd.read_csv(input_path, delimiter=';')

    rewards = []
    for _, row in df.iterrows():
        agent = SolarAgent(action=row["action"])
        reward = agent.calculate_reward(
            row["solar_potential_idx"],
            row["total_power_idx"],
            row["demand_power_idx"]
        )
        rewards.append(reward)

    df["reward"] = rewards
    df.to_csv(output_path, index=False)
    print(f"Test completed. Results saved to: {output_path}")

if __name__ == "__main__":

    input_file = Path(__file__).parent / 'data' / 'Solar_Agent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'reports' / 'reward_solar.csv'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    run_test(input_file, output_file)
