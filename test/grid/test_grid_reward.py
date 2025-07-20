from pathlib import Path
import pandas as pd
import os

class GridAgent:
    def __init__(self, battery_soc_idx: int, action: int):
        """
        Initialize the Grid agent with battery state of charge (SoC)
        and the selected action (0: idle, 1: produce).
        """
        self.battery_soc_idx = battery_soc_idx
        self.action = action

    def calculate_reward(
        self,
        battery_soc_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the GridAgent based on the current system state
        and the selected action stored in self.action (0: idle, 1: produce).
        The reward encourages the grid to support the microgrid only when strictly
        necessary, considering battery availability, demand and total system output.

        Parameters:
            battery_soc_idx (int): Discretized index of the battery's state of charge (SOC).
            total_power_idx (int): Discretized index of the total power delivered by all agents.
            demand_power_idx (int): Discretized index of the total system demand.

        Returns:
            float: Reward signal guiding the grid agent.
        """
        
        # Reward adjustment parameters
        sigma = 5
        kappa = 18
        mu = 6
        nu = 6
        beta = 6
        xi = 5
        pho = 3
        tau = 4
        
        power_gap = total_power_idx - demand_power_idx  # >0 = surplus, <0 = shortage

        # Action = produce
        if self.action == 1:
            if power_gap < 0 and battery_soc_idx == 0:
                # Grid is supplying during shortage and battery is empty → necessary → strong reward
                return kappa * (demand_power_idx or 1)
            elif power_gap >= 0 and battery_soc_idx == 0:
                # The battery has no energy but the demand is satisfied → penalize
                return -sigma * (power_gap or 1)
            elif power_gap < 0 and battery_soc_idx > 0:
                # Grid is helping, but battery could have helped → mild penalty
                return -mu * battery_soc_idx
            else:
                # Grid produces despite sufficient system power → wasteful → strong penalty
                return -nu * (demand_power_idx or 1) * (power_gap or 1)
        # Action = idle
        else:
            if power_gap < 0 and battery_soc_idx == 0:
                # Grid is not supplying during shortage and battery is empty → necessary → strong penalty
                return -beta * (demand_power_idx or 1)
            elif power_gap >= 0 and battery_soc_idx == 0:
                # The battery has no energy but the demand is satisfied → reward
                return xi * (power_gap or 1)
            elif power_gap < 0 and battery_soc_idx > 0:
                # Grid is not helping, but battery could have helped → reward
                return pho * battery_soc_idx
            else:
                # The grid does not produce despite having sufficient system power and energy in the batteries → excess → strong reward
                return tau * (demand_power_idx or 1) * (power_gap or 1)

def run_test(input_path, output_path):
    df = pd.read_csv(input_path, delimiter=',')

    rewards = []
    for _, row in df.iterrows():
        agent = GridAgent(
            battery_soc_idx=row["battery_soc_idx"],
            action=row["action"]
        )
        reward = agent.calculate_reward(
            battery_soc_idx=row["battery_soc_idx"],
            total_power_idx=row["total_power_idx"],
            demand_power_idx=row["demand_power_idx"]
        )
        rewards.append(reward)

    df["reward"] = rewards
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Grid reward test completed. Results saved to: {output_path}")

if __name__ == "__main__":
    input_file = Path(__file__).parent / 'data' / 'GridAgent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'reports' / 'reward_grid.csv'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    run_test(input_file, output_file)
