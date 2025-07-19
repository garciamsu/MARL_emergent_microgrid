from pathlib import Path
import pandas as pd
import os
import numpy as np

class BatteryAgent:
    def __init__(self, idx: int, action: int):
        """
        Initialize the battery agent with its current state of charge (SoC) index
        and the selected action to evaluate.

        :param idx: Discretized SoC index (0 to 4)
        :param action: Selected action (0: idle, 1: charge, 2: discharge)
        """
        self.idx = idx
        self.action = action

    def calculate_reward(
        self,
        renewable_potential_idx: int,
        total_power_idx: int,
        demand_power_idx: int
    ) -> float:
        """
        Computes the reward for the battery agent based on the system's renewable availability,
        power balance, and the agent's current action.

        Reward is assigned according to the following logic:
        - Discharging is rewarded when there is a power deficit and SoC > 0.
        - Charging is rewarded when there is excess renewable energy.
        - Idle is strongly penalized as it is not a valid behavior in this system.
        - Discharging with SoC = 0 is heavily penalized.
        - Charging without renewable surplus is slightly penalized.

        Parameters:
            renewable_potential_idx (int): Discretized index of renewable power available (0 to 6).
            total_power_idx (int): Discretized index of total power generated (0 to 6).
            demand_power_idx (int): Discretized index of system demand (0 to 6).

        Returns:
            float: A scalar reward guiding the learning of the agent.
        """

        # Reward adjustment parameters
        sigma = 8   # Heavy penalty for invalid discharge
        kappa = 12  # Strong reward for helping during deficit
        mu = 5      # Moderate penalty for discharging in surplus
        nu = 10     # Reward for charging with excess renewables
        beta = 5    # Light penalty for charging without surplus
        xi = 5      # Strong penalty for idling

        # Calculate power gap: positive → surplus, negative → deficit
        power_gap = total_power_idx - demand_power_idx

        # Action: discharge
        if self.action == 2:
            if self.idx == 0:
                # Discharge not possible at 0% SoC → heavy penalty
                return -sigma * max(demand_power_idx, 1)
            elif power_gap < 0:
                # Discharging during deficit → reward scales with deficit and SoC
                critical_deficit = abs(power_gap) >= 4 and self.idx >= 3
                factor = 1.3 if critical_deficit else 1.0
                return kappa * np.tanh(abs(power_gap)) * self.idx * factor

            else:
                if power_gap == 0:
                    return -mu  # small fixed penalty
                else:
                    # Discharging during surplus or equilibrium → penalty
                    return -mu * max(power_gap, 1)  # keep original for surplus

        # Action: charge
        elif self.action == 1:
            if renewable_potential_idx > demand_power_idx:
                # Charging when renewables exceed demand → reward
                soc_penalty = 0.5 if self.idx == 4 else 1
                return nu * renewable_potential_idx * soc_penalty
            
            else:
                # Charging without surplus → light penalty
                soc_load_penalty = 1 + 0.3 * self.idx 
                return -beta * max(demand_power_idx, 1) * soc_load_penalty


        # Action: idle
        else:
            # Idling is discouraged in all scenarios
            return -xi

def run_test(input_path, output_path):
    df = pd.read_csv(input_path, delimiter=';')

    rewards = []
    for _, row in df.iterrows():
        agent = BatteryAgent(idx=row["battery_soc_idx"], action=row["action"])
        reward = agent.calculate_reward(
            renewable_potential_idx=row["renewable_potential_idx"],
            total_power_idx=row["total_power_idx"],
            demand_power_idx=row["demand_power_idx"]
        )
        rewards.append(reward)

    df["reward"] = rewards
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Battery reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":

    input_file = Path(__file__).parent / 'data' / 'Battery_Agent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'reports' / 'reward_battery.csv'

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    run_test(input_file, output_file)