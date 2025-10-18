"""
Solar reward testing script.
This script evaluates the reward calculation logic for the solar agent.
"""
from pathlib import Path
import pandas as pd
from core.environment import SolarAgent  # Importar la clase desde la aplicación principal

def run_test(input_path: Path, output_path: Path):
    """
    Runs the reward calculation test for the solar agent using input data.

    :param input_path: Path to the input CSV file.
    :param output_path: Path to save the output CSV file with rewards.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    data_frame = pd.read_csv(input_path, delimiter=';')
    rewards = []

    for _, row in data_frame.iterrows():
        agent = SolarAgent(idx=row["solar_power_idx"], action=row["action"])
        reward = agent.calculate_reward(
            renewable_potential_idx=row["renewable_potential_idx"],
            total_power_idx=row["total_power_idx"],
            demand_power_idx=row["demand_power_idx"]
        )
        rewards.append(reward)

    data_frame["reward"] = rewards
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_frame.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Solar reward test completed. Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = Path(__file__).parent / 'data' / 'Solar_Agent_Reward_Table.csv'
    output_file = Path(__file__).parent / 'reports' / 'reward_solar.csv'

    try:
        run_test(input_file, output_file)
    except FileNotFoundError as file_err:
        print(f"File error: {file_err}")
    except ValueError as value_err:
        print(f"Value error: {value_err}")
