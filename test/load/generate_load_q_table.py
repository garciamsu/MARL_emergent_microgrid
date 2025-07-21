import pandas as pd
import json
from pathlib import Path

def generate_q_table(input_csv_path: str, output_json_path: str):
    """
    Reads a CSV file with state-action-reward data from LoadAgent
    and converts it into a Q-table format as JSON.

    Parameters:
    - input_csv_path: Path to the reward_load.csv file.
    - output_json_path: Path to save the resulting JSON file.
    """
    # Load the CSV file
    df = pd.read_csv(input_csv_path, delimiter=',', engine='python')

    # Define each state as a tuple of the input variables relevant for LoadAgent
    df['state'] = list(zip(
        df['battery_soc_idx'],
        df['renewable_potential_idx'],
        df['comfort_idx']
    ))

    # Build the Q-table dictionary: {state_str: {action_str: reward}}
    q_table = {}
    for _, row in df.iterrows():
        state = str(row['state'])  # Convert tuple to string for JSON key
        action = str(int(row['action']))
        reward = float(row['reward'])

        if state not in q_table:
            q_table[state] = {}
        q_table[state][action] = reward

    # Export as JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(q_table, f, indent=2)

    print(f"✅ Q-table saved as: {output_json_path}")

if __name__ == "__main__":
    input_file = Path(__file__).parent / 'reports' / 'reward_load.csv'
    output_file = Path(__file__).parent / 'reports' / 'load_q_table.json'

    generate_q_table(input_file, output_file)
