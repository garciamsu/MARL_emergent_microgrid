"""Generate a Q-table JSON from a wind agent state-action-reward CSV.

This script creates a JSON mapping from state tuples to action->reward for use
in offline lookup or analysis.
"""

import json
from pathlib import Path
import pandas as pd


def generate_q_table(input_csv_path: str, output_json_path: str):
    """Read CSV and export a Q-table JSON.

    Args:
        input_csv_path: path to the CSV with columns including 'action' and 'reward'.
        output_json_path: destination JSON path.
    """

    data_frame = pd.read_csv(input_csv_path, delimiter=",", engine="python")

    data_frame["state"] = list(
        zip(
            data_frame.get("wind_potential_idx", data_frame.get("renewable_potential_idx")),
            data_frame.get("total_power_idx"),
            data_frame.get("demand_power_idx"),
        )
    )

    q_table = {}
    for _, row in data_frame.iterrows():
        state = str(row["state"])
        action = str(int(row["action"]))
        reward = float(row["reward"])
        if state not in q_table:
            q_table[state] = {}
        q_table[state][action] = reward

    with open(output_json_path, "w", encoding="utf-8") as file_handle:
        json.dump(q_table, file_handle, indent=2)

    print(f"✅ Q-table saved as: {output_json_path}")


if __name__ == "__main__":
    input_file = Path(__file__).parent / "output" / "reward_wind.csv"
    output_file = Path(__file__).parent / "output" / "wind_q_table.json"
    generate_q_table(input_file, output_file)
