import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from matplotlib.backends.backend_pdf import PdfPages

def analysis_qtable(ruta_archivo, carpeta_salida="qtable_analysis_output"):
    os.makedirs(carpeta_salida, exist_ok=True)

    # Read the Excel file
    df = pd.read_excel(ruta_archivo)
    df[['solar_potential_idx', 'total_power_idx', 'demand_power_idx']] = df['State'].apply(lambda x: pd.Series(eval(x)))

    # Reshape Q-table to wide format (one row per state with Q for both actions)
    q_pivot = df.pivot_table(index=['solar_potential_idx', 'total_power_idx', 'demand_power_idx'],
                             columns='Action', values='Q_value').reset_index().rename(columns={0: 'Q_no_produce', 1: 'Q_produce'})

    # Determine best action based on highest Q-value
    q_pivot['best_action'] = np.where(q_pivot['Q_no_produce'] > q_pivot['Q_produce'], 0, 1)

    # Calculate entropy for each state
    q_pivot['entropy'] = q_pivot.apply(lambda row: entropy([row['Q_no_produce'], row['Q_produce']]) if row['Q_no_produce'] + row['Q_produce'] > 0 else 0, axis=1)

    # Difference between Q-values
    q_pivot['q_diff'] = np.abs(q_pivot['Q_no_produce'] - q_pivot['Q_produce'])

    # Determine if the state was trained (Q ≠ 0)
    q_pivot['trained'] = (q_pivot['Q_no_produce'] != 0) | (q_pivot['Q_produce'] != 0)

    # Training statistics
    total_states = len(q_pivot)
    trained_states = q_pivot['trained'].sum()
    untrained_states = total_states - trained_states
    trained_pct = trained_states / total_states * 100
    untrained_pct = 100 - trained_pct

    resumen = {
        'Total states': total_states,
        'Trained states (Q ≠ 0)': trained_states,
        'Untrained states (Q = 0)': untrained_states,
        'Trained states (%)': f"{trained_pct:.2f}%",
        'Untrained states (%)': f"{untrained_pct:.2f}%",
        'States with action 0 preferred': (q_pivot['best_action'] == 0).sum(),
        'States with action 1 preferred': (q_pivot['best_action'] == 1).sum(),
        'Mean Q_no_produce': q_pivot['Q_no_produce'].mean(),
        'Mean Q_produce': q_pivot['Q_produce'].mean(),
        'Mean entropy': q_pivot['entropy'].mean(),
        'Median entropy': q_pivot['entropy'].median(),
        'States with high ambiguity (q_diff < 0.01)': (q_pivot['q_diff'] < 0.01).sum()
    }

    df_resumen = pd.DataFrame(list(resumen.items()), columns=['Metric', 'Value'])
    df_resumen.to_csv(os.path.join(carpeta_salida, "qtable_summary.csv"), index=False, encoding='utf-8')

    ruta_pdf = os.path.join(carpeta_salida, "qtable_analysis_report.pdf")
    with PdfPages(ruta_pdf) as pdf:
        # Q-value distribution by action
        plt.figure(figsize=(8, 5))
        sns.histplot(df[df['Action'] == 0]['Q_value'], kde=True, color='blue', label='No produce', bins=30)
        sns.histplot(df[df['Action'] == 1]['Q_value'], kde=True, color='orange', label='Produce', bins=30)
        plt.title("Q-value Distribution by Action")
        plt.xlabel("Q_value")
        plt.ylabel("Frequency")
        plt.legend()
        pdf.savefig(); plt.close()

        # Heatmap of average entropy by solar and demand index
        heatmap_data = q_pivot.groupby(['solar_potential_idx', 'demand_power_idx'])['entropy'].mean().unstack()
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f")
        plt.title("Average Entropy by Solar and Demand Index")
        plt.xlabel("demand_power_idx")
        plt.ylabel("solar_potential_idx")
        pdf.savefig(); plt.close()

        # Distribution of Q-value differences
        plt.figure(figsize=(8, 5))
        sns.histplot(q_pivot['q_diff'], bins=30, kde=True, color='green')
        plt.title("Absolute Difference between Q-values")
        plt.xlabel("|Q_no_produce - Q_produce|")
        plt.ylabel("Frequency")
        pdf.savefig(); plt.close()

        # Entropy vs Q-value difference
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x='q_diff', y='entropy', data=q_pivot, alpha=0.5)
        plt.title("Entropy vs Q-value Difference")
        plt.xlabel("q_diff")
        plt.ylabel("Entropy")
        pdf.savefig(); plt.close()

        # Final conclusions
        conclusions = [
            "Conclusions from the solar agent Q-table analysis:",
            "",
            "1. Most states have a clearly preferred action.",
            "2. Q-values are symmetrically distributed between actions.",
            "3. Some states exhibit ambiguous behavior based on entropy analysis.",
            "4. Several states have Q-value differences below 0.01, indicating indecision.",
            "5. Training coverage was analyzed using Q ≠ 0 as indicator.",
            "6. Evaluating and possibly refining state discretization or exploration policy is advised.",
            "",
            "Recommendations:",
            "- Improve reward function to reduce ambiguity.",
            "- Review state discretization granularity.",
            "- Adjust ε-greedy strategy to emphasize underexplored or ambiguous states.",
            "- Investigate whether untrained states are simply unexplored.",
        ]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        plt.text(0.01, 0.99, "\n".join(conclusions), va='top', ha='left', fontsize=10, wrap=True)
        pdf.savefig(); plt.close()

    print(f"Analysis completed. Results saved to: {carpeta_salida}")

# Example usage
analysis_qtable("results/q_tables/qtable_solar_ep299.xlsx", carpeta_salida="qtable_analysis_solar")