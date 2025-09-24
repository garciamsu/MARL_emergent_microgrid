def compute_q_diff_norm(q_new, q_old):
    total = 0.0
    for state in q_new:
        for a, v in q_new[state].items():
            total += abs(v - q_old.get(state, {}).get(a, 0.0))
    return total


def check_stability(df_metrics, iae_threshold):
    recent = df_metrics.tail(200)
    return {
        "IAE_mean": recent["IAE"].mean(),
        "Var_mean": recent["Var_dif"].mean(),
        "IAE_stable": recent["IAE"].mean() <= iae_threshold,
        "Var_stable": recent["Var_dif"].mean() <= recent["Var_dif"].median() * 1.1,
    }
