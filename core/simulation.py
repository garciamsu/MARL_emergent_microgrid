import pandas as pd
from core.environment import MultiAgentEnv
from core.registry import create_agent, create_policy, create_reward
import core.policies
import core.rewards

EPSILON_MIN = 0

def instantiate_agents(config, env):
    agents = {}
    for agent_type, spec in config["agents"].items():
        for idx in range(spec["count"]):
            name = f"{agent_type}#{idx}"
            policy = create_policy(spec["policy"])
            reward_fn = create_reward(spec["reward"])
            #agent = create_agent(
            #    agent_type, env=env, policy=policy, reward_fn=reward_fn, **spec
            #)
            #agents[name] = agent
    return agents


def run_training(config):

    env = MultiAgentEnv(config)
    agents = instantiate_agents(config, env)

    print(agents)

    '''
    num_episodes = config["simulation"]["episodes"]
    epsilon = config["simulation"].get("epsilon", {}).get("start", 1.0)

    prev_q_tables = {}
    df_episode_metrics = pd.DataFrame()

    for ep in range(num_episodes):
        env.reset()
        evolution = []

        if ep == 0:
            for ag in agents.values():
                prev_q_tables[type(ag).__name__] = {
                    state: {a: 0.0 for a in ag.actions} for state in ag.q_table
                }
        else:
            for ag in agents.values():
                prev_q_tables[type(ag).__name__] = copy.deepcopy(ag.q_table)

        if ep != num_episodes - 1:
            env.scale_demand = random.uniform(0.2, config.get("MAX_SCALE", 2))
        else:
            env.scale_demand = 1

        for i in range(env.max_steps - 1):
            state = {name: ag.get_discretized_state(env, i) for name, ag in agents.items()}

            for ag in agents.values():
                ag.choose_action(state[type(ag).__name__], epsilon)

            next_state = {
                name: ag.get_discretized_state(env, i + 1) for name, ag in agents.items()
            }
            for ag in agents.values():
                reward = ag.calculate_reward(*state[type(ag).__name__])
                ag.update_q_table(
                    state[type(ag).__name__], ag.action, reward, next_state[type(ag).__name__]
                )

            evolution.append(dict(state))

        if num_episodes > 1:
            epsilon = max(EPSILON_MIN, 1 - (ep / (num_episodes - 1)))

        df = pd.DataFrame(evolution)
        df.to_csv(f"results/evolution/learning_{ep}.csv", index=False)

        iae = df["0"].abs().sum() if "0" in df.columns else 0
        var_dif = df["0"].var() if "0" in df.columns else 0

        q_norms = {
            type(ag).__name__: compute_q_diff_norm(ag.q_table, prev_q_tables[type(ag).__name__])
            for ag in agents.values()
        }

        row = {"Episode": ep, "IAE": iae, "Var_dif": var_dif}
        row.update({f"Q_Norm_{k}": v for k, v in q_norms.items()})
        df_episode_metrics = pd.concat([df_episode_metrics, pd.DataFrame([row])],
                                       ignore_index=True)

        df_episode_metrics.to_excel("results/metrics_episode.xlsx",
                                    index=False, engine="openpyxl")

        print(f"End of episode {ep+1}/{num_episodes} with epsilon {epsilon:.3f}")

    if "IAE" in df_episode_metrics.columns:
        plot_metric(df_episode_metrics,
                    field="IAE",
                    ylabel="Integral Absolute Error",
                    filename_svg="results/plots/IAE_over_episodes.svg")

    if "Var_dif" in df_episode_metrics.columns:
        plot_metric(df_episode_metrics,
                    field="Var_dif",
                    ylabel="Variance of dif",
                    filename_svg="results/plots/Var_dif_over_episodes.svg")
    '''
    return "agents"
