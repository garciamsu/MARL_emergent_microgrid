import pandas as pd
import pytest

from core.environment import MultiAgentEnv


@pytest.mark.parametrize(
    "solar_count, wind_count, expected_solar, expected_wind, expected_total",
    [
        (0, 0, 0.0, 0.0, 0.0),
        (1, 0, 10.0, 0.0, 10.0),
        (0, 1, 0.0, 20.0, 20.0),
        (1, 1, 10.0, 20.0, 30.0),
    ],
)
def test_renewable_potential_respects_agent_counts(
    monkeypatch,
    solar_count,
    wind_count,
    expected_solar,
    expected_wind,
    expected_total,
):
    synthetic = pd.DataFrame(
        {
            "demand": [5.0],
            "price": [10.0],
            "solar_potential": [10.0],
            "wind_potential": [20.0],
        }
    )

    # Avoid filesystem dependency by bypassing CSV read.
    import core.environment as env_mod

    monkeypatch.setattr(env_mod, "read_dataset_csv", lambda _path: synthetic.copy())

    cfg = {
        "simulation": {
            "dataset": "ignored.csv",
            "dt_h": 1.0,
            "power_scale_factor": 1.0,
            "max_power": 100.0,
            "max_price": 100.0,
        },
        "discretization": {"delta_bins": 3, "power_bins": 2, "price_bins": 2, "soc_bins": 3},
        "agents": {
            "solar": {"count": solar_count},
            "wind": {"count": wind_count},
            "load": {"count": 0, "limits": {"comfort_threshold": 50}},
        },
    }

    env = MultiAgentEnv(cfg)
    env.episode_data = synthetic.copy()

    env.load_timestep_data(0)

    assert env.solar_potential == expected_solar
    assert env.wind_potential == expected_wind
    assert env.renewable_potential == expected_total
    assert env.renewable_potential == env.solar_potential + env.wind_potential
