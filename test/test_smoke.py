"""Smoke test to ensure one training episode runs without errors."""
from configs.loader import load_config
from core.simulation import run_training


def test_single_episode_runs():
    cfg = load_config("configs/default.yaml")
    cfg["simulation"]["episodes"] = 1
    agents, results = run_training(cfg)
    assert agents, "Agents dict should not be empty"
    assert len(results) == 1, "Exactly one episode result expected"
    assert not results[0].empty, "Episode dataframe should contain steps"
