"""Smoke test to ensure one training episode runs without errors."""
from configs.loader import load_config
from core.simulation import run_training


def test_single_episode_runs():
    """
    Verifies that a single training episode executes without errors.
    This ensures the basic training pipeline is functional.
    """
    # Load configuration
    cfg = load_config("configs/default.yaml")
    cfg["simulation"]["episodes"] = 1

    # Run training for a single episode
    agents, results = run_training(cfg)

    # Assertions to validate the results
    assert agents, "Agents dictionary should not be empty."
    assert len(results) == 1, "Exactly one episode result is expected."
    assert not results[0].empty, "Episode dataframe should contain steps."

    print("Smoke test passed: Single episode ran successfully.")
