"""Quick self-check script for the MARL microgrid project.

Runs a minimal training episode to verify that core components load and
execute without raising exceptions. Intended for fast CI or pre-commit sanity.
"""
from __future__ import annotations

from configs.loader import load_config
from core.simulation import run_training


def main():  # pragma: no cover - convenience script
    cfg = load_config("configs/default.yaml")
    # Force minimal episodes for speed
    cfg["simulation"]["episodes"] = 1
    run_training(cfg)
    print("Self-check passed: 1 episode executed.")


if __name__ == "__main__":  # pragma: no cover
    main()
