"""Shim de compatibilidad para tests.

Algunos tests hacen "from reward_debug import ..." esperando el módulo
en la ruta de importación. El código real está en `test/utils/reward_debug.py`.
Este archivo reexporta las funciones principales para mantener compatibilidad
sin modificar los tests.
"""
from test.utils.reward_debug import build_reward_from_config, explain_and_compute

__all__ = ["build_reward_from_config", "explain_and_compute"]
