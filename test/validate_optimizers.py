#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_optimizers.py
======================

Script de validación para verificar que todos los optimizadores
de hiperparámetros tienen las dependencias correctas instaladas,
especialmente scikit-optimize.

Uso:
    python test/validate_optimizers.py
"""

import sys
import os

# Agregar directorio raíz al path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

def validate_skopt():
    """Valida que scikit-optimize esté instalado."""
    try:
        import skopt
        from skopt import gp_minimize
        from skopt.space import Real
        print(f"✓ scikit-optimize v{skopt.__version__} instalado correctamente")
        return True
    except ImportError as e:
        print(f"✗ ERROR: scikit-optimize no está instalado")
        print(f"  Instalar con: pip install scikit-optimize")
        return False


def validate_optimizer(agent_name, script_path):
    """Valida un script optimizador específico."""
    print(f"\n{'='*60}")
    print(f"Validando: {agent_name.upper()}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    # Verificar que el archivo existe
    if not os.path.exists(script_path):
        print(f"✗ ERROR: Archivo no encontrado")
        return False
    
    print(f"✓ Archivo encontrado")
    
    # Intentar importar las dependencias que usaría el script
    try:
        # Imports comunes
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        print(f"✓ Dependencias básicas (pandas, numpy, matplotlib, tqdm)")
        
        # scikit-optimize (opcional pero recomendado)
        from skopt import gp_minimize
        from skopt.space import Real
        print(f"✓ scikit-optimize (optimización bayesiana disponible)")
        
        # Imports específicos del proyecto
        from test.optimizer_utils import compute_total_margin
        print(f"✓ test.optimizer_utils")
        
        # Import de reward específica del agente
        reward_classes = {
            'battery': 'DefaultBatteryReward',
            'grid': 'DefaultGridReward',
            'load': 'DefaultLoadReward',
            'solar': 'DefaultSolarReward',
            'wind': 'DefaultWindReward'
        }
        
        if agent_name in reward_classes:
            reward_class = reward_classes[agent_name]
            exec(f"from core.rewards import {reward_class}")
            print(f"✓ core.rewards.{reward_class}")
        
        print(f"\n✓✓✓ {agent_name.upper()}: TODAS LAS VALIDACIONES PASADAS ✓✓✓")
        return True
        
    except ImportError as e:
        print(f"✗ ERROR de importación: {e}")
        return False
    except Exception as e:
        print(f"✗ ERROR inesperado: {e}")
        return False


def main():
    """Función principal de validación."""
    print("="*60)
    print(" VALIDACIÓN DE OPTIMIZADORES DE HIPERPARÁMETROS")
    print("="*60)
    
    # Validar scikit-optimize primero
    skopt_ok = validate_skopt()
    
    # Definir optimizadores a validar
    optimizers = {
        'battery': os.path.join(PROJECT_ROOT, 'test/battery/B_battery_hyperparam_optimizer.py'),
        'grid': os.path.join(PROJECT_ROOT, 'test/grid/B_grid_hyperparam_optimizer.py'),
        'load': os.path.join(PROJECT_ROOT, 'test/load/B_load_hyperparam_optimizer.py'),
        'solar': os.path.join(PROJECT_ROOT, 'test/solar/B_solar_hyperparam_optimizer.py'),
        'wind': os.path.join(PROJECT_ROOT, 'test/wind/B_wind_hyperparam_optimizer.py'),
    }
    
    # Validar cada optimizador
    results = {}
    for agent_name, script_path in optimizers.items():
        results[agent_name] = validate_optimizer(agent_name, script_path)
    
    # Resumen final
    print(f"\n\n{'='*60}")
    print(" RESUMEN DE VALIDACIÓN")
    print(f"{'='*60}")
    
    for agent_name, status in results.items():
        status_str = "✓ PASS" if status else "✗ FAIL"
        print(f"{agent_name:10s} : {status_str}")
    
    all_passed = all(results.values())
    
    print(f"{'='*60}")
    if all_passed and skopt_ok:
        print("✓✓✓ VALIDACIÓN COMPLETA: TODOS LOS OPTIMIZADORES OK ✓✓✓")
        print("\nLos siguientes métodos de optimización están disponibles:")
        print("  - random_search   : Búsqueda aleatoria")
        print("  - bayesian        : Optimización bayesiana (scikit-optimize)")
        print("  - evolutionary    : Algoritmo evolutivo")
    else:
        print("✗✗✗ VALIDACIÓN FALLIDA: REVISAR ERRORES ARRIBA ✗✗✗")
        if not skopt_ok:
            print("\n⚠ WARNING: Sin scikit-optimize, solo estará disponible:")
            print("  - random_search")
            print("  - evolutionary")
    print(f"{'='*60}\n")
    
    return 0 if (all_passed and skopt_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
