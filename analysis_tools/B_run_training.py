#!/usr/bin/env python3
"""
B_run_training.py

Wrapper para ejecutar main.py con limpieza automática delegada.
Lee toda la configuración desde configs/default.yaml.
"""

import os
import sys
import subprocess
from pathlib import Path

# Añadir raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.loader import load_config


def main():
    """Ejecuta el entrenamiento principal."""
    print("="*80)
    print("🚀 B_run_training.py - Lanzar Entrenamiento")
    print("="*80)
    
    # Cargar configuración para mostrar parámetros clave
    try:
        config = load_config()
    except Exception as e:
        print(f"\n❌ ERROR cargando configuración: {e}")
        sys.exit(1)
    
    # Mostrar parámetros clave
    print(f"\n📋 Parámetros de Simulación (desde configs/default.yaml):")
    print(f"   Episodios: {config['simulation']['episodes']}")
    print(f"   Dataset: {config['simulation']['dataset']}")
    print(f"   Seed: {config['simulation']['seed']}")
    print(f"   dt_h: {config['simulation']['dt_h']}")
    
    epsilon_cfg = config['simulation']['epsilon']
    print(f"   Epsilon: schedule={epsilon_cfg['schedule']}, start={epsilon_cfg['start']}, end={epsilon_cfg['end']}")
    
    print(f"\n🧹 Limpieza de results/: delegada a main.py (analysis_tools.utils.clear_directories)")
    print(f"\n▶️  Ejecutando main.py...\n")
    
    # Ejecutar main.py con Python del entorno actual
    main_path = Path(__file__).parent.parent / "main.py"
    
    try:
        result = subprocess.run(
            [sys.executable, str(main_path)],
            cwd=str(Path(__file__).parent.parent),
            check=True
        )
        
        if result.returncode == 0:
            print("\n✅ Entrenamiento completado exitosamente.")
            print("   Resultados en: results/evolution/ y results/logs/")
        else:
            print(f"\n⚠️  main.py finalizó con código {result.returncode}")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR ejecutando main.py: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ ERROR: No se encontró main.py en {main_path}")
        sys.exit(1)
    
    print("="*80)


if __name__ == "__main__":
    main()
