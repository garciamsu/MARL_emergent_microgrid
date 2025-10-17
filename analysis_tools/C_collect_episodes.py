#!/usr/bin/env python3
"""
C_collect_episodes.py

Consolida todos los CSVs de episodios individuales en un solo archivo agregado
con métricas por episodio para facilitar análisis posteriores.
"""

import os
import sys
from pathlib import Path

# Añadir raíz al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis_tools.utils import load_all_episodes_metrics
from configs.loader import load_config


def main():
    """Consolida episodios en un único CSV agregado."""
    print("="*80)
    print("📦 C_collect_episodes.py - Consolidar Episodios")
    print("="*80)
    
    # Cargar configuración
    try:
        config = load_config()
    except Exception as e:
        print(f"\n❌ ERROR cargando configuración: {e}")
        sys.exit(1)
    
    # Patrón de búsqueda
    pattern = "results/evolution/episode_*.csv"
    
    print(f"\n🔍 Buscando episodios en: {pattern}")
    
    # Cargar y consolidar
    try:
        df_consolidated = load_all_episodes_metrics(pattern=pattern)
    except Exception as e:
        print(f"\n❌ ERROR consolidando episodios: {e}")
        sys.exit(1)
    
    if df_consolidated.empty:
        print(f"\n⚠️  No se encontraron episodios para consolidar.")
        print("   Ejecuta primero B_run_training.py para generar datos.")
        sys.exit(1)
    
    print(f"✅ {len(df_consolidated)} episodios consolidados.")
    
    # Guardar consolidado
    output_path = "results/evolution/episodes_consolidated.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        df_consolidated.to_csv(output_path, index=False)
        print(f"\n💾 Guardado en: {output_path}")
        print(f"   Columnas: {list(df_consolidated.columns)}")
        print(f"   Filas: {len(df_consolidated)}")
    except Exception as e:
        print(f"\n❌ ERROR guardando consolidado: {e}")
        sys.exit(1)
    
    # Mostrar primeras filas
    print(f"\n📋 Primeras 5 filas del consolidado:")
    print(df_consolidated.head())
    
    print("\n" + "="*80)
    print("✅ Consolidación completada.")
    print("   Siguiente paso: D_compute_metrics.py")
    print("="*80)


if __name__ == "__main__":
    main()
