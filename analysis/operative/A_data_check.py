#!/usr/bin/env python3
"""
A_data_check.py

Valida datasets y columnas necesarias para las métricas de análisis.
Integra validación de extracción de demanda y verificación de valores.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Añadir raíz al path para imports
# This script lives under analysis/operative/, so repo root is two levels up.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.loader import load_config
from core.csv_handler import read_dataset_csv, read_result_csv



def validate_dataset(dataset_path: str) -> bool:
    """
    Valida que el dataset contenga las columnas necesarias.
    
    Args:
        dataset_path: Ruta al archivo CSV del dataset.
    
    Returns:
        True si es válido, False en caso contrario.
    """
    print(f"\n📊 Validando dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ ERROR: Dataset no encontrado en {dataset_path}")
        return False
    
    try:
        df = read_dataset_csv(dataset_path)
    except Exception as e:
        print(f"❌ ERROR leyendo dataset: {e}")
        return False
    
    print(f"   Filas: {len(df)}")
    print(f"   Columnas: {list(df.columns)}")
    
    # Columnas requeridas - buscar con sufijos si es necesario
    required_base = ["demand"]
    optional_patterns = ["solar_power", "wind_power", "solar", "wind"]
    
    missing = [c for c in required_base if c not in df.columns]
    
    if missing:
        print(f"❌ ERROR: Falta columna requerida 'demand'")
        return False
    
    # Verificar que haya al menos una columna de generación renovable
    renewable_cols = [c for c in df.columns if any(p in c for p in optional_patterns)]
    
    if not renewable_cols:
        print(f"❌ ERROR: No se encontraron columnas de generación renovable (solar/wind)")
        return False
    
    print(f"✅ Dataset válido con columnas encontradas:")
    print(f"   - Demanda: demand")
    print(f"   - Renovables: {renewable_cols}")
    
    # Verificar rangos
    if df["demand"].min() < 0:
        print(f"⚠️  ADVERTENCIA: Demanda tiene valores negativos (mín: {df['demand'].min()})")
    
    # Verificar rangos de renovables dinámicamente
    for col in renewable_cols:
        if df[col].min() < 0:
            print(f"⚠️  ADVERTENCIA: {col} tiene valores negativos")
    
    print(f"\n📈 Rangos de valores:")
    print(f"   Demanda: [{df['demand'].min():.2f}, {df['demand'].max():.2f}] W")
    for col in renewable_cols:
        print(f"   {col}: [{df[col].min():.2f}, {df[col].max():.2f}] W")
    
    return True


def validate_evolution_columns(pattern: str = "results/evolution/episode_*.csv") -> bool:
    """
    Valida que los CSVs de episodios tengan las columnas necesarias para métricas.
    
    Args:
        pattern: Patrón glob para buscar archivos CSV de episodios.
    
    Returns:
        True si son válidos, False en caso contrario.
    """
    import glob
    
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"\n⚠️  ADVERTENCIA: No se encontraron episodios en {pattern}")
        print("   Ejecuta primero B_run_training.py para generar datos.")
        return False
    
    print(f"\n📂 Validando columnas en episodios ({len(files)} archivos)...")
    
    # Validar primer episodio
    sample_file = files[0]
    try:
        df = read_result_csv(sample_file)
    except Exception as e:
        print(f"❌ ERROR leyendo {sample_file}: {e}")
        return False
    
    # Columnas requeridas para métricas
    required_cols = [
        "episode",
        "step",
        "env_energy_balance",
        "env_total_renewable",
        "env_demand_power",
        "env_total_power",
        "power_grid#0",
        "reward_solar#0",
        "reward_wind#0",
        "reward_battery#0",
        "reward_grid#0",
        "reward_load#0",
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"❌ ERROR: Faltan columnas requeridas en episodios: {missing}")
        print(f"   Columnas encontradas: {list(df.columns)}")
        return False
    
    print(f"✅ Episodios válidos con columnas requeridas.")
    
    # Verificar que env_demand_power > 0 siempre (excluyendo step=-1, que es el estado inicial)
    df_active = df[df["step"] >= 0]  # Excluir estado inicial (step=-1)
    zero_demand = (df_active["env_demand_power"] == 0).sum()
    if zero_demand > 0:
        print(f"⚠️  ADVERTENCIA: {zero_demand} filas con env_demand_power = 0 en pasos activos (muestra: {sample_file})")
    else:
        print(f"✅ env_demand_power > 0 en todas las filas de pasos activos (muestra: {sample_file})")
    
    return True


def main():
    """Punto de entrada principal del script de validación."""
    print("="*80)
    print("🔍 A_data_check.py - Validación de Datos para Análisis")
    print("="*80)
    
    # Cargar configuración
    try:
        config = load_config()
    except Exception as e:
        print(f"\n❌ ERROR cargando configuración: {e}")
        sys.exit(1)
    
    # 1. Validar dataset
    dataset_name = config["simulation"]["dataset"]
    # Si el nombre ya incluye .csv, no lo duplicamos
    if dataset_name.endswith('.csv'):
        dataset_path = os.path.join("assets", "datasets", dataset_name)
    else:
        dataset_path = os.path.join("assets", "datasets", f"{dataset_name}.csv")
    
    dataset_valid = validate_dataset(dataset_path)
    
    # 2. Validar columnas de episodios (si existen)
    episodes_valid = validate_evolution_columns()
    
    # Resumen
    print("\n" + "="*80)
    print("📋 RESUMEN DE VALIDACIÓN")
    print("="*80)
    print(f"   Dataset válido: {'✅' if dataset_valid else '❌'}")
    print(f"   Episodios válidos: {'✅' if episodes_valid else '⚠️  (sin datos aún)'}")
    
    if not dataset_valid:
        print("\n❌ VALIDACIÓN FALLIDA: Corrige el dataset antes de continuar.")
        sys.exit(1)
    
    if not episodes_valid:
        print("\n⚠️  Sin episodios para validar. Ejecuta B_run_training.py primero.")
    else:
        print("\n✅ VALIDACIÓN EXITOSA: Todo listo para análisis.")
    
    print("="*80)


if __name__ == "__main__":
    main()
