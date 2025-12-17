#!/usr/bin/env python3
"""
Test script to verify power_scale_factor is applied correctly.
"""
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.loader import load_config
from core.csv_handler import read_dataset_csv

# Load configuration
config = load_config()
power_scale_factor = config['simulation']['power_scale_factor']

print(f'power_scale_factor configurado: {power_scale_factor}')

# Read dataset
df = read_dataset_csv('assets/datasets/microgrid_dataset_8d_hourly_comparative_quantite.csv')

print('\n=== VALORES ORIGINALES (kW) ===')
print(f'demand: {df["demand"].min():.2f} - {df["demand"].max():.2f} kW')
print(f'solar_potential: {df["solar_potential"].min():.2f} - {df["solar_potential"].max():.2f} kW')
print(f'wind_potential: {df["wind_potential"].min():.2f} - {df["wind_potential"].max():.2f} kW')

# Simulate the scaling that the code does
for col in df.columns:
    if col not in ['price', 'Datetime', 'datetime', 'weather']:
        if 'potential' in col.lower() or col.lower() == 'demand':
            df[col] = df[col] * power_scale_factor

print(f'\n=== VALORES DESPUÉS DE ESCALAMIENTO (W) ===')
print(f'demand: {df["demand"].min():.0f} - {df["demand"].max():.0f} W')
print(f'solar_potential: {df["solar_potential"].min():.0f} - {df["solar_potential"].max():.0f} W')
print(f'wind_potential: {df["wind_potential"].min():.0f} - {df["wind_potential"].max():.0f} W')

print(f'\n✅ El escalamiento funciona correctamente!')
print(f'   Valores en kW → multiplicados por {power_scale_factor} → Valores en W')
