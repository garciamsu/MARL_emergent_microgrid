import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración del periodo de tiempo y frecuencia
start_time = datetime(2025, 1, 8, 0, 0, 0)  # Fecha inicial
end_time = start_time + timedelta(days=7)  # Una semana
frequency = "1min"  # Frecuencia de los datos

# Generación del índice de tiempo
timestamps = pd.date_range(start=start_time, end=end_time, freq=frequency)

# Simulación de variables
np.random.seed(42)  # Para reproducibilidad

# Irradiancia solar (W/m²): Variación diurna, máximos en el mediodía
irradiance = [
    max(0, 1000 * np.sin((time.hour + time.minute / 60) / 24 * 2 * np.pi - np.pi / 2))
    + np.random.normal(0, 50)  # Añadir ruido
    for time in timestamps
]

# Costo de mercado ($/kWh): Tendencia diaria con picos nocturnos y variaciones aleatorias
market_cost = [
    0.10 + 0.05 * np.sin((time.hour + time.minute / 60) / 24 * 2 * np.pi)  # Variación diaria
    + np.random.normal(0, 0.01)  # Ruido
    for time in timestamps
]

# Perfiles de consumo (kW): Ciclos de consumo con variabilidad por hora
demand = [
    5 + 3 * np.sin((time.hour + time.minute / 60) / 24 * 2 * np.pi)  # Demanda base
    + np.random.normal(0, 0.5)  # Ruido
    for time in timestamps
]

# Creación del DataFrame
data = pd.DataFrame({
    "Timestamp": timestamps,
    "Irradiance (W/m²)": irradiance,
    "Market Cost ($/kWh)": market_cost,
    "Commercial Demand (kW)": demand
})

# Guardar como archivo Excel
file_path = "/mnt/data/microgrid_scenario_data.xlsx"
data.to_excel(file_path, index=False)

file_path
