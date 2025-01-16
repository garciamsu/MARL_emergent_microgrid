import matplotlib.pyplot as plt

# Configuración de las gráficas
plt.figure(figsize=(15, 10))

# Irradiancia solar
plt.subplot(3, 1, 1)
plt.plot(data["Timestamp"], data["Irradiance (W/m²)"], color="orange", label="Irradiance (W/m²)")
plt.title("Irradiance Solar (W/m²)")
plt.ylabel("W/m²")
plt.grid(True)
plt.legend()

# Costo de mercado
plt.subplot(3, 1, 2)
plt.plot(data["Timestamp"], data["Market Cost ($/kWh)"], color="blue", label="Market Cost ($/kWh)")
plt.title("Costo de Mercado ($/kWh)")
plt.ylabel("$/kWh")
plt.grid(True)
plt.legend()

# Demanda comercial
plt.subplot(3, 1, 3)
plt.plot(data["Timestamp"], data["Commercial Demand (kW)"], color="green", label="Commercial Demand (kW)")
plt.title("Demanda Comercial (kW)")
plt.xlabel("Tiempo")
plt.ylabel("kW")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
