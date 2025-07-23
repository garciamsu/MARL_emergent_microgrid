
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ruta del directorio que contiene los archivos CSV
folder_path = "./results/evolution/"
output_folder = "./results/stability"

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Inicializar lista para almacenar la recompensa media por episodio
episode_rewards = []

# Iterar sobre los archivos esperados (learning_0.csv hasta learning_1999.csv)
for i in range(2000):
    file_name = f"learning_{i}.csv"
    file_path = os.path.join(folder_path, file_name)

    # Verificar que el archivo existe
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)

        # Calcular recompensa total por fila
        df["reward_total"] = (
            df["reward_solar"] +
            df["reward_wind"] +
            df["reward_bat"] +
            df["reward_grid"] +
            df["reward_demand"]
        )

        # Calcular media móvil (rolling mean) con ventana de 24 pasos
        rolling_mean = df["reward_total"].rolling(window=24).mean()

        # Promedio de la media móvil para el episodio completo
        mean_reward = rolling_mean.mean()
        episode_rewards.append(mean_reward)
    else:
        # Agregar None si el archivo no existe
        episode_rewards.append(None)

# Crear DataFrame de resultados
results_df = pd.DataFrame({
    "episode": list(range(2000)),
    "rolling_mean_reward": episode_rewards
})

# Guardar CSV
csv_output_path = os.path.join(output_folder, "rolling_mean_rewards.csv")
results_df.to_csv(csv_output_path, index=False, encoding='utf-8')

# Graficar
plt.figure(figsize=(12, 6))
plt.plot(results_df["episode"], results_df["rolling_mean_reward"], label="Rolling Mean Reward (24h)")
plt.xlabel("Episode")
plt.ylabel("Rolling Mean Reward")
plt.title("Stability of Episodic Learning (MARL)")
plt.grid(True)
plt.legend()

# Guardar imagen
image_output_path = os.path.join(output_folder, "rolling_mean_rewards.png")
plt.savefig(image_output_path)
plt.close()
