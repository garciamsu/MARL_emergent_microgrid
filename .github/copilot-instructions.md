¡Absolutamente\! Aquí tienes las instrucciones clave para **GitHub Copilot**, estrictamente formateadas con encabezados estilizados y listas para copiar y pegar.

## 📌 Contexto Global del Proyecto 🌍

Este repositorio implementa un sistema **Multi-Agent Reinforcement Learning (MARL)** para la operación de una microgrid.

  * **Agentes:** Solar, Wind, Battery, Grid, Load.
  * **Método:** Q-learning tabular.
  * **Arquitectura:** Orientada a Objetos (OOP).
  * **Componentes Clave:** `core/environment.py` (simulación), `core/simulation.py` (entrenamiento).
  * **Datos:** Dataset anual con resolución horaria.

-----

## 🚨 Cambios Mandatorios para Copilot 🛠️

Los siguientes cambios deben integrarse manteniendo la arquitectura existente.

### 2.1 Ventanas Contiguas Aleatorias de 24 Horas

  * **Lógica:** Cada episodio usará una ventana contigua aleatoria de 24 horas del dataset.
  * **Coherencia:** Se debe preservar la coherencia temporal (solar, eólica, demanda, etc.).
  * **Restricciones:** **NO** usar *shuffling* ni muestrear hora por hora.
  * **Implementación Obligatoria:**
    ```python
    start = np.random.randint(0, len(env.dataset) - 24)
    episode_data = env.dataset.iloc[start:start + 24]
    ```
  * **Archivos Afectados:** `core/simulation.py`, `core/environment.py`.

### 2.2 SOC Inicial Aleatorio (Battery)

  * **Lógica:** El Estado de Carga (SOC) inicial de la batería debe ser aleatorio en cada episodio.
  * **Implementación Obligatoria:**
    ```python
    initial_soc = np.random.uniform(0.1, 0.9)
    ```
  * **Integración:** Este valor debe pasarse a `env.reset()` y al agente batería.

### 2.3 Remover Escalado Artificial de Demanda

  * **Objetivo:** Utilizar la demanda directamente del dataset, sin alteraciones.
  * **Acciones:** Desactivar o eliminar toda lógica de escalado artificial de demanda (`demand_scale`) en `simulation.py` y `environment.py`.
  * **Configuración:** Conservar la estructura del `config`, pero ignorar/eliminar modos como "random" o "fixed".

### 2.4 Modificación de `Environment.reset()`

  * **Nueva Firma:**
    ```python
    def reset(self, episode_data, initial_soc):
    ```
  * **Funciones:**
      * Resetear acumuladores y índices de discretización.
      * Asignar `initial_soc` a la batería.
      * Cargar la ventana `episode_data` como el dataset del episodio.
  * **Compatibilidad:** NO debe romper `get_dataset()` ni el acceso a variables (e.g., `solar_power_0`).

-----

## 2.5 Asegurar Bucle de Entrenamiento Coherente (En `core/simulation.run_training`)

El bucle debe seguir estos pasos en cada episodio:

1.  **Escoger** ventana de 24h (`episode_data`).
2.  **Generar** SOC inicial *random* (`initial_soc`).
3.  **Llamar** a `env.reset(episode_data, initial_soc)`.
4.  **Ejecutar** ciclo de 24 pasos.
5.  **Actualizar** Q-tables.
6.  **Guardar** CSV por episodio.

-----

## 🧠 Reglas Obligatorias de Arquitectura 🏛️

  * **✔ OOP obligatorio:** Ningún archivo debe volverse procedural.
  * **✔ No reescribir clases:** Modificar solo funciones específicas.
  * **✔ Comentarios y Docstrings:** Estrictamente en **INGLÉS**.
  * **✔ Mantener API:** NO alterar nombres de métodos, atributos, `update_power()`, `digitize_clip`, ni la construcción de Q-table.
  * **✔ Simulación Horaria:** Mantener la relación 1-step $\rightarrow$ 1-hour.

-----

## 🗂 Archivos A Tocar y Detalles de Implementación 📝

### ✔ `core/simulation.py`

  * **Inyección de Lógica:** Añadir al inicio del bucle de episodio:
    ```python
    start = np.random.randint(0, len(env.dataset) - 24)
    episode_data = env.dataset.iloc[start:start + 24]

    initial_soc = np.random.uniform(
        config["agents"]["battery"]["limits"]["initial_soc_min"],
        config["agents"]["battery"]["limits"]["initial_soc_max"]
    )

    env.reset(episode_data, initial_soc)
    ```

### ✔ `core/environment.py`

  * **Implementar `reset`:**
    ```python
    def reset(self, episode_data, initial_soc):
        self.episode_data = episode_data  # Guardar la ventana
        self.battery.soc = initial_soc    # Inyectar SOC
        self.current_step = 0             # Resetear paso
        # ... Lógica de reseteo de acumuladores ...
    ```
  * **Reemplazar Acceso a Dataset:**
      * Donde se acceda a datos del dataset global por índice de paso, usar:
        ```python
        value = self.episode_data.iloc[self.current_step][varname]
        ```

### ✔ `configs/default.yaml` y `dataloader/loader.py`

  * **Configuración:** Deshabilitar lógicas de `demand_scale` y verificar que los límites `initial_soc_min/max` estén presentes.
  * **Dataloader:** Asegurar la lectura del dataset anual completo para que `simulation.py` pueda seleccionar la ventana.

-----

## ❌ Prohibiciones Estrictas 🚫

  * **No** crear nuevas rutas de dataset.
  * **No** generar agentes nuevos ni modificar el `agents/` folder.
  * **No** modificar `core/rewards.py`.
  * **No** reemplazar la lógica de Q-learning (`select_action()`, `update()`, etc.).
  * **No** romper la construcción del espacio de estados (`state_space`).
  * **No** alterar la estructura de logs o CSVs por episodio.