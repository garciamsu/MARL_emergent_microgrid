# Guía de Pruebas

Este directorio contiene los scripts de prueba diseñados para validar el comportamiento y las recompensas de los agentes en el sistema de microgrids. A continuación, se describe el propósito de cada script, cómo ejecutarlo y cómo interpretar los resultados.

## Estructura de Directorios
- **`/input`**: Contiene los archivos de entrada necesarios para las pruebas, como tablas de recompensas en formato CSV.
- **`/output`**: Almacena los resultados generados por los scripts, como tablas Q y visualizaciones.

## Orden de Ejecución
1. **Ejecutar los scripts de recompensa (`test_*_reward.py`)**: Calculan las recompensas para cada agente y generan archivos CSV en el subdirectorio `/output`.
2. **Generar tablas Q (`generate_*_q_table.py`)**: Convierte los archivos CSV de recompensas en tablas Q en formato JSON.
3. **Generar reportes (`*_reward_report.py`)**: Produce visualizaciones y análisis basados en los datos de recompensa.

## Scripts de Prueba

### 1. `test_smoke.py`
- **Propósito**: Verificar que el sistema puede ejecutar un episodio de entrenamiento completo sin errores.
- **Uso**: Ejecutar el script directamente:
  ```bash
  python test_smoke.py
  ```
- **Interpretación de Resultados**: Si el episodio se ejecuta correctamente, el script imprimirá un mensaje de éxito. Cualquier error indica problemas en la configuración o en la lógica del sistema.

### 2. `test_*_reward.py`
- **Propósito**: Validar el cálculo de recompensas para cada agente.
- **Uso**: Ejecutar el script correspondiente al agente:
  ```bash
  python battery/test_battery_reward.py
  python grid/test_grid_reward.py
  python load/test_load_reward.py
  python solar/test_solar_reward.py
  ```
- **Interpretación de Resultados**: Genera un archivo CSV con las recompensas calculadas en `/output`. Verificar que los valores sean consistentes con las expectativas del modelo.

### 3. `generate_*_q_table.py`
- **Propósito**: Convertir los datos de recompensa en tablas Q en formato JSON.
- **Uso**: Ejecutar el script correspondiente al agente:
  ```bash
  python battery/generate_battery_q_table.py
  python grid/generate_grid_q_table.py
  python load/generate_load_q_table.py
  python solar/generate_solar_q_table.py
  ```
- **Interpretación de Resultados**: Genera un archivo JSON con la tabla Q en `/output`. Este archivo se utiliza para entrenar y evaluar las políticas de los agentes.

### 4. `*_reward_report.py`
- **Propósito**: Generar reportes analíticos y visualizaciones basados en los datos de recompensa.
- **Uso**: Ejecutar el script correspondiente al agente:
  ```bash
  python battery/battery_reward_report.py
  python grid/grid_reward_report.py
  python load/load_reward_report.py
  python solar/solar_reward_report.py
  ```
- **Interpretación de Resultados**: Genera gráficos y análisis en `/output`, como histogramas, boxplots y mapas de calor. Estos reportes ayudan a evaluar el desempeño de los agentes.

## Depuración de Recompensas

### Módulo `reward_debug.py`
El módulo `reward_debug.py` proporciona herramientas para depurar y entender el cálculo de recompensas de los agentes. Permite imprimir en consola los hiperparámetros, los inputs y una traza paso a paso del cálculo.

### Uso
Para utilizar el módulo, importe la función `explain_and_compute` en el script de prueba correspondiente. Ejemplo:

```python
from utils.reward_debug import explain_and_compute

# Dentro del script de prueba
reward_value = explain_and_compute(reward_fn, agent, env, state_tuple, step=True)
```

### Parámetros
- **`reward_fn`**: La función de recompensa que se está depurando.
- **`agent`**: El agente que ejecuta la acción.
- **`env`**: El entorno en el que opera el agente.
- **`state_tuple`**: El estado discreto del agente.
- **`step`**: Si es `True`, activa el modo paso a paso, esperando la barra espaciadora para continuar.

### Ejemplo de Salida
```plaintext
=== Depuración de recompensa: DefaultBatteryReward ===

--- Hiperparámetros ---
  theta: 0.5  # Ponderación para penalización por déficit renovable
  beta: 0.3  # Ponderación para bonificación por excedente renovable

--- Atributos del agente ---
  soc_max: 1.0
  action: 2

--- Atributos del entorno (env) ---
  demand_power: 50

--- Inputs (state_tuple) ---
  soc_idx: 3  # Índice discreto del estado de carga de la batería
  demand_power_idx: 2  # Índice discreto de la demanda de potencia

--- Tipo de recompensa no reconocido para trazado; usando compute() directo ---
Resultado: -0.15
```

### Notas
- El modo paso a paso (`step=True`) permite pausar la ejecución después de cada cálculo parcial, útil para analizar cada etapa del proceso.
- Si se presiona `q`, la ejecución se interrumpe.

### Recomendaciones
- Utilice este módulo para validar que los cálculos de recompensa sean consistentes con las expectativas del modelo.
- Combine esta herramienta con los scripts de prueba existentes para una depuración más efectiva.

## Notas Generales
- **Requisitos Previos**: Asegúrese de que los archivos de entrada necesarios (CSV) estén presentes en el subdirectorio `/input`.
- **Resultados**: Los resultados de cada prueba se guardan en el subdirectorio `/output`.
- **Errores Comunes**: Si un archivo de entrada no se encuentra, el script lanzará un error indicando el archivo faltante.

## Estructura de Resultados
Cada archivo generado contiene:
- **Archivos CSV**: Resultados de recompensas calculadas.
- **Archivos JSON**: Tablas Q generadas para entrenamiento.
- **Visualizaciones**: Gráficos que muestran el análisis de recompensas y acciones.

Utilice estos scripts para validar y depurar el comportamiento de los agentes en el sistema de microgrids.