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