# Guía de Pruebas

Este directorio contiene los scripts de prueba diseñados para validar el comportamiento y las recompensas de los agentes en el sistema de microgrids. A continuación, se describe el propósito de cada script, cómo ejecutarlo y cómo interpretar los resultados.

## Scripts de Prueba

### 1. `test_smoke.py`
- **Propósito**: Verificar que el sistema puede ejecutar un episodio de entrenamiento completo sin errores.
- **Uso**: Ejecutar el script directamente:
  ```bash
  python test_smoke.py
  ```
- **Interpretación de Resultados**: Si el episodio se ejecuta correctamente, el script imprimirá un mensaje de éxito. Cualquier error indica problemas en la configuración o en la lógica del sistema.

### 2. `battery/test_battery_reward.py`
- **Propósito**: Validar el cálculo de recompensas para el agente de batería.
- **Uso**: Ejecutar el script directamente:
  ```bash
  python battery/test_battery_reward.py
  ```
- **Interpretación de Resultados**: Genera un archivo CSV con las recompensas calculadas. Verificar que los valores sean consistentes con las expectativas del modelo.

### 3. `grid/test_grid_reward.py`
- **Propósito**: Validar el cálculo de recompensas para el agente de red eléctrica.
- **Uso**: Ejecutar el script directamente:
  ```bash
  python grid/test_grid_reward.py
  ```
- **Interpretación de Resultados**: Genera un archivo CSV con las recompensas calculadas. Revisar los valores para asegurar que reflejen el comportamiento esperado.

### 4. `load/test_load_reward.py`
- **Propósito**: Validar el cálculo de recompensas para el agente de carga.
- **Uso**: Ejecutar el script directamente:
  ```bash
  python load/test_load_reward.py
  ```
- **Interpretación de Resultados**: Genera un archivo CSV con las recompensas calculadas. Asegurarse de que los valores sean coherentes con las políticas definidas.

### 5. `solar/test_solar_reward.py`
- **Propósito**: Validar el cálculo de recompensas para el agente solar.
- **Uso**: Ejecutar el script directamente:
  ```bash
  python solar/test_solar_reward.py
  ```
- **Interpretación de Resultados**: Genera un archivo CSV con las recompensas calculadas. Confirmar que los resultados sean consistentes con las expectativas del modelo.

## Notas Generales
- **Requisitos Previos**: Asegúrese de que los archivos de entrada necesarios (CSV) estén presentes en los directorios correspondientes.
- **Resultados**: Los resultados de cada prueba se guardan en archivos CSV dentro de subdirectorios `reports/`.
- **Errores Comunes**: Si un archivo de entrada no se encuentra, el script lanzará un error indicando el archivo faltante.

## Estructura de Resultados
Cada archivo CSV generado contiene:
- **Columnas de Entrada**: Datos utilizados para calcular las recompensas.
- **Columna `reward`**: Recompensas calculadas para cada fila de entrada.

Utilice estos scripts para validar y depurar el comportamiento de los agentes en el sistema de microgrids.