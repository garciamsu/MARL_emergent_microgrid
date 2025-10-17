# 🚨 ANÁLISIS CRÍTICO: Problema con env_demand_power en Fila 15

## 📋 Resumen del Problema

En la **fila 15** del CSV (`results/evolution/episode_0.csv`), se detectó que:

```
env_demand_power = 0.00 W
```

A pesar de que hay demanda en el dataset (`base_demand = 20W`).

---

## 🔍 Análisis Detallado de la Fila 15

### **Datos de los Agentes:**

| Agente   | Action | Potential | Power     | Observación                      |
|----------|--------|-----------|-----------|----------------------------------|
| Solar    | 0      | 73.26 W   | 0.00 W    | No inyecta (action=0)           |
| Wind     | 1      | 64.34 W   | 64.34 W   | Inyecta completamente           |
| Battery  | 2      | 0.00 W    | 0.00 W    | Intenta descargar pero SOC bajo |
| Grid     | 0      | 0.00 W    | 0.00 W    | No importa (action=0)           |
| **Load** | **0**  | **20.00 W** | **0.00 W** | ❌ **PROBLEMA AQUÍ**          |

### **Totales del Entorno:**

```
env_total_renewable = 64.34 W  ✓ (solar + wind)
env_total_power     = 64.34 W  ✓ (renewables + battery + grid)
env_demand_power    = 0.00 W   ❌ (debería ser ~20W o menos)
env_energy_balance  = 64.34 W  ✓ (total - demand)
```

---

## 🐛 Causa Raíz del Problema

### **Configuración Actual:**

```yaml
# configs/default.yaml
load:
  limits:
    p_load: 200  # ← Carga controlable de 200W
```

### **Dataset:**
```
base_demand = 20W  # Demanda del dataset
```

### **Lógica del Load Agent:**

```python
# agents/load_agent.py (líneas 43-45)
if self.action == 0:  # Shed controllable load
    controllable_demand = max(0, base_demand - self.p_load)
    self.power = -controllable_demand
```

### **Cálculo Problemático:**

```python
base_demand = 20W
p_load = 200W

# Cuando action = 0:
controllable_demand = max(0, 20 - 200) = max(0, -180) = 0W
power = -0 = 0W

# Cuando action = 1:
power = -20W  ✓ Funciona correctamente
```

---

## 💥 **Impacto del Problema:**

1. **Desconexión total inadvertida:**
   - Cuando `load.action=0` (debería reducir carga), en realidad **desconecta todo**
   - `p_load=200W` es mucho mayor que `base_demand=20W`

2. **Balance energético incorrecto:**
   - `env_demand_power = 0W` cuando debería ser mayor
   - El sistema piensa que no hay demanda cuando `action=0`

3. **Rewards incorrectos:**
   - Los agentes reciben rewards basados en `env_demand_power = 0`
   - Esto distorsiona el aprendizaje

4. **Curtailment artificial:**
   - 64.34W de generación sin demanda → 100% de curtailment

---

## ✅ **Soluciones Propuestas:**

### **Solución 1: Ajustar `p_load` en la Configuración (RECOMENDADA)**

**Cambio en `configs/default.yaml`:**

```yaml
load:
  limits:
    p_load: 5  # Reducir de 200W a 5W
    # O usar un porcentaje: p_load: 10  (50% de 20W)
```

**Efecto:**
```python
# Con p_load = 5W
controllable_demand = max(0, 20 - 5) = 15W
power = -15W  ✓

# Con p_load = 10W  
controllable_demand = max(0, 20 - 10) = 10W
power = -10W  ✓
```

**Ventajas:**
- ✅ Cambio mínimo (1 línea)
- ✅ Lógica del agente se mantiene
- ✅ Fácil de ajustar por experimento

---

### **Solución 2: Cambiar Lógica a Porcentaje**

**Cambio en `agents/load_agent.py`:**

```python
def update_power(self, env):
    base_demand = getattr(env, 'base_demand', 0.0)
    
    if self.action == 1:
        # Full demand
        self.potential = base_demand
        self.power = -base_demand
    else:  # action == 0
        # Reduce by percentage (e.g., 50%)
        reduction_percentage = 0.5  # o leer de config
        reduced_demand = base_demand * (1 - reduction_percentage)
        self.potential = base_demand
        self.power = -reduced_demand
```

**Con base_demand = 20W:**
```python
# action = 0: reduce 50%
reduced_demand = 20 * (1 - 0.5) = 10W
power = -10W  ✓

# action = 1: full demand
power = -20W  ✓
```

**Ventajas:**
- ✅ Más robusto ante variaciones de demanda
- ✅ No depende de valores absolutos
- ✅ Escalable a diferentes datasets

**Desventajas:**
- ⚠️ Cambia la semántica del agente
- ⚠️ Requiere más cambios en el código

---

### **Solución 3: Usar `p_load` como Porcentaje (HÍBRIDA)**

**Cambio en `configs/default.yaml`:**

```yaml
load:
  limits:
    p_load_percentage: 0.25  # 25% de reducción
```

**Cambio en `agents/load_agent.py`:**

```python
def update_power(self, env):
    base_demand = getattr(env, 'base_demand', 0.0)
    p_load_pct = self.limits.get("p_load_percentage", 0.5)
    
    if self.action == 1:
        self.potential = base_demand
        self.power = -base_demand
    else:  # action == 0
        # Reduce by configured percentage
        reduced_demand = base_demand * (1 - p_load_pct)
        self.potential = base_demand
        self.power = -reduced_demand
```

**Ventajas:**
- ✅ Configurable
- ✅ Robusto
- ✅ Claro en intención

---

## 📊 **Comparación de Soluciones:**

| Solución | Facilidad | Robustez | Semántica | Recomendación |
|----------|-----------|----------|-----------|---------------|
| **1. Ajustar p_load**      | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | ⭐⭐⭐⭐   | ✅ **PARA YA** |
| **2. Lógica porcentual**   | ⭐⭐⭐     | ⭐⭐⭐⭐⭐   | ⭐⭐⭐     | ⏳ Futuro |
| **3. Híbrida (% config)**  | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | 🎯 **IDEAL** |

---

## 🎯 **Recomendación Inmediata:**

### **Opción A: Fix Rápido (Ahora)**
```yaml
# configs/default.yaml - Línea 167
p_load: 10  # Cambiar de 200 a 10 (50% de 20W)
```

### **Opción B: Fix Robusto (Recomendado)**
1. Cambiar config a porcentaje
2. Modificar lógica en `load_agent.py`
3. Documentar el cambio

---

## 📝 **Validación Post-Fix:**

Después de aplicar el fix, verificar:

```bash
# 1. Ejecutar simulación
python main.py

# 2. Analizar fila 15 nuevamente
python analyze_row15.py

# Esperado:
# - load.power ≠ 0 cuando action=0
# - env_demand_power > 0
# - Balance energético realista
```

---

## 🚨 **Impacto en Otras Filas:**

Revisar todas las filas donde `load.action = 0`:

```python
import pandas as pd

df = pd.read_csv('results/evolution/episode_0.csv', sep=';')
problematic_rows = df[(df['action_load#0'] == 0) & (df['power_load#0'] == 0)]

print(f"Filas afectadas: {len(problematic_rows)}")
print(problematic_rows[['action_load#0', 'power_load#0', 'env_demand_power']])
```

---

## ✅ **Conclusión:**

El problema **NO es que la demanda no se extrae del dataset**, sino que:

1. ✅ La demanda SÍ se extrae correctamente (`base_demand = 20W`)
2. ❌ El `load_agent` la reduce a 0 por configuración incorrecta de `p_load`
3. ❌ `env_demand_power = 0` porque `load.power = 0`

**Fix inmediato:** Cambiar `p_load: 200` a `p_load: 10` en la configuración.

---

**Fecha:** 17 de octubre de 2025  
**Archivo Analizado:** `results/evolution/episode_0.csv`, Fila 15  
**Validado con:** `analyze_row15.py`
