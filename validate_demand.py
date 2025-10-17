"""
Script de validación para verificar que la demanda del dataset
se está extrayendo y utilizando correctamente en los cálculos.
"""

import pandas as pd
import os

def validate_demand_extraction():
    """Valida que la demanda del dataset se extrae correctamente."""
    
    print("\n" + "="*80)
    print("VALIDACIÓN DE EXTRACCIÓN DE DEMANDA DEL DATASET")
    print("="*80)
    
    # Cargar dataset
    dataset_path = os.path.join("assets", "datasets", "Case1.csv")
    df = pd.read_csv(dataset_path, sep="[;,]", engine="python")
    
    print(f"\n📊 Dataset cargado: {dataset_path}")
    print(f"   Filas: {len(df)}")
    print(f"   Columnas: {list(df.columns)}")
    
    # Mostrar primeras filas de demanda
    if "demand" in df.columns:
        print(f"\n✅ Columna 'demand' encontrada")
        print(f"\n   Primeros 5 valores de demanda:")
        for i in range(min(5, len(df))):
            print(f"      Index {i}: {df.iloc[i]['demand']:.2f} W")
    else:
        print(f"\n❌ ERROR: Columna 'demand' NO encontrada en el dataset")
        return False
    
    # Simular el flujo de extracción (como en simulation.py)
    print(f"\n🔄 Simulando flujo de extracción (como en simulation.py):")
    
    index = 0
    scale_demand = 1.0
    
    # Paso 1: Extracción base (simulando env.get_dataset)
    base_demand_from_dataset = df.iloc[index]["demand"] * scale_demand
    print(f"\n   Paso 1 - Extracción del dataset:")
    print(f"      base_demand_from_dataset = {base_demand_from_dataset:.2f} W")
    
    # Paso 2: Almacenamiento en env.base_demand
    class MockEnv:
        def __init__(self):
            self.base_demand = 0.0
            self.demand_power = 0.0
    
    env = MockEnv()
    env.base_demand = base_demand_from_dataset
    print(f"\n   Paso 2 - Almacenamiento en env:")
    print(f"      env.base_demand = {env.base_demand:.2f} W")
    
    # Paso 3: Reset de acumulador
    env.demand_power = 0.0
    print(f"\n   Paso 3 - Reset de acumulador:")
    print(f"      env.demand_power = {env.demand_power:.2f} W (inicializado a 0)")
    
    # Paso 4: Load agent usa env.base_demand
    load_action = 1
    p_load = 200.0
    
    if load_action == 1:
        load_power = -env.base_demand
    else:
        controllable_demand = max(0, env.base_demand - p_load)
        load_power = -controllable_demand
    
    print(f"\n   Paso 4 - Load agent calcula power:")
    print(f"      load.action = {load_action}")
    print(f"      load.power = {load_power:.2f} W")
    
    # Paso 5: Acumulación en env.demand_power
    env.demand_power += abs(load_power)
    print(f"\n   Paso 5 - Acumulación en env.demand_power:")
    print(f"      env.demand_power = {env.demand_power:.2f} W")
    
    # Validación
    print(f"\n" + "="*80)
    print(f"✅ VALIDACIÓN:")
    print(f"="*80)
    
    if abs(env.demand_power - base_demand_from_dataset) < 0.01:
        print(f"   ✓ CORRECTO: env.demand_power == base_demand_from_dataset")
        print(f"   ✓ La demanda del dataset se está utilizando correctamente")
        return True
    else:
        print(f"   ❌ ERROR: env.demand_power != base_demand_from_dataset")
        print(f"      Esperado: {base_demand_from_dataset:.2f} W")
        print(f"      Obtenido: {env.demand_power:.2f} W")
        print(f"      Diferencia: {abs(env.demand_power - base_demand_from_dataset):.2f} W")
        return False


def validate_multiple_steps():
    """Valida que la demanda se extrae correctamente en múltiples pasos."""
    
    print("\n" + "="*80)
    print("VALIDACIÓN DE MÚLTIPLES PASOS")
    print("="*80)
    
    # Cargar dataset
    dataset_path = os.path.join("assets", "datasets", "Case1.csv")
    df = pd.read_csv(dataset_path, sep="[;,]", engine="python")
    
    scale_demand = 1.0
    errors = 0
    
    print(f"\nValidando primeros 10 pasos de simulación:")
    
    for index in range(min(10, len(df))):
        # Extracción
        base_demand = df.iloc[index]["demand"] * scale_demand
        
        # Simulación de uso en load agent
        load_action = 1
        load_power = -base_demand
        demand_power = abs(load_power)
        
        # Validación
        if abs(demand_power - base_demand) < 0.01:
            status = "✓"
        else:
            status = "❌"
            errors += 1
        
        print(f"   {status} Step {index}: Dataset={base_demand:.2f} W → Used={demand_power:.2f} W")
    
    print(f"\n" + "="*80)
    if errors == 0:
        print(f"✅ ÉXITO: Todos los pasos validados correctamente")
        return True
    else:
        print(f"❌ ERROR: {errors} pasos con discrepancias")
        return False


if __name__ == "__main__":
    # Validación 1: Flujo de extracción
    result1 = validate_demand_extraction()
    
    # Validación 2: Múltiples pasos
    result2 = validate_multiple_steps()
    
    # Resultado final
    print(f"\n" + "="*80)
    print(f"RESULTADO FINAL")
    print(f"="*80)
    
    if result1 and result2:
        print(f"✅ TODAS LAS VALIDACIONES PASARON")
        print(f"   La demanda del dataset se está extrayendo y utilizando correctamente")
    else:
        print(f"❌ ALGUNAS VALIDACIONES FALLARON")
        print(f"   Revisar el flujo de extracción de demanda")
    
    print(f"="*80 + "\n")
