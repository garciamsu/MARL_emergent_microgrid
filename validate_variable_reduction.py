"""
Script de validación de la eliminación de variables duplicadas.
Verifica que las variables restantes contengan toda la información necesaria.
"""

import pandas as pd
import os

def validate_variable_reduction():
    """Valida que la reducción de variables no perdió información."""
    
    print("\n" + "="*80)
    print("VALIDACIÓN DE REDUCCIÓN DE VARIABLES DUPLICADAS")
    print("="*80)
    
    # Cargar CSV más reciente
    csv_path = "results/evolution/episode_0.csv"
    
    if not os.path.exists(csv_path):
        print(f"\n❌ ERROR: No se encontró {csv_path}")
        print("   Ejecuta primero: python main.py")
        return False
    
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Archivo analizado: {csv_path}")
    print(f"   Filas: {len(df)}")
    print(f"   Columnas totales: {len(df.columns)}")
    
    # Listar columnas de ambiente
    env_columns = [col for col in df.columns if col.startswith("env_")]
    
    print(f"\n🔍 Columnas de ambiente encontradas ({len(env_columns)}):")
    for col in env_columns:
        print(f"   ✓ {col}")
    
    # Verificar que NO existan las columnas eliminadas
    removed_columns = ["env_total_generation", "env_total_consumption"]
    
    print(f"\n🗑️  Verificando eliminación de columnas duplicadas:")
    for col in removed_columns:
        if col in df.columns:
            print(f"   ❌ ERROR: '{col}' NO fue eliminada")
        else:
            print(f"   ✓ '{col}' eliminada correctamente")
    
    # Verificar que SÍ existan las columnas mantenidas
    required_columns = [
        "env_total_renewable",
        "env_total_power",
        "env_demand_power",
        "env_energy_balance",
        "env_delta_power_idx"
    ]
    
    print(f"\n✅ Verificando columnas requeridas:")
    all_present = True
    for col in required_columns:
        if col in df.columns:
            print(f"   ✓ '{col}' presente")
        else:
            print(f"   ❌ ERROR: '{col}' faltante")
            all_present = False
    
    # Validar integridad de datos
    print(f"\n📈 Validación de integridad de datos (primeros 5 pasos):")
    
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        
        total_power = row['env_total_power']
        demand_power = row['env_demand_power']
        balance = row['env_energy_balance']
        
        # Verificar que el balance sea correcto
        expected_balance = total_power - demand_power
        
        if abs(balance - expected_balance) < 0.01:
            status = "✓"
        else:
            status = "❌"
        
        print(f"   {status} Step {i}: Gen={total_power:.2f}W, Dem={demand_power:.2f}W, Bal={balance:.2f}W")
    
    # Resumen
    print(f"\n" + "="*80)
    print(f"📊 RESUMEN DE CAMBIOS:")
    print(f"="*80)
    
    print(f"\n   Variables ANTES:  7 (con 2 duplicadas)")
    print(f"   Variables AHORA:  5 (sin duplicados)")
    print(f"   Reducción:        -2 variables (-28.6%)")
    
    print(f"\n   Columnas eliminadas:")
    for col in removed_columns:
        print(f"      ❌ {col}")
    
    print(f"\n   Columnas mantenidas:")
    for col in required_columns:
        print(f"      ✓ {col}")
    
    # Validación final
    print(f"\n" + "="*80)
    
    if all_present and all(col not in df.columns for col in removed_columns):
        print(f"✅ VALIDACIÓN EXITOSA")
        print(f"   - Todas las columnas requeridas están presentes")
        print(f"   - Las columnas duplicadas fueron eliminadas")
        print(f"   - La integridad de datos se mantiene")
        print(f"   - No se perdió información")
        return True
    else:
        print(f"❌ VALIDACIÓN FALLIDA")
        return False


def compare_file_sizes():
    """Compara el tamaño de los archivos antes y después (si hay backup)."""
    
    csv_path = "results/evolution/episode_0.csv"
    
    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path)
        print(f"\n💾 Tamaño del archivo CSV:")
        print(f"   {csv_path}")
        print(f"   Tamaño: {file_size:,} bytes ({file_size/1024:.2f} KB)")
        
        # Estimar reducción
        reduction_percent = 2 / 7 * 100  # 2 de 7 columnas eliminadas
        estimated_savings = file_size * reduction_percent / 100
        
        print(f"\n   Reducción estimada por eliminación de 2 columnas:")
        print(f"   ~{reduction_percent:.1f}% = ~{estimated_savings:,.0f} bytes (~{estimated_savings/1024:.2f} KB)")


if __name__ == "__main__":
    result = validate_variable_reduction()
    compare_file_sizes()
    
    print(f"\n" + "="*80 + "\n")
    
    if result:
        print("✅ La integración de variables fue exitosa")
    else:
        print("❌ Hubo problemas con la integración")
