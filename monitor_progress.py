#!/usr/bin/env python3
"""
Monitor en tiempo real del progreso de la simulación
"""

import time
import os
from pathlib import Path

results_dir = Path('results/evolution')

print("="*70)
print("📊 MONITOR DE SIMULACIÓN - 500 EPISODIOS")
print("="*70)
print("\n⏳ Esperando que comience la simulación...\n")

last_count = 0
start_time = time.time()

try:
    while True:
        # Contar archivos de episodios
        episode_files = list(results_dir.glob('episode_*.csv'))
        current_count = len(episode_files)
        
        if current_count > last_count:
            elapsed = time.time() - start_time
            episodes_per_sec = current_count / elapsed if elapsed > 0 else 0
            eta_seconds = (500 - current_count) / episodes_per_sec if episodes_per_sec > 0 else 0
            eta_minutes = eta_seconds / 60
            
            # Calcular porcentaje
            progress = (current_count / 500) * 100
            
            # Barra de progreso
            bar_length = 40
            filled = int(bar_length * current_count / 500)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"\r[{bar}] {current_count}/500 ({progress:.1f}%) | "
                  f"ETA: {eta_minutes:.1f} min | "
                  f"Velocidad: {episodes_per_sec:.2f} ep/s", 
                  end='', flush=True)
            
            last_count = current_count
            
            # Terminar si llegamos a 500
            if current_count >= 500:
                print("\n\n✅ ¡Simulación completada!")
                break
        
        time.sleep(2)  # Revisar cada 2 segundos
        
except KeyboardInterrupt:
    print(f"\n\n⚠️  Monitor interrumpido. Episodios completados: {current_count}/500")
except Exception as e:
    print(f"\n\n❌ Error: {e}")
