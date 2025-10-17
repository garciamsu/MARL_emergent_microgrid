#!/usr/bin/env python3
"""
Monitor simplificado del progreso
"""
import time
from pathlib import Path

results_dir = Path('results/evolution')
print("🔄 Monitoreando simulación de 500 episodios...")
print("   (Ctrl+C para detener el monitor, la simulación continuará)\n")

last_count = 0
start_time = time.time()

try:
    while True:
        episode_files = list(results_dir.glob('episode_*.csv'))
        current_count = len(episode_files)
        
        if current_count != last_count:
            elapsed = time.time() - start_time
            progress = (current_count / 500) * 100
            
            if current_count > 0 and elapsed > 0:
                rate = current_count / elapsed
                eta = (500 - current_count) / rate if rate > 0 else 0
                
                bar_len = 50
                filled = int(bar_len * current_count / 500)
                bar = "█" * filled + "░" * (bar_len - filled)
                
                print(f"\r[{bar}] {current_count:3d}/500 ({progress:5.1f}%) | "
                      f"ETA: {eta/60:4.1f} min | {rate:4.2f} ep/s", 
                      end='', flush=True)
            
            last_count = current_count
            
            if current_count >= 500:
                print("\n\n✅ ¡Simulación completada!")
                break
        
        time.sleep(1)
        
except KeyboardInterrupt:
    print(f"\n\n⚠️  Monitor detenido en episodio {current_count}/500")
    print("   (La simulación sigue ejecutándose en segundo plano)")
