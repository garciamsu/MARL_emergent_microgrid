import os
import importlib

# Obtiene el directorio actual
current_dir = os.path.dirname(__file__)

# Importa todos los módulos que terminan en '_agent.py'
for filename in os.listdir(current_dir):
    if filename.endswith('_agent.py'):
        module_name = f"agents.{filename[:-3]}"
        importlib.import_module(module_name)