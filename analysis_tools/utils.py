import glob
import os
from typing import Dict, List


def clear_directories(short: bool = True) -> Dict[str, List[str]]:
    """Limpia (solo archivos) varias carpetas de resultados agregando los mensajes en bloques.

    Directorios considerados:
        - results/
        - results/evolution/
        - results/plots/
        - results/logs/*agent (battery, grid, load, solar, wind)

    El comportamiento anterior imprimía una línea por cada evento (borrado, vacío, inexistente, ignorado).
    Ahora se agrupan por categoría para reducir el ruido:
        Ignored (not a file): <lista>
        Deleted: <lista>
        No files in: <lista>
        Directory does not exist: <lista>
        Could not delete: <archivo -> error>

    Returns:
        dict con las listas recopiladas (útil para pruebas o logging estructurado).
    """

    # Asegurar ciertos directorios base mínimos (evita muchos 'no existe')
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/evolution", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    directories = [
        "results/",
        "results/evolution/",
        "results/plots/",
        "results/logs/batteryagent",
        "results/logs/gridagent",
        "results/logs/loadagent",
        "results/logs/solaragent",
        "results/logs/windagent",
    ]

    collected: Dict[str, List[str]] = {
        "ignored": [],
        "deleted": [],
        "empty": [],
        "missing": [],
        "errors": [],  # formato: "ruta -> error"
    }

    for dir_path in directories:
        if not os.path.exists(dir_path):
            collected["missing"].append(dir_path)
            continue

        files = glob.glob(os.path.join(dir_path, "*"))
        if not files:
            collected["empty"].append(dir_path)
            continue

        for file_path in files:
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    collected["deleted"].append(file_path)
                except Exception as e:  # pragma: no cover - muy raro pero útil
                    collected["errors"].append(f"{file_path} -> {e}")
            else:
                collected["ignored"].append(file_path)

    # Construir salida agregada en orden similar al flujo original
    output_lines: List[str] = []
    if collected["ignored"]:
        output_lines.append("Ignored (not a file): " + ", ".join(collected["ignored"]))
    if collected["deleted"]:
        output_lines.append("Deleted: " + ", ".join(collected["deleted"]))
    if collected["empty"]:
        output_lines.append("No files in: " + ", ".join(collected["empty"]))
    if collected["missing"]:
        output_lines.append("Directory does not exist: " + ", ".join(collected["missing"]))
    if collected["errors"]:
        output_lines.append("Could not delete: " + ", ".join(collected["errors"]))

    if short:
        # Mensaje ultra resumido solo con conteos
        print(
            "Cleanup => "
            f"deleted:{len(collected['deleted'])} | "
            f"ignored:{len(collected['ignored'])} | "
            f"empty:{len(collected['empty'])} | "
            f"missing:{len(collected['missing'])} | "
            f"errors:{len(collected['errors'])}"
        )
    else:
        if output_lines:
            print(" | ".join(output_lines) + " | Cleanup completed.")
        else:
            print("Cleanup completed.")

    return collected