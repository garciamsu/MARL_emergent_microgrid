import yaml


def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # Validación mínima: este proyecto asume dt_h == 1.0
    sim = config.get("simulation", {})
    dt_h = float(sim.get("dt_h", 1.0))
    if abs(dt_h - 1.0) > 1e-9:
        raise ValueError(
            f"Configuración inválida: simulation.dt_h={dt_h}. Este proyecto asume dt_h == 1.0."
        )
    # TODO: validate config with schema
    return config