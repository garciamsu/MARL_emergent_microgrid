import yaml


def load_config(path="configs/default.yaml"):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    # TODO: validate config with schema
    return config