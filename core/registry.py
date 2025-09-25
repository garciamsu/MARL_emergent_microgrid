AGENT_REGISTRY = {}
POLICY_REGISTRY = {}
REWARD_REGISTRY = {}

def register_agent(name):
    def wrapper(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return wrapper


def register_policy(name):
    def wrapper(cls):
        POLICY_REGISTRY[name] = cls
        return cls
    return wrapper


def register_reward(name):
    def wrapper(cls):
        REWARD_REGISTRY[name] = cls
        return cls
    return wrapper


def create_agent(agent_type, *args, **kwargs):
    cls = AGENT_REGISTRY.get(agent_type)
    if cls is None:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return cls(*args, **kwargs)


def create_policy(cfg):
    cls = POLICY_REGISTRY.get(cfg["type"])
    if cls is None:
        raise ValueError(f"Unknown policy type {cfg['type']}")
    return cls(**{k: v for k, v in cfg.items() if k != "type"})


def create_reward(cfg):
    cls = REWARD_REGISTRY.get(cfg["type"])
    if cls is None:
        raise ValueError(f"Unknown reward type {cfg['type']}")
    return cls(**cfg.get("params", {}))
