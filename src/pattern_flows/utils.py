import yaml
from pathlib import Path

def load_yaml(path: str) -> dict:
    """Load a YAML file as a Python dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merges dict2 into dict1."""
    result = dict1.copy()
    for k, v in dict2.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(base_path: str, model_path: str) -> dict:
    """Load and merge base and model-specific config."""
    base = load_yaml(base_path)
    model = load_yaml(model_path)
    return deep_merge(base, model)

def get_ckpt_file(config, model_name, epoch=None):
    """Load checkpoint file."""
    dir = Path(config.output_dir) / f"{model_name}"
    dir.mkdir(exist_ok=True, parents=True)

    if epoch:
        return dir / f"{model_name}_epoch_{epoch}.pth"
    else:
        return dir / f"{model_name}.pth"

def get_save_dir(config, model_name):
    """Load save directory."""
    dir = Path(config.output_dir) / f"{model_name}"
    dir.mkdir(exist_ok=True, parents=True)