import yaml
import json
import os

def load_config():
    """Load configurations of yaml file"""
    current_path = os.path.dirname(__file__)

    with open(os.path.join(current_path, "config.yaml"), "r") as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)

    # Empty class for yaml loading
    class cfg: pass
    
    for key in config:
        setattr(cfg, key, config[key])

    return cfg
