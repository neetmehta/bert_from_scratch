import importlib.util
import sys

def load_config(config_path):
    """Dynamically loads a Python config file as a module.
    Args:
        config_path (str): Path to the config .py file.
    Returns:
        module: The loaded config module.
    """
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["config_module"] = config_module
    spec.loader.exec_module(config_module)
    return config_module