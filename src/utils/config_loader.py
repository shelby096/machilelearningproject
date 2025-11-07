import yaml
import json
import os
from typing import Dict, Any


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    return config


def save_config(config: Dict[str, Any], config_path: str, format_type: str = 'yaml') -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
        format_type: File format ('yaml' or 'json')
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    if format_type.lower() == 'yaml':
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
    elif format_type.lower() == 'json':
        with open(config_path, 'w') as file:
            json.dump(config, file, indent=2)
    else:
        raise ValueError(f"Unsupported format type: {format_type}")


class ConfigLoader:
    """Configuration loader class for managing multiple config files."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = config_dir
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a specific configuration file."""
        if config_name in self._configs:
            return self._configs[config_name]
        
        # Try YAML first, then JSON
        yaml_path = os.path.join(self.config_dir, f"{config_name}.yaml")
        json_path = os.path.join(self.config_dir, f"{config_name}.json")
        
        if os.path.exists(yaml_path):
            config = load_yaml_config(yaml_path)
        elif os.path.exists(json_path):
            config = load_json_config(json_path)
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_name}")
        
        self._configs[config_name] = config
        return config
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.load_config("data_config")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.load_config("model_config")
    
    def get_train_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.load_config("train_config")
