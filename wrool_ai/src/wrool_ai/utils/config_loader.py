import yaml
import json
from typing import Dict, Any
from pathlib import Path
import os

class ConfigLoader:
    """Configuration loader for Wrool-AI"""
    
    def __init__(self):
        self.config_path = Path("config")
        self.default_config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        default_path = self.config_path / "default.yaml"
        return self._load_yaml(default_path)
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration file"""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_config(self, config_name: str = None) -> Dict[str, Any]:
        """Load configuration with environment overrides"""
        config = self.default_config.copy()
        
        # Load environment-specific config
        if config_name:
            env_path = self.config_path / f"{config_name}.yaml"
            if env_path.exists():
                env_config = self._load_yaml(env_path)
                config.update(env_config)
        
        # Override with environment variables
        self._override_with_env_vars(config)
        
        return config
    
    def _override_with_env_vars(self, config: Dict[str, Any]):
        """Override configuration with environment variables"""
        for key in list(config.keys()):
            env_key = f"WROOL_{key.upper()}"
            if env_key in os.environ:
                try:
                    config[key] = json.loads(os.environ[env_key])
                except json.JSONDecodeError:
                    config[key] = os.environ[env_key]