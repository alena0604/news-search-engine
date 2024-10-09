from omegaconf import OmegaConf
import os


class AppConfig:
    """Application configuration loader for environment variables and YAML configs."""

    def __init__(self, config_path: str = None):
        if config_path and os.path.exists(config_path):
            self.config = OmegaConf.load(config_path)
        else:
            self.config = OmegaConf.create()

    def get(self, key: str, default=None):
        """Get configuration value by key from environment variables or YAML config."""
        value = os.getenv(key.upper())
        if value is not None:
            return value

        return self.config.get(key, default)
