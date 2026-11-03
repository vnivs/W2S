import yaml
from typing import Dict, Any


class Config:

    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def __getitem__(self, key):
        return self._config[key]

    def get(self, key, default=None):
        return self._config.get(key, default)

    @property
    def env(self) -> Dict[str, Any]:
        """Environment settings."""
        return self._config['env']

    @property
    def training(self) -> Dict[str, Any]:
        """Training settings."""
        return self._config['training']

    @property
    def network(self) -> Dict[str, Any]:
        """Network settings."""
        return self._config['network']

    @property
    def experiment(self) -> Dict[str, Any]:
        """Experiment settings."""
        return self._config['experiment']


def load_config(config_path: str = 'config.yaml') -> Config:
    """Load a YAML configuration file."""
    return Config(config_path)
