
import os
from box.exceptions import BoxValueError
import yaml
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from src.entity import UNetRTrainerConfig

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e

PARAMS_FILE_PATH = Path('./params.yaml') 

class ConfigurationManager:
    def __init__(self, params_file_path=PARAMS_FILE_PATH):
        self.params = read_yaml(params_file_path)

    def get_UNetR_params(self) -> UNetRTrainerConfig:
        params = self.params.UNetRTrainingArguments

        model_trainer_config = UNetRTrainerConfig(
            image_size=params.image_size,
            patch_size=params.patch_size,
            hidden_dim=params.hidden_dim,
            num_channels=params.num_channels,
            num_layers=params.num_layers,
            num_heads=params.num_heads,
            mlp_dim=params.mlp_dim,
            dropout_rate=params.dropout_rate

        )

        return model_trainer_config