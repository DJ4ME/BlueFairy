import json
from pathlib import Path

import pandas as pd

from ollamaUtils import PATH as OLLAMA_PATH


DEFAULT_MODELS_FILE_NAME = "models.json"


def select_models(
        file: Path = OLLAMA_PATH / DEFAULT_MODELS_FILE_NAME,
        min_size: float = 0.0,
        max_size: float = 500.0,
        parameter_min_size: float = 0.0,
        parameter_max_size: float = 500.0,
        family: str = None
) -> pd.DataFrame:
    """
    The function reads the available models from a json file and filters them based on the specified size range and family.
    If no family is specified, it returns all models within the size range.
    :param file: the path to the models file
    :param min_size: the minimum size of the models to be selected
    :param max_size: the maximum size of the models to be selected
    :param parameter_min_size: the minimum number of parameters of the models to be selected
    :param parameter_max_size: the maximum number of parameters of the models to be selected
    :param family: the family of models to be selected
    :return: a list of model names that match the criteria
    """
    _models = json.load(open(file, "r", encoding="utf-8"))["models"]
    selected_models = []
    for model in _models:
        size_gb = model["size"] / (1024 ** 3)
        num_parameters_billions = float(model["details"].get("parameter_size", 0)[:-1])
        model_family = model["details"].get("family", "").lower()
        if min_size <= size_gb <= max_size:
            if parameter_min_size <= num_parameters_billions <= parameter_max_size:
                if family is None or family.lower() in model_family.lower():
                    selected_models.append([model["name"], num_parameters_billions, size_gb])
    return pd.DataFrame(selected_models, columns=["model_name", "params (B)", "size (GB)"])


if __name__ == "__main__":
    # Example usage
    models = select_models(min_size=3.0, max_size=5.0, family=None)
    print(f"Found {len(models)} models between 1GB and 5GB:")
    print(models.sort_values(by="size (GB)"))