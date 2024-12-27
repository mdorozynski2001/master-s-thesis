from pathlib import Path

import yaml
from munch import Munch

RESULT_JSON = "result.json"
# file with evaluation results made on whole unitem test dataset.
RESULT_SPLITTED_DATASETS = "result_splitted_datasets.json"
# file with evaluation results made on every single unitem dataset separatelly
RESULT_ONLY_COMBINED = "result_only_combined.json"
# file with only combined evaluation results from every single unitem dataset


def load_yaml(filepath: Path) -> Munch:
    with filepath.open("r") as file:
        d = yaml.safe_load(file)
        return Munch(d)
