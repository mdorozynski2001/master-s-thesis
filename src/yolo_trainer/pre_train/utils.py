from pathlib import Path


def create_yaml(yaml_path: Path, dataset_path: Path, split: str) -> None:
    with open(yaml_path, "w") as file:
        file.write(f"path: {dataset_path}\n")
        file.write("train: train/images\n")
        file.write("val: val/images\n")
        if split == "test":
            file.write("test: test/images\n")
        file.write("names:\n")
        file.write("   0: person\n")
        file.write("   1: head\n")
        file.write("   2: forklift\n")
