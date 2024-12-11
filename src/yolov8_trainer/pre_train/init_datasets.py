import json
from pathlib import Path

import typer

from yolov8_trainer.common import DATASETS_DIR, SPLITS, WORKDIR


def inner_main(
    datasets_dir: Path = typer.Argument(
        DATASETS_DIR, file_okay=False, dir_okay=True, help=""
    )
) -> None:

    all_datasets: dict[str, list[str]] = {key: [] for key in SPLITS}

    for split in SPLITS:
        split_dir = datasets_dir / split
        datasets_paths = [f for f in split_dir.glob("*") if f.is_dir()]
        datasets_paths.sort()

        print(f"For {split} found: {len(datasets_paths)} datasets")

        for dataset in datasets_paths:
            dataset_txt_file_path = dataset / f"{dataset.name}_{split}.yaml"
            with open(dataset_txt_file_path, "w") as file:
                single_dataset_path = WORKDIR / dataset
                file.write(f"path: {single_dataset_path}\n")
                file.write(f"{split}: {split}/images\n")
                file.write("names:\n")
                file.write("   0: person\n")
                file.write("   2: forklift\n")

                all_datasets[split].append(str(single_dataset_path))

    json_path = "all_datasets.json"
    with open(json_path, "w") as file:
        json.dump(all_datasets, file, indent=4)

    print(f"List of every dataset save in: {json_path}")


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
