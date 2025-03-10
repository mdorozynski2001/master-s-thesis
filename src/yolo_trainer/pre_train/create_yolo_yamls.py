import json
from datetime import datetime
from pathlib import Path

import typer
from tqdm import tqdm

from yolo_trainer.common import DATASETS_DIR, WORKDIR
from yolo_trainer.pre_train.utils import create_yaml


def inner_main(
    train_json_path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=False, help=""),
    output_dir: Path = typer.Option(None, file_okay=False, dir_okay=True, help=""),
) -> None:
    with open(train_json_path, "r") as file:
        train_data = json.load(file)

    current_datetime = datetime.now()
    if output_dir is None:
        formatted_date = current_datetime.strftime("%Y_%m_%d_%H_%M")
        output_dir = DATASETS_DIR / f"yolo_data_{formatted_date}"
    else:
        output_dir = WORKDIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in train_data.keys():
        if len(train_data[split]) == 0:
            continue

        out_split_dir = output_dir / split
        out_split_txt_dir = out_split_dir / "labels"
        out_split_images_dir = out_split_dir / "images"
        out_split_dir.mkdir(parents=True, exist_ok=True)
        out_split_txt_dir.mkdir(parents=True, exist_ok=True)
        out_split_images_dir.mkdir(parents=True, exist_ok=True)

        id = 0

        for dataset in tqdm(train_data[split]):
            labels_files_dir = Path(dataset) / split / "labels"
            images_files_dir = Path(dataset) / split / "images"
            labels_path_list = list(Path(labels_files_dir).glob("*"))
            for label_path in labels_path_list:
                img_list = list(Path(images_files_dir).glob(f"{label_path.stem}.*"))
                if len(img_list) != 1:
                    raise NameError(f"{label_path.name} has {len(img_list)} match(es)!!")
                img_path = img_list[0]

                label_symlink_path = out_split_txt_dir / f"{id}.txt"
                img_symlink_path = out_split_images_dir / f"{id}{img_path.suffix}"

                try:
                    label_symlink_path.symlink_to(label_path)
                except FileExistsError:
                    print("Symlink already exists.")
                except OSError as e:
                    raise OSError(f"Error: {e}")

                try:
                    img_symlink_path.symlink_to(img_path)
                except FileExistsError:
                    print("Symlink already exists.")
                except OSError as e:
                    raise OSError(f"Error: {e}")

                id += 1

        out_yaml_path = output_dir / f"{split}.yaml"
        create_yaml(out_yaml_path, output_dir, split)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
