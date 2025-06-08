import typer
from pathlib import Path

def inner_main(
    labels_path: Path = typer.Argument(..., help="Path to the labels directory"),
) -> None:
    labels = list(labels_path.rglob("*.txt"))
    print(f"Number of labels: {len(labels)}")

    labels_amount = {
        "person": 0,
        "forklift": 0,
    }

    for label in labels:
        with open(label, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("0"):
                    labels_amount["person"] += 1
                elif line.startswith("2"):
                    labels_amount["forklift"] += 1
    print(labels_amount)

def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()