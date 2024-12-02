import os
import shutil
from pathlib import Path

import typer


def resolve_symlinks_and_replace(
    symlink_root: Path = typer.Argument(
        None, dir_okay=True, file_okay=False, help="Path to dir with symlinks."
    ),
    original_root: Path = typer.Argument(
        None, dir_okay=True, file_okay=False, help="Path to dir with original files."
    ),
) -> None:
    for root, dirs, files in os.walk(symlink_root):
        for name in files:
            symlink_path = os.path.join(root, name)

            if os.path.islink(symlink_path):
                target_path = os.readlink(symlink_path)
                resolved_target = os.path.abspath(
                    os.path.join(os.path.dirname(symlink_path), target_path)
                )

                original_file_path = find_original_file(resolved_target, original_root)

                if original_file_path and os.path.exists(original_file_path):
                    print(
                        f"Changing symling: {symlink_path} to file {original_file_path}"
                    )
                    os.unlink(symlink_path)
                    shutil.copy2(original_file_path, symlink_path)
                else:
                    print(f"WARING: No file founded for symlink: {symlink_path}")

        for name in dirs:
            dir_symlink_path = os.path.join(root, name)

            if os.path.islink(dir_symlink_path):
                target_path = os.readlink(dir_symlink_path)
                resolved_target = os.path.abspath(
                    os.path.join(os.path.dirname(dir_symlink_path), target_path)
                )

                original_dir_path = find_original_file(resolved_target, original_root)

                if original_dir_path and os.path.exists(original_dir_path):
                    print(
                        f"Changing dir's symlink: {dir_symlink_path} to dir {original_dir_path}"
                    )
                    os.unlink(dir_symlink_path)
                    shutil.copytree(original_dir_path, dir_symlink_path)
                else:
                    print(f"Warning: No dir founded for symlink: {dir_symlink_path}")


def find_original_file(target_path: str, original_root: Path) -> str | None:
    relative_path = os.path.relpath(target_path, start=original_root)

    potential_path = os.path.join(original_root, relative_path)

    if os.path.exists(potential_path):
        return potential_path
    return None


def main() -> None:
    typer.run(resolve_symlinks_and_replace)


if __name__ == "__main__":
    main()
