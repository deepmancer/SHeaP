"""
Converts FLAME pickle files to PyTorch .pt files.
"""

import argparse
from pathlib import Path

import torch

from sheap.load_flame_pkl import load_pkl_format_flame_model


def convert_flame(flame_base_dir, overwrite):
    flame_base_dir = Path(flame_base_dir)
    assert flame_base_dir is not None  # for mypy
    assert flame_base_dir.exists(), (
        f"FLAME_BASE_DIR not found at {flame_base_dir}. "
        "Please set arg flame_base_dir to the FLAME model directory, "
        " or set the FLAME_BASE_DIR environment variable."
    )
    pickle_files = list(flame_base_dir.glob("**/**/*.pkl"))
    skip_files = ["FLAME_masks.pkl"]
    for model_path in pickle_files:
        if model_path.name in skip_files:
            continue
        print(f"Converting {model_path}...")
        data = load_pkl_format_flame_model(model_path)
        new_path = model_path.with_suffix(".pt")
        if new_path.exists() and not overwrite:
            print(f"Skipping {new_path} because it already exists.")
            continue
        torch.save(data, new_path)
        print(f"Saved {new_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flame_base_dir",
        type=str,
        help="Path to the FLAME model directory. "
        "Defaults to the FLAME_BASE_DIR environment variable.",
        default="FLAME2020/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    convert_flame(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
