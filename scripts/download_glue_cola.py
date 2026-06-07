from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


COLA_URL = "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip"


def download_cola(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "CoLA.zip"
    target_dir = data_dir / "CoLA"
    if not (target_dir / "train.tsv").exists():
        urlretrieve(COLA_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
    return target_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="glue_data")
    args = parser.parse_args()
    target = download_cola(Path(args.data_dir))
    print(f"CoLA data is ready at {target}")


if __name__ == "__main__":
    main()
