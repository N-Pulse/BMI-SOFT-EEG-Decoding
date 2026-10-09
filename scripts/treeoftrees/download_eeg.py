from pathlib import Path
from src.download import Downloader, MOVINGSpecs

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    spec = MOVINGSpecs(dest=str(ROOT / "data" / "raw" / "moving"), filename="edf.rar")
    Downloader().run(spec)