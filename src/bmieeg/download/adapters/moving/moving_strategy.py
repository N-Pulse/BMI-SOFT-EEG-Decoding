# ================================================================
# 0. Section: IMPORTS
# ================================================================
import hashlib
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from .moving_specs import MOVINGSpecs
from ...domain import DownloadStrategy, Registry


CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = (30, 300)
DOWNLOAD_RETRIES = 5


# ================================================================
# 1. Section: Functions
# ================================================================
@Registry.register("moving")
@dataclass
class MOVINGStrategy(DownloadStrategy):
    def validate(self, spec: MOVINGSpecs) -> None:
        _resolve_files(spec)

    def fetch(self, spec: MOVINGSpecs) -> None:
        remote_files = _resolve_files(spec)
        destination = Path(spec.dest)
        destination.mkdir(parents=True, exist_ok=True)

        for remote_file in remote_files:
            filename = remote_file["key"]
            expected_size = int(remote_file["size"])
            expected_checksum = _md5_checksum(remote_file["checksum"])
            archive_path = _safe_destination(destination, filename)
            partial_path = archive_path.with_suffix(archive_path.suffix + ".part")

            if _is_valid_file(archive_path, expected_size, expected_checksum):
                print(f"Skipping {filename}: already downloaded and verified.")
                continue

            archive_path.parent.mkdir(parents=True, exist_ok=True)
            actual_checksum = _stream_download(
                _download_url(remote_file),
                partial_path,
                expected_size,
                filename,
            )
            actual_size = partial_path.stat().st_size

            if actual_size != expected_size:
                raise ValueError(
                    f"Size mismatch for {filename!r}: expected "
                    f"{expected_size} bytes, received {actual_size} bytes"
                )

            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {filename!r}: expected "
                    f"{expected_checksum}, received {actual_checksum}"
                )

            partial_path.replace(archive_path)


# ----------------------------------------------------------------
# 1.1 Subsection: Helper Functions
# ----------------------------------------------------------------
def _resolve_files(spec: MOVINGSpecs) -> list[dict]:
    metadata = _get_metadata(spec.record_id)
    remote_files = metadata.get("files", [])

    if spec.filename is None:
        if not remote_files:
            raise ValueError(f"Zenodo record {spec.record_id} contains no files")
        return remote_files

    for remote_file in remote_files:
        if remote_file.get("key") == spec.filename:
            return [remote_file]

    available_files = [remote_file.get("key") for remote_file in remote_files]
    raise ValueError(
        f"File {spec.filename!r} was not found in Zenodo record {spec.record_id}. "
        f"Available files: {available_files}"
    )


def _get_metadata(record_id: int) -> dict:
    response = requests.get(
        f"https://zenodo.org/api/records/{record_id}",
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _download_url(remote_file: dict) -> str:
    links = remote_file.get("links", {})
    url = links.get("content") or links.get("download") or links.get("self")
    if not url:
        raise ValueError(f"No download URL was provided for {remote_file.get('key')!r}")
    return url


def _md5_checksum(checksum: str) -> str:
    algorithm, separator, value = checksum.partition(":")
    if separator != ":" or algorithm.lower() != "md5" or not value:
        raise ValueError(f"Expected a Zenodo MD5 checksum, received {checksum!r}")
    return value.lower()


def _stream_download(
    url: str,
    destination: Path,
    expected_size: int,
    description: str,
) -> str:
    digest = hashlib.md5()

    downloaded = destination.stat().st_size if destination.exists() else 0
    if downloaded > expected_size:
        destination.unlink()
        downloaded = 0

    if downloaded:
        with destination.open("rb") as partial:
            for chunk in iter(lambda: partial.read(CHUNK_SIZE), b""):
                digest.update(chunk)

    with tqdm(
        total=expected_size,
        initial=downloaded,
        desc=description,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        retries = 0
        while downloaded < expected_size:
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

            try:
                with requests.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=REQUEST_TIMEOUT,
                ) as response:
                    if downloaded and response.status_code == requests.codes.ok:
                        # The server ignored Range; restart once from the beginning.
                        destination.unlink()
                        digest = hashlib.md5()
                        downloaded = 0
                        progress.reset()
                        continue

                    if downloaded and response.status_code != requests.codes.partial_content:
                        raise requests.HTTPError(
                            f"Expected HTTP 206 for resume, received "
                            f"{response.status_code}"
                        )

                    response.raise_for_status()
                    with destination.open("ab" if downloaded else "wb") as output:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            output.write(chunk)
                            digest.update(chunk)
                            downloaded += len(chunk)
                            progress.update(len(chunk))

                if downloaded == 0:
                    raise requests.RequestException("The server returned an empty response")
                retries = 0
            except requests.RequestException as error:
                retries += 1
                if retries > DOWNLOAD_RETRIES:
                    raise RuntimeError(
                        f"Download failed after {DOWNLOAD_RETRIES} retries at "
                        f"{downloaded} of {expected_size} bytes"
                    ) from error
                print(
                    f"Connection interrupted at {downloaded} of {expected_size} bytes; "
                    f"retrying ({retries}/{DOWNLOAD_RETRIES})..."
                )

    return digest.hexdigest()


def _is_valid_file(path: Path, expected_size: int, expected_checksum: str) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return _file_md5(path) == expected_checksum


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_destination(destination: Path, filename: str) -> Path:
    destination_root = destination.resolve()
    file_path = (destination_root / filename).resolve()
    if not file_path.is_relative_to(destination_root):
        raise ValueError(f"Unsafe Zenodo filename: {filename!r}")
    return file_path