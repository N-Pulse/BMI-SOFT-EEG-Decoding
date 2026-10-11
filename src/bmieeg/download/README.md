# Download

Ported from the EMG repo (`src/bmiemg/download`).

Layers: `service` (`Downloader`) → `domain` (specs, strategies, registry) → `adapters` (one per source). To add a source, subclass `DownloadStrategy` + `DownloadSpec` under `adapters/` and decorate the strategy with `@Registry.register("<source-name>")`.

## Archive source

Pulls files from EPFL's WebDAV archive (`make-archives.epfl.ch`).

Sessions are uploaded there in BIDS format after acquisition. One `.xdf` holds both
EEG and EMG (64-channel ANT Neuro EEG plus EMG on the AUX channels), split into
`eeg/` and `emg/` folders by the BIDS conversion. This repo reads the same archive
root as the EMG repo and skips the `emg/` folders, via `extra={"ignore": ["emg"]}`
in `scripts/download/download_archive.py`.

1. Requires Python **3.12 or newer** (`domain/download_strategy.py` uses PEP 695
   generic syntax). Install dependencies:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your credentials (never commit them):

   ```env
   MAKER_USERNAME=<your username>
   MAKER_APP_PASSWORD=<your app password>
   MAKER_DAV_USER=<your dav user>
   ```

3. Set the remote `url` and local `dest` in `scripts/download/download_archive.py`, then run it (it loads `.env` automatically):

   ```bash
   python -m scripts.download.download_archive
   ```
   Run it from the project root, as a module (`-m`), so that `from src.download import ...`
   resolves.

### Getting the credentials

- **`MAKER_USERNAME`** — Is your EPFL username
- **`MAKER_APP_PASSWORD`** — Go to Profile Settings (top right) on the Archive MAKE Drive > Security > Create New App Password
- **`MAKER_DAV_USER`** — Go to Files Settings (bottom left) on the Archive MAKE drive > WebDAV > Copy the URL and past the code in the URL that comes after https://make-archives.epfl.ch/remote.php/dav/files/
