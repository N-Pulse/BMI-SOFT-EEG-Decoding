# Prune Dataset (EEG)
A pipeline that turns recordings into the files the model trains on. A protocol (a YAML file) lists the steps to run, and the code only executes that list. Several protocols can live side by side, each producing a different dataset.

It is copied from the EMG repo (`src/bmiemg/prune`) with the name changed to `bmieeg`, and it keeps the same design. What is different for EEG is listed in section 8.

---

## 1. Architecture
The package follows a three-layer split:

| Layer | Folder | Contents |
|---|---|---|
| Service | `service/` | `DatasetPruner`: wires everything together and runs a protocol. No logic of its own. |
| Domain | `domain/` | `DataActor`s (the data), `PruneStep`s (the transformations), `Protocol`. No I/O. |
| Adapter | `adapters/` | `Source` (paths), `Loader` + `FileLoader`s (read files), `ProtocolLoader` (read YAML), `Exporter` + `Writer`s (write files). |

The pipeline is: **load files → actors → steps (in order) → actors → writers → files**.

Every extensible family (`PruneStep`, `FileLoader`, `Writer`) is plugged in through a registry built on `bmieeg.utils.Registry`. A class registers itself with a decorator under a string key, and the YAML or the file suffix refers to that key. Adding a new implementation never requires touching the service or the loaders.

## 2. Data actors
Actors are dataclasses carried between steps. Each step declares which actor types it takes and returns (`PruneStep[InT, OutT]`), so the order of the steps matters.

| Actor | Produced by | Contents |
|---|---|---|
| `TrialData` | the file loaders, `CutTrialsStep` | `signal_ch` `(channels, time)`, per-sample `markers` `(1, time)`, `channel_names`, `sfreq`, and `windows`, the list of the trials of the recording (`TrialWindow`: start and end in seconds, movement, repetition, orientation). It is a whole recording until `cut_trials`, then one trial. |
| `DatasetPartition` | `TTVSplitStep` | One of `train`/`val`/`test` and its trials. |
| `Dataset` | `TTVSplitStep` | `client` and `server` partition lists. |

`TrialData` inherits from `MetadataActor`, which derives `session_number`, `subject_number` and `date` from the `bids_local` folder layout (`<date>/<subject>/<session>/<modality>/<file>`). Loaders for sources without that layout set `subject_id` instead, which `subject_number` returns first. The people of MOVING get the number 1001 and up, so they never have the number of one of ours.

---

## 3. Protocols
Each protocol is a YAML file in `configs/prune/<protocol_name>.yaml`:

- `steps:` lists the steps in run order, keyed by each step's registry key with its constructor parameters (`{}` for none).
- `export:` lists the writers to run, keyed by the writer's registry key, with their parameters.

### eeg_board_v1
Adapts our own recordings (gel electrodes, `data/bids`) and MOVING (`data/raw/moving/edf`) to the board and splits the trials 70/15/15 for every person:

`channel_select` (the 14 electrodes of the board) → `analogue_filter` (causal first-order 0.2 Hz high-pass) → `analogue_filter` (causal first-order 80 Hz low-pass) → `resample` (160 Hz) → `to_full_scale` (full scale of the converter) → `adc_quantize` (24 bits) → `cut_trials` → `ttv_split`, exported with `dataset`.

Every step before `cut_trials` works on a whole recording, like the board works on a continuous signal. With the same band, rate and converter, the result is the same, number by number, as `scripts/ttv/adapt.py` and `scripts/ttv/board.py`.

---

## 4. Steps
Every step is a `PruneStep`: it receives a list of `DataActor`s and returns the transformed list. Parameters are the step's own dataclass fields, so the attributes listed below are exactly the YAML keys.

### ChannelSelectStep - `channel_select` (`TrialData → TrialData`)
Keeps the channels of a list, in the order of the list and with its names (the recordings write `CZ`, the board writes `Cz`). A channel that moves less than `flat_microvolts` is flat and is dropped. A recording without any of the channels raises `ValueError`.

- `channels: list[str]`
- `flat_microvolts: float = 1e-6`

### AnalogueFilterStep - `analogue_filter` (`TrialData → TrialData`)
Applies a Butterworth filter emulating the analogue front end of the board. With `causal: true` it only looks at the past, like the real filter, and starts as if the signal had always been at its first value.

- `btype: "bandpass" | "bandstop" | "lowpass" | "highpass"`
- `cutoff: float | [low, high]`
- `order: int = 1`
- `causal: bool = False`: `False` is zero-phase (`sosfiltfilt`), it keeps the signal aligned with the markers.

### ResampleStep - `resample` (`TrialData → TrialData`)
Polyphase-resamples the signal (`scipy.signal.resample_poly`, the filter against aliasing is included) to a target rate. Markers are held from their source sample.

- `sfreq: float`: target sampling rate in Hz.
- `padtype: str = "constant"`

### ToFullScaleStep - `to_full_scale` (`TrialData → TrialData`)
Writes the signal (microvolts) as a part of the full scale of the converter, from -1 to 1, which is what `adc_quantize` needs.

- `full_scale_microvolts: float`: the biggest signal the converter can show, plus or minus.

### AdcQuantizeStep - `adc_quantize` (`TrialData → TrialData`)
Quantizes the part of the full scale to the signed counts of the converter: `rint(x * 2^(bits-1))`, saturated to `[-2^(bits-1), 2^(bits-1) - 1]` (a signal that is too big stays at the limit, it does not wrap around), stored as `int16` (`int32` above 16 bits).

- `bits: int`: resolution of the converter, 2 to 32.

### CutTrialsStep - `cut_trials` (`TrialData → TrialData`)
Cuts every trial listed in `windows` out of the whole recording. The markers of a trial hold the code of its movement all the way through.

- `movement_codes: dict[str, int]`: name of the movement to a code from 1 to 99. A movement without a code raises `ValueError`.

### TTVSplitStep - `ttv_split` (`TrialData → Dataset`)
Shuffles the trials (seeded) into train/val/test, optionally for every person or every movement on its own. It produces one `Dataset`:
- `client`: `train`, `val` and `test`. The client test markers are zeroed out.
- `server`: `test` with its markers, for scoring.

- `train_ratio`, `val_ratio`, `test_ratio: float`: must sum to 1.
- `user_balance: bool = False`, `movement_balance: bool = False`
- `seed: int = 0`
- `group_by: "subject" | None = None`: whole people in one part (it cannot be combined with the balances).

---

## 5. File loaders
`Loader` walks `data/<source_name>/` recursively and gives each file to the `FileLoader` registered for its suffix. Each `FileLoader` implements `load_actors(path)`.

| Class | Suffix | Produces |
|---|---|---|
| `LocalSessionLoader` | `.fif` | One `TrialData` for a whole session of ours (`<name>_eeg.fif` with its `_events.tsv` next to it, in microvolts). The trials go from the prep signal to the end of a normal iti. Sessions with fewer than 12 trials are test recordings and are left out. |
| `MovingEdfLoader` | `.edf` | One `TrialData` for a whole person of MOVING (in volts, turned into microvolts). The three blocks of one gesture (rest, imagined, executed) with the cross before them are one trial. |

## 6. Writers
`Exporter` runs each writer from the protocol's `export:` section. Output goes to `data/dataset/<protocol_name>/`, and every folder of one run shares a UTC timestamp.

| Class | Key | Writes | Parameters |
|---|---|---|---|
| `DatasetWriter` | `dataset` | `Dataset` → `client/` and `server/`, in each a folder `<partition>_<timestamp>/` with one `<trial>_raw.fif` for every trial | `fmt: ".fif"`, `zip_groups: bool = True` |

The counts of the converter are whole numbers that a `.fif` file keeps exactly (up to 2^24).

---

## 7. Usage
Run from the repo root. Paths default to `data/` and `configs/prune/`.

```bash
PYTHONPATH=src python scripts/01_prune_dataset/prune_eeg.py
```

or, in Python:

```python
from bmieeg.prune import DatasetPruner

pruner = DatasetPruner(source_name=["bids", "raw/moving/edf"])   # folders under data/
actors = pruner.run("eeg_board_v1")                               # load + apply steps
pruner.export(actors, "eeg_board_v1")                             # write files (explicit)
```

`run` doesn't write anything, so call `export` when you want files. To try steps without writing a YAML, pass them as `custom_steps`.

To add a step, subclass `PruneStep`, register it with `@StepRegistry.register("my_step")`, export it from `domain/steps/__init__.py` and document it in section 4. The same works for a loader (`FileLoaderRegistry`) and a writer (`WriterRegistry`). The import in the package `__init__.py` is required, because registration happens at import time.

## 8. Differences from the EMG package
- No XDF steps yet (`signal_split`, `signal_unwarp`, `protocol_definer`, `remove_session`, `marker_unification`, `build_epochs`) and no ADC normalize step: the loaders of the archive `.xdf` files are not ported.
- New: `ChannelSelectStep`, `ToFullScaleStep`, `CutTrialsStep`, `LocalSessionLoader`, `MovingEdfLoader`, `TrialWindow`.
- `AnalogueFilterStep` (causal) starts from the first value of the signal. An electrode has an offset of millivolts, a filter that starts from zero rings for seconds.
- `DatasetWriter` writes one file for every trial, because the trials do not all have the same electrodes (MOVING has 2 to 6 of the 14 board electrodes, our sessions have 12 or 13), so they cannot be joined into one `Raw`.
- `ttv_split` is the random split of the EMG repo, for the trials of every person. It is not the split of `artifacts/ttv_split/ttv_split.csv`.
- The full scale in `eeg_board_v1.yaml` (+- 83886.08 microvolts, the range of our local recorder) is a placeholder until the hardware team gives the range of the board.
