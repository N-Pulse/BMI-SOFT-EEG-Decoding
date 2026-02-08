# To run a sample EEG stream to tryout written code, you can use the following code

#%% To start the LSL player, run this cell
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.datasets import sample

source_id = "44d28a9c6af6455f932228fc27c799b2"

# choose filename that you want to stream in a loop
    # we can also run a file recorded with the DSI-24 headset! 
    # You can change the file format from .xdf to .fif in xdf_to_fif_mock_stream.py
    # example:
        # fname = './data/sub-P019_ses-S001_task-Default_run-003_eeg_raw.fif'
fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname, chunk_size=300, source_id=source_id).start()
fs = player.info["sfreq"] # 300 Hz
interval = player.chunk_size / fs  # Time between two push operations
print(f"Sampling frequency: {fs}")
print(f"Interval between 2 push operations: {interval} seconds.")

#%% To check the streams, run this cell
from mne_lsl.lsl import resolve_streams
streams = resolve_streams()
print(streams[0])

#%% To stop the LSL player, run this cell
player.stop()

# %%
