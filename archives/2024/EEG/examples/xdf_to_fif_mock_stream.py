## change .xdf-file (LabRecorder's standard file) to .fif 
## you can use a .fif-file for PlayerLSL and let it run in a loop

#%%
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream

import time
import numpy as np
import mne
import uuid
source_id = uuid.uuid4().hex

import matplotlib.pyplot as plt

#%% #### CREATE FIF FILE FROM THE XDF DATA (OWN ACQUISITION)
import pyxdf

fname = './DSI/data/sub-P019/ses-S001/eeg/sub-P019_ses-S001_task-Default_run-003_eeg.xdf'
streams, header = pyxdf.load_xdf(fname)
stimulus_stream = streams[0]
eeg_stream = streams[1]
eeg_data = eeg_stream['time_series']
sfreq = int(eeg_stream['info']['nominal_srate'][0])

# Create an MNE Raw object from the EEG data
info = mne.create_info(ch_names=[f"EEG{i}" for i in range(eeg_data.shape[1])], 
                       sfreq=sfreq, ch_types=["eeg"] * eeg_data.shape[1])

raw = mne.io.RawArray(eeg_data.T, info)  # Transpose to match MNE's format

# Save it to a .fif file
# raw.save('./data/sub-P019_ses-S001_task-Default_run-003_eeg_raw.fif', overwrite=True)
 

# %%
