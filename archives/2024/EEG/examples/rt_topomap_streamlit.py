#%% Show topographic maps in real-time (simplified version of rt_topomap.py from FCBG)
import time
import uuid
import numpy as np
import streamlit as st

from matplotlib import pyplot as plt
from mne import set_log_level

from mne_lsl.datasets import sample
from mne_lsl.player import PlayerLSL as Player
from mne_lsl.stream import StreamLSL as Stream
from mne.viz import plot_topomap

from scipy.signal import periodogram
from scipy.integrate import simpson

set_log_level("WARNING")

#%% Streamlit UI
st.title("🔬 Real-Time EEG Topomap Visualization")

# Initialize session state variables
if "running" not in st.session_state:
    st.session_state.running = False

# Buttons
if st.button("▶ Start Stream"):
    st.session_state.running = True

if st.button("🛑 Stop Stream"):
    st.session_state.running = False

# Create a placeholder for the topomap
topomap_placeholder = st.empty()

#%% Connect to EEG stream
# variables
duration = 30
bufsize = 5
winsize = 3

try:
    # if the DSI-24 is connected, use the DSI-24 stream
    source_id = "Gwennie-24"
    stream = Stream(bufsize=bufsize, source_id=source_id) # 300 samples/s * bufsize samples in stream.n_buffer 
    stream.connect(acquisition_delay=0.1, processing_flags="all")
    print("Connected to DSI-24 stream")
except:
    # if the DSI-24 is not connected, use the sample EEG stream
    from mne_lsl.player import PlayerLSL as Player
    from mne_lsl.datasets import sample
    source_id = "44d28a9c6af6455f932228fc27c799b2"
    fname = sample.data_path() / "sample-ant-raw.fif"
    player = Player(fname, chunk_size=300, source_id=source_id).start()
    stream = Stream(bufsize=bufsize, source_id=source_id) # 300 samples/s * bufsize samples in stream.n_buffer 
    stream.connect(acquisition_delay=0.1, processing_flags="all")
    print("Connected to sample stream")

stream = Stream(bufsize=2, source_id=source_id).connect()
stream.pick("eeg")
stream.set_montage("standard_1020")
fs = stream.info["sfreq"] 

# main loop
start = time.time()

while st.session_state.running:
    data, _ = stream.get_data(winsize)

    # update topomap
    band = (8,13) # frequency band
    freqs, psd = periodogram(data, fs)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    
    bandpower = simpson(psd[:, idx_band], dx=freq_res)
    metric = bandpower

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_topomap(metric, stream.info, axes=ax, show=False)

    # Update Streamlit UI
    topomap_placeholder.pyplot(fig)

    # print(metric)
    time.sleep(0.1)

stream.disconnect()
player.stop()
# st.write("### ✅ Stream Disconnected!")
