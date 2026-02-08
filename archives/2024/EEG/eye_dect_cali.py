import time
import uuid
import numpy as np
from matplotlib import pyplot as plt
from mne import set_log_level
from mne_lsl.datasets import sample
from mne_lsl.stream import StreamLSL
from mne_lsl.player import PlayerLSL as Player

# Suppress verbose MNE output
set_log_level("WARNING")

# Parameters
winsize = 1  # seconds of data to pull
n_blinks = 5  # number of training blinks
cue_interval = 2.5  # time between blink cues
calibration_duration = n_blinks * cue_interval  # total calibration time

# Setup LSL Stream from sample data
source_id = uuid.uuid4().hex
fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname, chunk_size=200, source_id=source_id).start()
fs = player.info["sfreq"]
stream = StreamLSL(bufsize=2, source_id=source_id).connect()
stream.pick("Fp1")
stream.set_montage("standard_1020")
stream.filter(2, 25)

# Calibration: blink training with visual cue
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Blink when you see the cue!", fontsize=14)

training_data = []
baseline_data = []

print("Starting blink training. Please blink when cued...")

start_time = time.time()
for i in range(n_blinks):
    ax.clear()
    ax.set_title(f"Blink #{i + 1} NOW!", fontsize=16, color="red")
    plt.draw()
    plt.pause(0.5)

    time.sleep(0.3)
    data, _ = stream.get_data(winsize=winsize)
    training_data.append(data)

    ax.set_title("...wait...", fontsize=14)
    plt.draw()
    plt.pause(0.01)

    time.sleep(cue_interval - 0.8)

# Flatten training data and compute baseline & threshold
training_data = np.concatenate(training_data, axis=1).ravel()
baseline = np.mean(training_data)
std_dev = np.std(training_data)
threshold = baseline + 4 * std_dev

print(f"Calibration complete. Baseline: {baseline:.2f}, STD: {std_dev:.2f}, Threshold: {threshold:.2f}")

# Real-time blink detection
detected_blinks = []
ax.clear()
ax.set_title("Real-time blink detection")
plt.draw()

run_duration = 15  # seconds
start = time.time()

while time.time() - start < run_duration:
    data, _ = stream.get_data(winsize=winsize)
    signal = data.ravel()

    blink_detected = np.any(np.abs(signal - baseline) > 4 * std_dev)
    if blink_detected:
        detected_blinks.append(time.time())
        print(f"Blink detected at {time.time() - start:.2f}s")

    ax.clear()
    ax.plot(signal, label="EEG (Fp1)")
    ax.axhline(baseline, color='green', linestyle='--', label='Baseline')
    ax.axhline(threshold, color='red', linestyle='--', label='Blink Threshold')
    ax.axhline(baseline - 4 * std_dev, color='red', linestyle='--')
    ax.legend()
    plt.pause(0.01)

plt.ioff()
plt.close()
stream.disconnect()
try:
    player.stop()
except:
    pass

print(f"Total blinks detected: {len(detected_blinks)}")
