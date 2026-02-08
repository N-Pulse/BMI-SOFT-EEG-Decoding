import time
import uuid
import numpy as np
import matplotlib.pyplot as plt

from mne import set_log_level
from mne.time_frequency import psd_array_welch
from mne_lsl.datasets import sample
from mne_lsl.stream import StreamLSL
from mne_lsl.player import PlayerLSL as Player

# Setup
set_log_level("WARNING")
source_id = uuid.uuid4().hex
fname = sample.data_path() / "sample-ant-raw.fif"
player = Player(fname, chunk_size=200, source_id=source_id).start()
fs = player.info["sfreq"]

# Connect to the stream
stream = StreamLSL(bufsize=2, source_id=source_id).connect()
stream.pick("Fp1", "Fp2")
stream.set_montage("standard_1020")
stream.filter(1, 15)

# Parameters
winsize = 0.5  # window size in seconds
calibration_duration = 10  # seconds
refractory_period = 1.0  # seconds
channels = stream.info["ch_names"]

# Real-time plotting
plt.ion()

# --- Calibration ---
print("Calibrating PSD baseline for 30 seconds...")
psd_values = []

time.sleep(winsize)
start = time.time()
while time.time() - start < calibration_duration:
    data, _ = stream.get_data(winsize)
    for ch in range(data.shape[0]):
        psd, _ = psd_array_welch(data[ch], sfreq=fs, fmin=1, fmax=15, n_fft=256)
        psd_values.append(psd.mean())

psd_values = np.array(psd_values).reshape(-1, len(channels))  # shape: (samples, channels)
baseline = np.mean(psd_values, axis=0)
std = np.std(psd_values, axis=0)
thresholds = baseline + 4 * std

print("\nCalibration Complete:")
for i, ch in enumerate(channels):
    print(f"{ch} -> Baseline: {baseline[i]:.10f}, Threshold: {thresholds[i]:.10f}")

# --- Real-time Blink Detection ---
print("\nStarting real-time detection. Press Ctrl+C to stop.")
last_blink_time = 0
blinks_detected = 0

try:
    while True:
        data, _ = stream.get_data(winsize)
        psd_means = []
        blink_detected = False
        now = time.time()

        for ch in range(data.shape[0]):
            psd, _ = psd_array_welch(data[ch], sfreq=fs, fmin=1, fmax=15, n_fft=256)
            psd_mean = psd.mean()
            psd_means.append(psd_mean)
            if psd_mean > thresholds[ch]:
                blink_detected = True

        # Refractory check
        if blink_detected and (now - last_blink_time > refractory_period):
            blinks_detected += 1
            print(f"[{blinks_detected}] Blink detected! (Time: {time.strftime('%H:%M:%S')})")
            last_blink_time = now

        # Visualization
        plt.clf()
        plt.bar(channels, psd_means, color='skyblue')
        for i, th in enumerate(thresholds):
            plt.axhline(y=th, color='red', linestyle='--')
        if blink_detected:
            plt.text(0.5, 0.95, 'Blink Detected!', fontsize=14, color='red',
                     ha='center', va='top', transform=plt.gca().transAxes)
        plt.title(f"Blink Count: {blinks_detected}")
        plt.pause(0.01)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    plt.ioff()
    plt.close()
    stream.disconnect()
    player.stop()
    print(f"\nTotal blinks detected: {blinks_detected}")
