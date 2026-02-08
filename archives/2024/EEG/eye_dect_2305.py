# %%
import time
import uuid
import numpy as np
import matplotlib.pyplot as plt

from mne import set_log_level
from mne_lsl.datasets import sample
from mne_lsl.stream import StreamLSL
from mne_lsl.player import PlayerLSL as Player

set_log_level("WARNING")

# --- Connect to EEG sample stream ---
source_id = uuid.uuid4().hex
#fname = sample.data_path() / "sample-ant-raw.fif"
#player = Player(fname, chunk_size=200, source_id=source_id).start()
#fs = player.info["sfreq"]
stream = StreamLSL(bufsize=2, name = "Gwennie-24").connect()
fs = stream.info["sfreq"]

# Use front channels (eye blinks show up here)
stream.pick("Fp1", "Fp2")
stream.set_montage("standard_1020")
stream.filter(1, 15)

# variables

# --- Calibration phase ---
winsize = 3  # seconds
print("Calibrating for 10 seconds...")
time.sleep(winsize)

calibration_data = []
start_time = time.time()
while time.time() - start_time < 10:
    data, _ = stream.get_data(winsize)
    calibration_data.append(data)


calibration_data = np.concatenate(calibration_data, axis=1)
flat_data = calibration_data.flatten()
baseline = np.mean(flat_data)
std = np.std(flat_data)

baseline = np.mean(calibration_data)
std = np.std(calibration_data)

threshold = 4 * std

print(f"Baseline: {baseline:.10f}, Threshold: ±{threshold:.10f}")

# --- Real-time blink detection loop ---
plt.ion()
blinks_detected = 0
last_blink_time = 0
refractory_period = 1.5  # seconds

try:
    while True:
        data, _ = stream.get_data(winsize=5)  # 1-second window
        signal = data.mean(axis=0)  # average Fp1 & Fp2
        deviation = signal - baseline

        blink = deviation > threshold
        current_time = time.time()

        if np.any(blink):
            if current_time - last_blink_time > refractory_period:
                print("Blink detected!")
                blinks_detected += 1
                last_blink_time = current_time
            else:
                print("Blink ignored (within refractory period)")

        # Optional: real-time visualization
        plt.clf()
        plt.plot(signal, label="Signal")
        plt.hlines([baseline, baseline + threshold, baseline - threshold],
                   0, len(signal), colors="red", linestyles="dashed", label="Thresholds")
        plt.plot(blink * (baseline + threshold), 'k.', label="Blink Detected")
        plt.title(f"Blinks: {blinks_detected}")
        plt.legend()
        plt.pause(0.01)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    plt.ioff()
    plt.close()
    stream.disconnect()
    player.stop()
    print(f"Total blinks detected: {blinks_detected}")
