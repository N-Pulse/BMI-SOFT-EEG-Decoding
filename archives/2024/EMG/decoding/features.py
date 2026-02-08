import numpy as np

# Mean absolute value (MAV)
def mav(x):
    return np.mean(np.abs(x), axis=0)

# Standard Deviation (STD)
def std(x):
    return np.std(x, axis=0)

# Variance
def var(x):
    return np.var(x, axis=0)

# Maximum absolute Value (MaxAV)
def maxav(x):
    return np.max(np.abs(x), axis=0)

# Root mean square (RMS)
def rms(x):
    return np.sqrt(np.mean(x**2, axis=0))

# Waveform length (WL)
def wl(x):
    return np.sum(np.abs(np.diff(x, axis=0)), axis=0)

# Slope sign changes (SSC)
def ssc(x):
    return np.sum((np.diff(x, axis=0)[:-1] * np.diff(x, axis=0)[1:]) < 0, axis=0)

# Zero Crossing (ZC)
def zc(x):
    return np.sum(np.diff(np.sign(x), axis=0) != 0)

# Log detector
def log_det(x):
    return np.exp(1 / len(x) * np.sum(np.log(x), axis=0))

# Willison amplitude
def wamp(x):
    return np.sum((x > 0.2 * np.std(x)), axis=0)

# Frequency domain features (FFT-based)
def fft_values(x):
    return np.fft.fft(x, axis=0)

def fft_magnitude(x):
    return np.abs(fft_values(x))

def fft_power(x):
    return np.square(fft_magnitude(x))

def freqs(x):
    return np.fft.fftfreq(x.shape[0], d=1/1000)  # Assuming a sampling rate of 1000 Hz

# Total power
def total_power(x):
    return np.sum(fft_power(x), axis=0)

# Mean frequency
def mean_freq(x):
    return np.sum(freqs(x) * fft_power(x), axis=0) / np.sum(fft_power(x), axis=0)

# Median frequency
def median_freq(x):
    return np.median(freqs(x) * fft_power(x), axis=0)

# Peak frequency
def peak_freq(x):
    return freqs(x)[np.argmax(fft_power(x), axis=0)]

# Gesture IDs
ID_TO_GESTURE = {
    0: "Rest",
    1: "ThumbUp",
    2: "Scisors",
    5: "Palm",
    6: "Fist",
    9: "WristRotIn",
    10: "WristRotExt",
    13: "WristFlexion",
    14: "WristExt",
}

# npulse_gestures = {"Rest", "Fist", "WristFlexion", "WristExt", "WristRotIn", "WristRotExt", "Palm"}
npulse_gestures = {"Rest", "Fist"}

GESTURE_TO_ID = {v: k for k, v in ID_TO_GESTURE.items() if v in npulse_gestures}