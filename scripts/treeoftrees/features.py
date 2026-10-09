"""
Computes simple features of the EEG windows: the numbers a decision tree learns from.

Every window (channels x time points) is summarized by 16 numbers per channel: 7 about
the signal in time (for example how strong it is or how often it changes direction) and
9 about its frequencies (where the power of the signal is, and how strong each brain
rhythm is).
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import numpy as np

from scripts.treeoftrees.moving_data import DATA, cut_windows, load_subject


# ================================================================
# 1. Section: Time-domain features
# ================================================================
# Every feature takes windows (windows x channels x time points) and returns one
# number per window and channel (windows x channels)
def mav(x: np.ndarray) -> np.ndarray:
    """Mean absolute value: the average size of the signal, ignoring the sign."""
    return np.mean(np.abs(x), axis=2)


def std(x: np.ndarray) -> np.ndarray:
    """Standard deviation: how much the signal moves around its average."""
    return np.std(x, axis=2)


def maxav(x: np.ndarray) -> np.ndarray:
    """Maximum absolute value: the highest peak of the signal."""
    return np.max(np.abs(x), axis=2)


def rms(x: np.ndarray) -> np.ndarray:
    """Root mean square: the strength (energy) of the signal."""
    return np.sqrt(np.mean(x**2, axis=2))


def wl(x: np.ndarray) -> np.ndarray:
    """Waveform length: the total distance the signal travels up and down."""
    return np.sum(np.abs(np.diff(x, axis=2)), axis=2)


def ssc(x: np.ndarray) -> np.ndarray:
    """Slope sign changes: how many times the signal changes direction."""
    dx = np.diff(x, axis=2)
    return np.sum((dx[:, :, :-1] * dx[:, :, 1:]) < 0, axis=2)


def log_det(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Log detector: a smooth strength measure, less sensitive to single peaks."""
    return np.exp(np.mean(np.log(np.abs(x) + eps), axis=2))


# ================================================================
# 2. Section: Frequency-domain features
# ================================================================
# The MOVING recordings have 500 samples per second
SAMPLING_HZ: float = 500.0


def fft_power(x: np.ndarray) -> np.ndarray:
    """
    One-sided FFT power spectrum.

    Returns:
        shape (n_epochs, n_channels, n_freqs)
    """
    fft = np.fft.rfft(x, axis=2)
    return np.abs(fft) ** 2


def fft_freqs(x: np.ndarray, sfreq: float) -> np.ndarray:
    """
    One-sided FFT frequency vector.

    Returns:
        shape (n_freqs,)
    """
    return np.fft.rfftfreq(x.shape[2], d=1.0 / sfreq)


def total_power(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Total spectral power."""
    power = fft_power(x)
    return np.sum(power, axis=2)


def mean_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Mean frequency."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    denom = np.sum(power, axis=2)
    denom = np.maximum(denom, 1e-12)

    return np.sum(power * f[None, None, :], axis=2) / denom


def median_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Median frequency based on cumulative spectral power."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    cumulative_power = np.cumsum(power, axis=2)
    half_power = cumulative_power[:, :, -1:] / 2.0

    idx = np.argmax(cumulative_power >= half_power, axis=2)

    return f[idx]


def peak_freq(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Peak frequency."""
    power = fft_power(x)
    f = fft_freqs(x, sfreq)

    idx = np.argmax(power, axis=2)

    return f[idx]


def band_power(x: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    """Power of the signal between two frequencies (Hz): how strong one rhythm is."""
    # 1. Take the power of every frequency
    power = fft_power(x)
    freqs = fft_freqs(x, sfreq)

    # 2. Add up the powers of the frequencies inside the band
    return np.sum(power[:, :, (freqs >= low) & (freqs < high)], axis=2)


def theta(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Theta rhythm (4-8 Hz): slow waves."""
    return band_power(x, sfreq, 4, 8)


def alpha(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Alpha rhythm (8-13 Hz): the resting rhythm, called mu over the motor area."""
    return band_power(x, sfreq, 8, 13)


def beta_low(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Low beta rhythm (13-20 Hz): linked to movement, it gets weaker when we move."""
    return band_power(x, sfreq, 13, 20)


def beta_high(x: np.ndarray, sfreq: float) -> np.ndarray:
    """High beta rhythm (20-30 Hz): faster waves, also linked to movement."""
    return band_power(x, sfreq, 20, 30)


def gamma(x: np.ndarray, sfreq: float) -> np.ndarray:
    """Gamma rhythm (30-45 Hz): the fastest waves we keep."""
    return band_power(x, sfreq, 30, 45)


# ================================================================
# 3. Section: Mapped
# ================================================================
TIME_FEATURE_FUNCTIONS = [mav, std, maxav, rms, wl, ssc, log_det]
FREQ_FEATURE_FUNCTIONS = [
    total_power,
    mean_freq,
    median_freq,
    peak_freq,
    theta,
    alpha,
    beta_low,
    beta_high,
    gamma,
]

# The features the model uses. The frequency ones are slow on the board: for the board
# keep only TIME_FEATURE_FUNCTIONS
FEATURE_FUNCTIONS = TIME_FEATURE_FUNCTIONS + FREQ_FEATURE_FUNCTIONS


# ================================================================
# 4. Section: FUNCTIONS
# ================================================================
def get_features(windows: np.ndarray, sfreq: float = SAMPLING_HZ) -> np.ndarray:
    """Compute the features of every channel of every window.

    Returns a table with one row per window and one column per feature and
    channel: first the values of the first feature for all channels, then of the
    second feature, and so on.
    """
    columns = []
    for function in FEATURE_FUNCTIONS:
        # The frequency features need to know how many samples there are per second
        if function in FREQ_FEATURE_FUNCTIONS:
            columns.append(function(windows, sfreq))
        else:
            columns.append(function(windows))
    return np.concatenate(columns, axis=1)


def get_feature_names(channels: list[str]) -> list[str]:
    """Name the columns of get_features, for example "rms_C4"."""
    return [
        f"{function.__name__}_{channel}"
        for function in FEATURE_FUNCTIONS
        for channel in channels
    ]


# ================================================================
# 5. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Load one person and cut the windows
    raw = load_subject(sorted(DATA.glob("*.edf"))[0])
    windows, labels = cut_windows(raw)

    # 2. Compute the features
    features = get_features(windows)
    names = get_feature_names(raw.ch_names)
    print("windows:", windows.shape, "-> features:", features.shape)
    print("first columns:", names[:3], "... last:", names[-1])

    # 3. Look at one feature: the strength (rms) of channel C4 in each class
    column = names.index("rms_C4")
    for name in sorted(set(labels)):
        print(f"{name:15} mean rms_C4 = {features[labels == name, column].mean():.2f}")