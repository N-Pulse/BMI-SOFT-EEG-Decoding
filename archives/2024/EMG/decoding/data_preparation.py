import numpy as np
import pandas as pd
import scipy.io as sio # for loading MATLAB files
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #sets the matplotlib style to seaborn style
from scipy.io import loadmat 
from scipy.ndimage import convolve1d
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, welch
import os

raw_data_path = 'data/npulse/raw/' # path to raw data
processed_data_path = 'data/npulse/cleaned/' # path to processed data

# Create directories if they do not exist
os.makedirs(raw_data_path, exist_ok=True)
os.makedirs(processed_data_path, exist_ok=True)

files_list = os.listdir(raw_data_path)

if len(files_list) == 0:
    print("No data found in data/npulse/raw repository")
    exit()

def hampel_filter(series, window_size=3, n_sigmas=3):
    """Applies Hampel filter to a 1D pandas Series."""
    L = 1.4826  # scale factor for Gaussian distribution
    rolling_median = series.rolling(window=2 * window_size + 1, center=True).median()
    diff = np.abs(series - rolling_median)
    mad = L * diff.rolling(window=2 * window_size + 1, center=True).median()
    
    # Identify outliers
    outlier_idx = diff > n_sigmas * mad
    filtered = series.copy()
    filtered[outlier_idx] = rolling_median[outlier_idx]
    return filtered

# def band_pass_filter(series, timestamps, LF = 10, HF = 400):
#     filtered = series
#     return filtered

def compute_sampling_freq(timestamps):
    sampling_freq = 1 / np.mean(np.diff(timestamps))
    jitter = np.mean(np.diff(timestamps))
    print(f"Sampling Frequency: {sampling_freq} Hz")
    print(f"Jitter: {jitter} seconds")

    return (sampling_freq, jitter)

def normalize_window(series, timestamps, baseline_noise):
    'Return a preprocessed window (filtred, denoised) of the signal taking the mean and standard deviation of a reference noise'
    baseline_mean = baseline_noise[0]
    baseline_std = baseline_noise[1]
    normalized_series = (series-baseline_noise)/baseline_std

    return normalized_series
    