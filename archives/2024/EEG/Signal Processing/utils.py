import mne
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as plx
from mne.datasets import misc
import mne_connectivity
import pyxdf
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

channels_to_use = [
    # prefrontal
    'Fp1',
    'Fp2',
    # frontal
    'F7',
    'F3',
    'Fz',
    'F4',
    'F8',
    # central and temporal
    'T3',
    'C3',
    'Cz',
    'C4',
    'T4',
    # parietal
    'T5',
    'P3',
    'Pz',
    'P4',
    'T6',
    # occipital
    'O1',
    'O2',
]

def build_mne_object(fname, eeg_stream_name='Gwennie-24', stimulus_stream = 'stimulus_stream'):
    streams, header = pyxdf.load_xdf(fname)

    # Find the index of the stimulus and EEG streams
    eeg_index = []
    for stream in range(len(streams)):
        if streams[stream]["info"]["name"][0] == eeg_stream_name:
            eeg_index.append(stream)
    
    # The EEG channels are assumed to be constant across streams 
    # because this is built into the DSI-24 system
    eeg_index1 = eeg_index[0]
    ch_names = []
    for i in range(0, len(streams[eeg_index1]["info"]["desc"][0]["channels"][0]["channel"])):
        ch_names.append(streams[eeg_index1]["info"]["desc"][0]["channels"][0]["channel"][i]["label"][0])
    
    # Create the info object
    samp_frq = float(streams[eeg_index1]["info"]["nominal_srate"][0])
    ch_types = ['eeg'] * len(ch_names)

    # Find the stimulus stream in streams
    stimulus_stream = None
    for stream in range(len(streams)):
        if streams[stream]["info"]["name"][0] == "stimulus_stream":  # Match name
            stimulus_stream = streams[stream]
            break

    if stimulus_stream is None:
        raise ValueError("No 'stimulus_stream' found in the dataset.")


    # Extract stimulus timestamps and event markers
    first_timestamp =float(stimulus_stream["footer"]["info"]["first_timestamp"][0])

    event_timestamps = stimulus_stream["time_stamps"] 
    eeg_timestamps = streams[eeg_index1]["time_stamps"]
    event_index = np.searchsorted(eeg_timestamps, event_timestamps)

    event_dict = stimulus_stream["time_series"].flatten()  # Convert to 1D array

    # format the events array to correspond to what MNE expects
    events = np.column_stack([
        (event_index).astype(int),
        np.zeros(len(event_timestamps), dtype=int),
        event_dict
    ])

    info = mne.create_info(ch_names, sfreq = samp_frq, ch_types= ch_types, verbose=None)

    # Create the raw object 
    # Here we assume that there is only one EEG stream    
    eeg_data = streams[eeg_index1]["time_series"].T
    # # uV -> V
    eeg_data *= 1e-6  
    raw = mne.io.RawArray(eeg_data, info, verbose=None)

    fs = raw.info['sfreq']
    print(f'Frequency of Sampling: {fs} Hz')
    # Length in seconds
    print(f'Duration: {len(raw) / fs} seconds')

    return raw, events, samp_frq


def set_1020_montage(raw):
    """
    Set the 10-20 montage for the EEG data.
    Parameters
    ----------
    raw : mne.io.Raw
        The raw EEG data.
    """
    # Set the montage to 10-20 system
    sample_1020 = raw.copy().pick_channels(channels_to_use)
    assert len(channels_to_use) == len(sample_1020.ch_names)
    ch_map = {ch.lower(): ch for ch in sample_1020.ch_names}
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    len(ten_twenty_montage.ch_names)
    ten_twenty_montage.ch_names = [ch_map[ch.lower()] if ch.lower() in ch_map else ch 
                                for ch in ten_twenty_montage.ch_names]
    sample_1020.set_montage(ten_twenty_montage)
    return sample_1020


def extract_power_matrix(tfr=None, freq_bins=None):
    max_t = tfr.tmax
    time_bins = [(start, min(start + 0.2, max_t)) for start in np.arange(0, max_t, 0.2)]
    n_channels = len(tfr.ch_names)
    mat = np.zeros((n_channels, len(freq_bins), len(time_bins)))

    for i_f, (fmin, fmax) in enumerate(freq_bins):
        for i_t, (tmin, tmax) in enumerate(time_bins):
            tf_crop = tfr.copy().crop(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
            mean_power = tf_crop.data.mean(axis=(1, 2))  # (n_channels,)
            mat[:, i_f, i_t] = mean_power
    return mat, time_bins

def create_power_df(epoch_dict, freqs, freq_bins = None):
    n_cycles = freqs / 2.  # typical choice for multitaper
    power_dict = {}
    for epoch_type, epoch in epoch_dict.items():
        #power = tfr_multitaper(epoch, freqs=freqs, n_cycles=n_cycles, return_itc=False, picks='eeg')
        power = epoch.compute_tfr( method='multitaper',
            freqs=freqs,
            n_cycles=n_cycles,         
            picks='eeg',
            )
        matrix, time_bins = extract_power_matrix(tfr = power, freq_bins = freq_bins)
       
       # Append the list of channel-wise power values
        power_dict[epoch_type] = [matrix[i] for i in range(len(channels_to_use))]

    #Create the DataFrame 
    power_df = pd.DataFrame(power_dict, index=channels_to_use)
    
    return power_df


def create_corr_df(epoch_dict, plot=False):
    task_types = list(epoch_dict.keys())
    channel_pairs = list(combinations(channels_to_use, 2))

    # Initialize DataFrame to store correlation values
    fc_df = pd.DataFrame(index=[f"{ch1}-{ch2}" for ch1, ch2 in channel_pairs],
                        columns=task_types)


    # Loop over epoch types and get data
    for epoch_type, epochs in epoch_dict.items():
        epochs_data = epochs.get_data()  # shape: (n_channels, n_times) -> because we take average within this epoch
        print(f"Epoch Type: {epoch_type}, Shape: {epochs_data.shape}")
        ch_names = epochs.info['ch_names']
        epochs_data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
        n_channels = epochs_data.shape[0]

        # Initialize correlation matrix
        corr_matrix = np.zeros((n_channels, n_channels))

        for ch1, ch2 in channel_pairs:
            ch1_idx = ch_names.index(ch1)
            ch2_idx = ch_names.index(ch2)

            # Get the data for the two channels across all epochs and time points
            #print(f"Computing correlation for {ch1} and {ch2}")
            data_ch1 = epochs_data[ch1_idx, :]
            data_ch2 = epochs_data[ch2_idx, :]

            # Compute Pearson correlation for the two channels
            fc_df.loc[f"{ch1}-{ch2}", epoch_type] = pearsonr(data_ch1.flatten(), data_ch2.flatten())[0]  # correlation coefficient
            corr_matrix[ch1_idx, ch2_idx] = fc_df.loc[f"{ch1}-{ch2}", epoch_type]
            corr_matrix[ch2_idx, ch1_idx] = fc_df.loc[f"{ch1}-{ch2}", epoch_type]

        # Plot the matrix
        if plot:
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, xticklabels=ch_names, yticklabels=ch_names, 
                        cmap='RdBu_r', center=0, annot=False, square=True)
            plt.title(f"Functional Connectivity {epoch_type}")
            plt.tight_layout()
            plt.show()

    return fc_df, corr_matrix


def create_temp_df(epoch_dict, sampling_freq):
    sfreq = sampling_freq
    segment_duration = 0.2 
    seg_len = int(segment_duration * sfreq) #number of samples in each segment

    mstd_df = pd.DataFrame(index=channels_to_use, columns=epoch_dict.keys())

    for epoch_type, epochs in epoch_dict.items():
        evoked = epochs.copy().pick(channels_to_use)
        data = evoked.data 
        ch_names = evoked.ch_names
        n_channels, n_times = data.shape
        n_segments = n_times // seg_len #15

        for ch_idx, ch_name in enumerate(ch_names):
            mean_list = []
            std_list = []

            for s in range(n_segments):
                s_start = s * seg_len
                s_end = s_start + seg_len
                seg = data[ch_idx, s_start:s_end] 
                mean_list.append(np.mean(seg))
                std_list.append(np.std(seg))

            mstd_df.at[ch_name, epoch_type] = [mean_list, std_list] # loc didn't work that's why I used at

    return mstd_df

def create_final_df(power_df=None, fc_df=None, mstd_df=None):
    dataframes = []

    if power_df is not None:
        freq_df = power_df.copy()
        freq_df.index = pd.MultiIndex.from_product([['freq_features'], freq_df.index])
        dataframes.append(freq_df)

    if fc_df is not None:
        corr_df = fc_df.copy()
        corr_df.index = pd.MultiIndex.from_product([['corr_features'], corr_df.index])
        dataframes.append(corr_df)

    if mstd_df is not None:
        mstd_df = mstd_df.copy()
        mstd_df.index = pd.MultiIndex.from_product([['mstd_features'], mstd_df.index])
        dataframes.append(mstd_df)

    # Concatenate all non-None DataFrames
    if dataframes:
        combined_df = pd.concat(dataframes)
    else:
        combined_df = pd.DataFrame()  # Return empty if all inputs are None

    return combined_df
