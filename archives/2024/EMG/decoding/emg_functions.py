import os
import time
import csv
from pathlib import Path
import joblib
import argparse
import sys
# import serial
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from data_preparation import normalize_window, compute_sampling_freq
from plotting_results import show_results
from features import *

# SERIAL_PORT = 'COM7'
# BAUD_RATE = 115200

def check_filename_contains(file, condition):
    if condition in file:
        return True
    else:
        return False

def load_dataset(path, file_conditions=None):
    """
    Load files in a directory with or without condition on the name of the file.
    Return a dictionnary of data.
    """
    os.makedirs(path, exist_ok=True)
    files_list = os.listdir(path)

    if len(files_list) == 0:
        print("No data found in directory")
        exit()

    dict_data = {}
    if file_conditions is None:
        for file in files_list:
            print(f"Loading {file}...")
            raw_data = pd.read_csv(os.path.join(path, file))
            dict_data[file] = raw_data

    else:
        for file in files_list:
            if file_conditions in file:
                print(f"Loading {file}...")
                raw_data = pd.read_csv(os.path.join(path, file))
                dict_data[file] = raw_data
    
    return dict_data


# def measure_resting_state(port, nb_channels, duration=5):
#     '''
#     Online : record and return the mean and standard deviation for resting state signal
#     '''

#     try: 

#         header = ["Timestamp"]
#         for i in range(nb_channels):
#             header.append(f"Channel{i+1}")
        
#         resting_data = []

#         ser = serial.Serial(port, BAUD_RATE, timeout=3)
#         time.sleep(1)

#         print(f"Measure of the resting state during {duration} seconds")
#         input("Press any key to start the recording")
        
#         first_data = None
#         while first_data is None:
#             raw_data = ser.readline().decode('utf-8').strip()
#             channels = []
#             for i in range(nb_channels):
#                 channels.append(float(raw_data.split(" ")[i]))
            
#             if [isinstance(channel, float) for channel in channels].all():
#                 first_data = channels

#         start_recording_time = time.time()
#         current_time = start_recording_time

#         while current_time <= duration:
#             current_time = round(time.time() - start_recording_time, 3)
#             data = ser.readline().decode('utf-8').strip()

#             channels = []

#             if not data:
#                 continue

#             try:
#                 for i in range(nb_channels):
#                     channels.append(float(data.split(" ")[i]))
#             except ValueError:
#                 continue

#             buffer = [current_time]
#             for channel in channels:
#                 buffer.append(channel)

#             resting_data.append(buffer)
        
#         ser.close()

#         resting_df = pd.DataFrame(resting_data, columns=header)
        
#         baseline_mean_std = []

#         for i in range(nb_channels):
#             mean = resting_df[f"Channel{i}"].mean()
#             std = resting_df[f"Channel{i}"].std()
#             baseline_mean_std.append([mean, std])

#         return baseline_mean_std

#     except serial.SerialException as e:
#         print(f"Error: {e}")
#         return
    
def data_preparation(list_data):

    def compute_flat_label(row):
        if pd.isna(row['Action2']): # if Action2 is NaN, return Action1
            return row['Action1']
        else:
            return row['Action2']

    for data in list_data:
        data.drop(index=0, inplace=True)
        print(data.head(1))

        data['Gesture'] = data.apply(compute_flat_label, axis=1)
        # data['Gesture'] = data['Gesture'].map(gesture_to_id)
        data.drop(columns=['Action1', 'Action2'], inplace=True) 

        # Other preprocessing step here...

    return list_data
    
def extract_window_features(window_data, features, channel_columns, label_column='Gesture'):
    feature_vector = [feature(window_data[channel].values) for channel in channel_columns for feature in features]
    label = window_data[label_column].values[0]

    return (feature_vector, label)
    
def extract_features_and_labels(data, features, label_column='Gesture', window=200, step_size=50):
    """
    Extract features and labels from data
    Returns a set of array of features and array of labels
    data : DataFrame
    features : list of string
    """
    channels = pd.DataFrame()
    for col in data.columns:
        if "Channel" in col:
            channels[col] = data[col]
    y = data[label_column]
    
    features_list = []
    labels_list = []

    for i in range(0, len(data) - window + 1, step_size):
        window_data = data.iloc[i:i + window]

        feature_vector, label = extract_window_features(window_data, features, channels.columns, label_column)

        features_list.append(feature_vector)
        labels_list.append(label)
    
    return np.array(features_list), np.array(labels_list)

def data_split(dict_data, specific_split=False, test_files=None, split_ratio=0.8):

    if specific_split==True:
        if test_files is None:
            print("Warning: no test files given. Doing random split instead.")
            data_list = [data for key, data in dict_data.items()]
            train_data, test_data = train_test_split(data_list, split_ratio)
        else:
            test_data = [data for key, data in dict_data.items() if key in test_files]
            train_data = [data for key, data in dict_data.items() if key not in test_files]
    else: 
        data_list = [data for key, data in dict_data.items()]
        train_data, test_data = train_test_split(data_list, test_size=1-split_ratio)

    return train_data, test_data

def import_model(dir, name):
    """
    Load a trained model from dir.
    """
    os.makedirs(dir, exist_ok=True)
    model_path = os.path.join(dir, name)
    model = joblib.load(model_path)
    return model

def import_scaler(dir, name):
    """
    Load a fitted scaler from dir.
    """
    os.makedirs(dir, exist_ok=True)
    scaler_path = os.path.join(dir, name)
    scaler = joblib.load(scaler_path)
    return scaler

def save_model(model, dir, name):
    """
    Save a trained model to dir.
    """
    os.makedirs(dir, exist_ok=True)
    name = f"{name}.pkl"
    model_path = os.path.join(dir, name)
    joblib.dump(model, model_path)

def save_scaler(scaler, dir, name):
    """
    Save a fitted scaler to dir.
    """
    os.makedirs(dir, exist_ok=True)
    name = f"{name}.pkl"
    scaler_path = os.path.join(dir, name)
    joblib.dump(scaler, scaler_path)
    
def train_model(training_features, training_labels, fast_training=True, params=None):
    if fast_training:
        if params is None:
            optimized_param = {
                'max_depth': 10,
                'min_samples_leaf': 1,
                'min_samples_split': 10,
                'n_estimators': 100
            }
        model = RandomForestClassifier(**params)
        model.fit(training_features, training_labels)

    else:
        params = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        model = RandomForestClassifier()
        grid_search = GridSearchCV(estimator=model, param_grid=params)
        grid_search.fit(training_features, training_labels)

        model = grid_search.best_estimator_
        model.fit(training_features, training_labels)

    return model
    
def offline_decoding(test_data, model, features, scaler, window_size=200, step_size=100):
    '''
    test_data : DataFrame with columns ['Timestamps', 'Channel1', ..., 'ChannelN', 'Action1', 'Action2']
    model : trained classification model
    window_size : ms
    window_steo : ms

    show performances of the model on the test data
    '''
    fs, jitter = compute_sampling_freq(test_data['Timestamps'])

    window_size = int(window_size/1000 * fs)
    step_size = int(step_size/1000 * fs)

    testing_features, testing_labels = extract_features_and_labels(test_data, features, window_size=window_size, step_size=step_size)
    testing_features = scaler.transform(testing_features)

    y_pred = model.predict(testing_features)

    show_results(testing_labels, y_pred)


def measure_resting_state(channels, fs=1000, duration=5):
    """
    Online measure of the resting state for all channels.
    Returns a list of [mean, std] for each channel.
    """

    print(f"Measure of the resting state during {duration} seconds")
    input("Press any key to start the recording")

    n_samples = int(fs * duration)
    all_channels = [[] for _ in channels]

    for _ in range(n_samples):
        for idx, chan in enumerate(channels):
            all_channels[idx].append(chan.value)
        time.sleep(1 / fs)

    baseline_mean_std = []
    for channel in all_channels:
        arr = np.array(channel)
        baseline_mean_std.append([arr.mean(), arr.std()])

    print("Resting state measurement complete.")
    for i, (mean, std) in enumerate(baseline_mean_std):
        print(f"Channel {i+1}: mean={mean:.2f}, std={std:.2f}")

    return baseline_mean_std


def online_decoding(model, baseline_noise, scaler, channels, features, duration=30, window_size=200, step_size=100, fs=1000):
    """
    Real-time decoding loop.
    - model: trained classifier
    - scaler: fitted scaler
    - channels: list of AnalogIn objects
    - features: list of feature functions
    - window_size: in ms
    - step_size: in ms
    - fs: sampling frequency (Hz)
    """
    print("Starting online decoding. Press Ctrl+C to stop.")
    n_channels = len(channels)
    window_samples = int(window_size / 1000 * fs)
    step_samples = int(step_size / 1000 * fs)
    buffer = [[] for _ in range(n_channels)]

    predicted_labels = []

    def channel_scaler(x, mean, std):
        return (x-mean)/std

    start_time = time.time()

    try:
        # Pre-fill buffer
        for _ in range(window_samples):
            for i, chan in enumerate(channels):
                buffer[i].append(chan.value)
            time.sleep(1/fs)

        while True:
            # Check duration condition
            if (time.time() - start_time) > duration:
                print("Duration reached. Stopping online decoding.")
                break

            # Slide window by step_size
            for _ in range(step_samples):
                for i, chan in enumerate(channels):
                    buffer[i].append(channel_scaler(chan.value, baseline_noise[i, 0], baseline_noise[i, 1]))
                    buffer[i].pop(0)
                time.sleep(1/fs)

            # Prepare window data as DataFrame-like for feature extraction
            window_data = {f'Channel{i+1}': np.array(buffer[i]) for i in range(n_channels)}
            window_df = pd.DataFrame(window_data)

            # Extract features for this window
            feature_vector, _ = extract_window_features(window_df, features, window_df.columns, label_column=None)
            feature_vector = np.array(feature_vector).reshape(1, -1)
            feature_vector = scaler.transform(feature_vector)

            # Predict
            detected_mvmt = model.predict(feature_vector)[0]
            print(f"Detected movement: {detected_mvmt}")
            predicted_labels.append(detected_mvmt)

    except KeyboardInterrupt:
        print("Online decoding stopped.")

    return predicted_labels

