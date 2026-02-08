import argparse
from pathlib import Path
import time
import numpy as np
from sklearn.preprocessing import StandardScaler

# These are defined in your emg_functions.py
from emg_functions import (
    load_dataset,
    extract_features_and_labels,
    train_model,
    import_scaler,
    import_model,
    save_model,
    save_scaler,
    measure_resting_state,
    online_decoding,
    show_results,
    data_split,
    #wl, wamp, var, rms # these are our feature functions
    zc, ssc, wl, log_det, wamp, var, mav, rms
)

# Import your ADS1263 driver and RPi.GPIO
import ADS1263
import RPi.GPIO as GPIO

# Define paths (adjust as needed if not using relative paths or constants)
# You might have these defined elsewhere, but for completeness:
models = Path("models") # Example path
results = Path("results") # Example path


def setup_ads1263_emg():
    """Initializes the ADS1263 ADC for EMG signal reading."""
    adc = ADS1263.ADS1263()
    if adc.ADS1263_init_ADC1() != 0:
        print("ADS1263 init failed.")
        exit()
    # Set to single-ended mode for individual electrode readings
    adc.ADS1263_SetMode(0)
    return adc

# Define a wrapper for reading channels that mimics the AnalogIn object structure
class ADS1263Channel:
    def __init__(self, adc_instance, channel_num):
        self.adc = adc_instance
        self.channel_num = channel_num

    @property
    def value(self):
        """Reads the raw ADC value and converts it to a voltage."""
        # This will return the raw value, which online_decoding will then process
        val = self.adc.ADS1263_GetChannalValue(self.channel_num)
        # Convert raw value to a voltage or a suitable scaled value if necessary
        # The original ADS1263 example converts to voltage using (val * 5.0) / 0x7fffff
        # You need to decide if you want raw ADC values or converted voltages here.
        # For consistency with the `emg_functions.py` which likely expects raw or mV,
        # it might be better to return a raw value that your scaling handles.
        # However, for demonstration, let's keep it in volts as per your original example.
        voltage = (val * 5.0) / 0x7fffff
        return voltage # Return voltage or raw value depending on your feature extraction's expectation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EMG decoding.")
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to the data directory where all data are stored.",
        default=None
    )
    parser.add_argument(
        "--nb_channels",
        type=int,
        help="Number of channels used.",
        default=3,
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        help="Path to the model directory where the model is stored.",
        default=models
    )
    parser.add_argument(
        "--scaler_dir", # Added scaler_dir as it's used
        type=Path,
        help="Path to the scaler directory where the scaler is stored.",
        default=models # Often scaler is in the same directory as the model
    )
    parser.add_argument(
        "--result_dir",
        type=Path,
        help="Path to the result directory where all plots will be stored.",
        default=results
    )
    parser.add_argument(
        "--test_data_files",
        type=str,
        help="Names of the testing files in data_dir.",
    )
    parser.add_argument(
        "--model_file_ext",
        type=str,
        help="Extension of the model/scaler file name in model_dir.",
        default=""
    )
    parser.add_argument(
        "--online",
        type=bool,
        help="True if online, False if offline.",
        default=False
    )
    parser.add_argument(
        "--file_conditions",
        type=list,
        help="Conditions to filter the training files",
        default=None
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Size of the window.",
        default=200
    )
    parser.add_argument(
        "--step_size",
        type=int,
        help="Size of the step.",
        default=100
    )
    parser.add_argument(
        "--sampling_freq",
        type=int,
        help="Sampling frequency (per channel).",
        default=1000
    )
    parser.add_argument(
        "--port",
        type=str,
        help="Port connected in serial.",
        default='COM7'
    )
    parser.add_argument(
        "--online_decoding_duration",
        type=int,
        help="Duration of the online decoding.",
        default=30
    )

    args = parser.parse_args()
    nb_channels = args.nb_channels

    # --- ADS1263 Setup ---
    adc_instance = None # Initialize to None
    try:
        adc_instance = setup_ads1263_emg()
        # Create a list of channel objects that wrap the ADS1263_GetChannalValue method
        channels = [ADS1263Channel(adc_instance, i) for i in range(nb_channels)]

        #if data_dir is not None:
              #data_dir = args.data_dir
        model_dir = args.model_dir
        scaler_dir = args.scaler_dir

        file_conditions = args.file_conditions
        test_data_files = args.test_data_files
        window_size = args.window_size
        step_size = args.step_size
        fs = args.sampling_freq

        features = [wl, wamp, var, rms]

        if args.online == True:
            print("ONLINE")
            # Online EMG decoding
            if (model_dir is not None) and (scaler_dir is not None):
                print("IF")
                ext = args.model_file_ext
                model_files = [file for file in model_dir.iterdir() if f"model.{ext}" in file.name]
                scaler_files = [file for file in model_dir.iterdir() if f"scaler.{ext}" in file.name]

                if not model_files or not scaler_files:
                     print(f"Error: Model or scaler files not found in {model_dir}. Please train a model or check paths.")
                     exit()

                model_name = model_files[0]
                scaler_name = scaler_files[0]
                scaler_name='scaler.pkl'
                model_name='RF_model.pkl'

                # Import existing model and scaler
                scaler = import_scaler(scaler_dir,scaler_name) # Corrected argument order if needed by your function
                print("SCALER")
                model = import_model(model_dir, model_name) # Corrected argument order if needed
                print("MODEL")
            else:
                print("ELSE")
                # Train model
                dict_data = load_dataset(data_dir, file_conditions=file_conditions)
                train_data = [data for key, data in dict_data.items()]
                training_features, training_labels = extract_features_and_labels(train_data, features, window=window_size, step_size=step_size)

                scaler = StandardScaler()
                training_features = scaler.fit_transform(training_features)

                model = train_model(training_features, training_labels, fast_training=True)

                nb_dof = len(np.unique(training_labels))
                nb_feat = len(features)
                ext = f"ch{nb_channels}_dof{nb_dof}_window{window_size}_step{step_size}_feat{nb_feat}"
                model_name = f"model_{ext}"
                scaler_name = f"scaler_{ext}"
                save_model(model, model_dir, model_name)
                save_scaler(scaler, model_dir, scaler_name)

            channels_scaler = measure_resting_state(channels, duration=5)

            predicted_labels = online_decoding(
                model,
                channels_scaler,
                scaler,
                channels,
                features,
                duration=args.online_decoding_duration,
                window_size=window_size,
                step_size=window_size,
                fs=1000
            )
            show_results(predicted_labels)

        else:
            print("OFFLINE")
            # Offline EMG decoding
            dict_data = load_dataset(data_dir, file_conditions=file_conditions)

            train_data, test_data = data_split(dict_data, specific_split=True, test_files=test_data_files)

            training_features, training_labels = extract_features_and_labels(train_data, features, window=window_size, step_size=step_size)
            testing_features, testing_labels = extract_features_and_labels(test_data, features, window=window_size, step_size=step_size)

            scaler = StandardScaler()
            training_features = scaler.fit_transform(training_features)
            testing_features = scaler.transform(testing_features)

            model = train_model(training_features, training_labels, fast_training=True)
            predicted_labels = model.predict(testing_features) # Corrected from .fit to .predict for unseen data

            metrics = ["accuracy", "recall", "precision", "f1-score"]
            labels = [str(i) for i in np.unique(testing_labels)] # Ensure labels are strings for show_results
            show_results(predicted_labels, true_labels=testing_labels, metrics=metrics, labels=labels)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if adc_instance: # Only exit if ADC was successfully initialized
            adc_instance.ADS1263_Exit()
        GPIO.cleanup()
