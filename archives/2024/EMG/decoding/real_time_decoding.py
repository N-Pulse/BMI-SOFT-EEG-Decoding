import board
import busio
import digitalio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
from adafruit_mcp3xxx.mcp3008 import MCP3008
from adafruit_mcp3xxx.analog_in import AnalogIn

from emg_functions import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the EMG decoding.")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the data directory where all data are stored.",
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
    # port = args.port

    # # I2C setup
    # i2c = busio.I2C(board.SCL, board.SDA)   # I2C interface initialisation

    # ads = ADS.ADS1115(i2c,data_rate=860)       
    # channels = [AnalogIn(ads, getattr(ADS, f'P{i}')) for i in range(nb_channels)]

    # SPI setup
    spi = busio.SPI(clock=board.SCK, MISO=board.MISO, MOSI=board.MOSI)
    cs = digitalio.DigitalInOut(board.D5)  # Chip select pin
    mcp = MCP3008(spi, cs)
    channels = [AnalogIn(mcp, i) for i in range(nb_channels)]

    # fs = 1000  # Hz
    # duration = 1  # seconds
    # samples = [[] for _ in range(nb_channels)]
    # start = time.time()
    # for _ in range(fs * duration):
    #     for i, chan in enumerate(channels):
    #         samples[i].append(chan.value)
    #     time.sleep(1/fs)

    
    data_dir = args.data_dir
    # data_path = os.path.join("data", "npulse", "raw")
    model_dir = args.model_dir
    scaler_dir = args.scaler_dir

    file_conditions=args.file_conditions
    # file_conditions="_5_"
    test_data_files = args.test_data_files
    # test_files = ["WS_R_5_250509100145.csv"]
    window_size = args.window_size
    # window_size=200
    step_size = args.step_size
    # step_size=100
    fs = args.sampling_freq
    # fs = 1000 

    features = [wl, wamp, var, rms]

    if args.online == True:
        # Online EMG decoding
        if (model_dir is not None) and (scaler_dir is not None):
            ext = args.model_file_ext
            model_files = [file for file in model_dir if f"model{ext}" in file]
            scaler_files =  [file for file in model_dir if f"scaler{ext}" in file]
            # Choose the first 
            model_name = model_files[0]
            scaler_name = scaler_files[0]
            # Import existing model and scaler
            scaler = import_scaler(model_dir, model_name)
            model = import_model(scaler_dir, scaler_name)
        else:
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
        # Offline EMG decoding
        dict_data = load_dataset(data_dir, file_conditions=file_conditions)

        train_data, test_data = data_split(dict_data, specific_split=True, test_files=test_data_files) 

        training_features, training_labels = extract_features_and_labels(train_data, features, window=window_size, step_size=step_size)
        testing_features, testing_labels = extract_features_and_labels(test_data, features, window=window_size, step_size=step_size)

        scaler = StandardScaler()
        training_features = scaler.fit_transform(training_features)
        testing_features = scaler.transform(testing_features)

        model = train_model(training_features, training_labels, fast_training=True)
        predicted_labels = model.fit(testing_features)

        # offline_decoding(
        #     test_data, 
        #     model, 
        #     features, 
        #     scaler, 
        #     window_size=window_size, 
        #     step_size=step_size)
        metrics = ["accuracy", "recall", "precision", "f1-score"]
        labels = [i.astype('str') for i in np.unique(testing_labels)]
        show_results(predicted_labels, true_labels=testing_labels, metrics=metrics, labels=labels)
