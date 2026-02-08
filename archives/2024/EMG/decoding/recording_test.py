import os
import time
import csv
# import serial # No longer needed for Raspberry Pi ADC
import pandas as pd
import numpy as np
import keyboard
from datetime import datetime

# --- Raspberry Pi / ADS1263 Imports and Setup ---
import ADS1263
import RPi.GPIO as GPIO # For cleanup

# Define a wrapper for reading channels that mimics the Arduino's analogRead output
# and provides a 'value' property like adafruit_ads1x15 or adafruit_mcp3xxx
class ADS1263Channel:
    def __init__(self, adc_instance, channel_num):
        self.adc = adc_instance
        self.channel_num = channel_num

    @property
    def value(self):
        """Reads the raw ADC value from the ADS1263."""
        # The Arduino analogRead returns values from 0-1023.
        # The ADS1263 returns up to 0x7fffff (8,388,607) for 24-bit.
        # For recording raw data, we can just return this large raw value.
        # Your previous script used float conversion: channels.append(float(data.split(" ")[j]))
        # Returning a raw float or integer here is appropriate.
        # If your subsequent processing (features, etc.) expects a voltage, you'll need
        # to apply the voltage conversion here or adjust those functions.
        # For simplicity and direct replacement of 'analogRead', let's return raw.
        # If you prefer voltage here, use: return (self.adc.ADS1263_GetChannalValue(self.channel_num) * 5.0) / 0x7fffff
        # return float(self.adc.ADS1263_GetChannalValue(self.channel_num))
        return (self.adc.ADS1263_GetChannalValue(self.channel_num) * 5.0) / 0x7fffff

def setup_ads1263_recording():
    """Initializes the ADS1263 ADC for recording."""
    adc = ADS1263.ADS1263()
    if adc.ADS1263_init_ADC1() != 0:
        print("ADS1263 init failed. Exiting.")
        GPIO.cleanup()
        exit() # Exiting gracefully if ADC fails
    adc.ADS1263_SetMode(0) # 0 = single-ended mode, 1 = differential mode
    return adc

# --- Original Functions (modified slightly for context) ---
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def countdown(n=3):
    for i in range(n, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1)

def record_data(adc_instance, channels_list, sequence, file_path, nb_channels, duration=3, transition_time=3):
    """
    Records EMG data from ADS1263 on Raspberry Pi to a CSV file.

    Args:
        adc_instance: The initialized ADS1263 object.
        channels_list: A list of ADS1263Channel objects for active channels.
        sequence (list): List of actions/gestures to record.
        file_path (str): Full path to the CSV file to save data.
        nb_channels (int): Number of EMG channels being recorded.
        duration (int): Duration in seconds for each action.
        transition_time (int): Time allowed for transition (though spacebar driven).
    """
    try:
        print(f"You will keep each position during {duration} seconds.")
        print(f"You will have to press the space while you change position.")
        print(f"It is enough time, don't rush...")
        time.sleep(1)
        input(f"Press ENTER to start sequence...")
        countdown()

        with open(file_path, mode='w', newline='') as file:
            header = ["Timestamp"]
            for i in range(nb_channels):
                header.append(f"Channel{i+1}")
            header.append("Action1") # Current action
            header.append("Action2") # Next action (during transition)
            writer = csv.writer(file)
            writer.writerow(header)

            print("Starting data recording...")
            # No need for ser.readline() stabilization with ADS1263 directly
            # time.sleep(1) # Small delay to ensure everything is ready
            start_sequence_time = time.time()

            for i in range(len(sequence) - 1): # Iterate through actions, excluding the last one (it's the end state)
                print(f"Recording '{sequence[i]}'")
                start_action_time = time.time()
                prompted = False
                transition_started = False

                while True:
                    current_time_in_sequence = round(time.time() - start_sequence_time, 3)
                    
                    # Read all channels using the ADS1263Channel objects
                    current_channel_values = [chan.value for chan in channels_list]

                    buffer_row = [current_time_in_sequence]
                    buffer_row.extend(current_channel_values)
                    buffer_row.append(sequence[i]) # Current action

                    # Prompt for transition after a certain time, or when space is pressed
                    # Adjusted prompt logic for clarity
                    if not prompted and (time.time() - start_action_time) > (duration - 1): # Prompt 1 sec before duration ends
                        print(f"Prepare to press and hold SPACE to transition to '{sequence[i+1]}'...")
                        prompted = True
                    
                    # Read spacebar state
                    if keyboard.is_pressed('space'):
                        transition_started = True
                        buffer_row.append(sequence[i+1]) # Indicate transition to next action
                    else:
                        buffer_row.append("") # No transition action if space not pressed
                        if transition_started: # If space was *just released*
                            print(f"Transition complete. Starting '{sequence[i+1]}'\n")
                            break # Exit this action's recording loop

                    writer.writerow(buffer_row)
                    # Introduce a small delay to control sampling rate
                    # If you target ~1000Hz (1ms delay) like Arduino, use time.sleep(0.001)
                    # The ADS1263 can read much faster; adjust based on your desired sample rate
                    time.sleep(0.001) # Approx 1000 Hz, adjust as needed. ADS1263 can go up to 38400 Sps

            # Record the very last action in the sequence
            print(f"Recording final action: '{sequence[-1]}'")
            start_final_action_time = time.time()
            while (time.time() - start_final_action_time) < duration:
                current_time_in_sequence = round(time.time() - start_sequence_time, 3)
                current_channel_values = [chan.value for chan in channels_list]
                buffer_row = [current_time_in_sequence]
                buffer_row.extend(current_channel_values)
                buffer_row.append(sequence[-1]) # Last action
                buffer_row.append("") # No next action
                writer.writerow(buffer_row)
                time.sleep(0.001) # Match the sampling rate

        print(f"Data recorded successfully to '{file_path}'")

    except KeyboardInterrupt:
        print("\nRecording interrupted by user.")
    except Exception as e: # Catch other potential errors
        print(f"An error occurred during recording: {e}")
    finally:
        # Cleanup ADS1263 and GPIO
        if adc_instance:
            adc_instance.ADS1263_Exit()
        GPIO.cleanup()
        print("ADS1263 and GPIO cleaned up.")


# --- Main execution block (similar to your original Python script) ---

# Load Excel file with sequences
file_path_excel = "EMG_recording_protocol.xlsx" # Renamed to avoid conflict with file_path variable
try:
    xls = pd.ExcelFile(file_path_excel)
    df_sequences = pd.read_excel(xls, sheet_name="Sequences")
    nb_of_sequences = np.shape(df_sequences)[1]

    # Extract sequences
    sequences = {}
    for seq in range(nb_of_sequences):
        sequences[seq]=df_sequences.iloc[:,seq].dropna().tolist()
except FileNotFoundError:
    print(f"Error: Protocol file '{file_path_excel}' not found. Please ensure it's in the script directory.")
    exit()

# Nb of channels used
nb_channels = 0
while nb_channels not in range(1, 10):
    try:
        nb_channels = int(input("Enter number of channels (1-9): "))
    except ValueError:
        print("Invalid input. Please enter a number.")

# User inputs
participant = input("Enter participant initials: ")

hand = 0
while hand not in ['L', 'R']:
    hand = input("Enter which hand (L or R): ").upper()

# Create necessary folders
base_folder = os.path.join(os.getcwd(), "data")
create_folder(base_folder)

# Choose sequence
print("Select sequence to run:")
# Dynamically print sequence options based on what's in your Excel
for i in range(nb_of_sequences):
    print(f"{i+1} - {sequences[i][0]} (first action example)") # Show first action as a hint

sequence_choice = 0
while sequence_choice not in range(1,nb_of_sequences+1):
    try:
        sequence_choice = int(input(f"Enter sequence number (1-{nb_of_sequences}): "))
    except ValueError:
        print("Invalid input. Please enter a number.")

while(True):
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")
    filename = f"{participant}_{hand}_ch{nb_channels}_seq{sequence_choice}_{timestamp}.csv" # Changed sequence_choice to seq{X}
    file_path_output = os.path.join(base_folder, filename) # Renamed to avoid conflict

    # --- Initialize ADS1263 and start recording ---
    adc = None
    try:
        adc = setup_ads1263_recording()
        # Create ADS1263Channel objects for the specified number of channels
        # Assuming channels are 0 to nb_channels-1
        channels_for_recording = [ADS1263Channel(adc, i) for i in range(nb_channels)]

        record_data(
            adc,
            channels_for_recording,
            sequences[sequence_choice-1],
            file_path_output,
            nb_channels
        )
    except KeyboardInterrupt:
        print("\nRecording process terminated by user.")
    except Exception as e:
        print(f"An error occurred during setup or recording: {e}")
    finally:
        print("Problem with ADS")
