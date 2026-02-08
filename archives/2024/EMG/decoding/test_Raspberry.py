"""
Installation:
sudo apt update
sudo apt install python3-pip git
pip3 install RPi.GPIO spidev

Clone Waveshare Driver:
git clone https://github.com/waveshare/High-Precision-AD-HAT
Files to use : ADS1256.py and config.py

Configuration:
- make sure SPI enabled on Pi :
sudo raspi-config → Interfacing Options → SPI → Enable

"""

import ADS1256
import RPi.GPIO as GPIO
import time

def setup_ads1256():
    if not ADS1256.ADS1256_init():
        print("ADS1256 init failed.")
        exit()
    ADS1256.ADS1256_SetMode(1)  # 1 = single-ended mode

def read_channels(channels):
    readings = {}
    for ch in channels:
        val = ADS1256.ADS1256_GetChannalValue(ch)
        voltage = (val * 5.0) / 0x7fffff  # convert to volts (±5V full scale)
        readings[ch] = round(voltage, 4)
    return readings

def main():
    try:
        setup_ads1256()
        print("Reading EMG channels (CH0, CH1, CH2):")
        while True:
            voltages = read_channels([0, 1, 2])
            print(f"CH0: {voltages[0]} V | CH1: {voltages[1]} V | CH2: {voltages[2]} V")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ADS1256.ADS1256_Reset()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
