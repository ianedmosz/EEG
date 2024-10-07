from os.path import exists
import serial
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, correlate
import matplotlib.pyplot as plt
import time
import os
import datetime

# Duración de la prueba
start_time = time.time()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

def apply_filter(data, lowcut=12.0, highcut=30.0, fs=250, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def extract_peak_features(channel, threshold_factor=1):
    threshold = np.mean(channel) + threshold_factor * np.std(channel)
    peaks, _ = find_peaks(channel, height=threshold)
    num_peaks = len(peaks)
    avg_height = np.mean(channel[peaks]) if num_peaks > 0 else 0
    avg_distance = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
    avg_prominence = np.mean(peak_prominences(channel, peaks)[0]) if num_peaks > 0 else 0
    avg_width = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
    return num_peaks, avg_height, avg_distance, avg_prominence, avg_width


def arduino_uno():
    try:
        ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
        time.sleep(2)
        fs = 250
        data_diff = []

        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    c3_value, c4_value = map(float, line.split())
                    diff_value = c3_value - c4_value
                    data_diff.append(diff_value)

                    if len(data_diff) >= 1000:
                        break
                except ValueError:
                    print(f"Error en los datos recibidos: {line}")
                    continue

    except KeyboardInterrupt:
        print("Interrupción")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        ser.close()
        end_time = time.time()
        trial_duration = end_time - start_time
        print(f"La prueba duró: {trial_duration:.2f} segundos")
        return np.array(data_diff), trial_duration

def save_filtered_data(filtered_signals):
    subject_input = input(
        'Por favor ingrese el ID del sujeto y el número de repetición (separados por espacio): ').split(' ')
    subject_ID = subject_input[0]
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    folder = f'S{subject_ID}_{datetime.datetime.now().strftime("%d%m%Y_%H%M")}'
    os.makedirs(folder, exist_ok=True)
    path = f'/home/r3i_00/Documents/EEG_Proyect/{folder}/filtered_data.csv'

    data_filtered = pd.DataFrame({
        'Diferencial_C3_C4': filtered_signals
    })
    try:
        data_filtered.to_csv(path, index=False)
        print(f"Datos filtrados guardados en: {path}")
    except Exception as e:
        print(f'Error: {e}')
    return folder, subject_ID



def plot_peaks(filtered_data):
    threshold = np.mean(filtered_data) + np.std(filtered_data)
    peaks, _ = find_peaks(filtered_data, height=threshold)

    plt.figure(figsize=(12, 6))
    plt.plot(filtered_data, label="Señal Filtrada - Diferencial C3-C4", color="yellow")
    plt.plot(peaks, filtered_data[peaks], "x", label="Picos", color="red")
    plt.axhline(y=threshold, color="purple", linestyle="--", label=f"Umbral = {threshold:.2f}")
    plt.title("Picos en la Señal Filtrada (C3-C4)")
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid()
    plt.show()


def fano_calcular(spike_counts):
    if len(spike_counts) > 0:
        mean_spike_count = np.mean(spike_counts)
        var_spike_count = np.var(spike_counts)
        fano_factor = var_spike_count / mean_spike_count if mean_spike_count > 0 else 0
        print(f"Fano Factor: {fano_factor}")
    else:
        print("No hay picos")


data_diff, trial_duration = arduino_uno()
filtered_diff = apply_filter(data_diff)

folder, subject_ID = save_filtered_data(filtered_diff)
plot_peaks(filtered_diff)

features = extract_peak_features(filtered_diff)
print(f"Valores de los picos: {features}")

fano_calcular(features)
