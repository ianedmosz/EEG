import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences, correlate, correlation_lags
import matplotlib.pyplot as plt
import os
import datetime

def butter_bandpass(lowcut, highcut, fs, order=8):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def leer_datos_csv(ruta_csv):
    data = pd.read_csv(ruta_csv)
    return data['T7'], data['F8'], data['Cz'], data['P4']


def filtrar_senal(senal, fs, lowcut=1, highcut=40.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, senal)  # Aplica el filtro Butterworth

def extract_peak_features(channel, fs, threshold_factor=1):
    threshold = np.mean(channel) + threshold_factor * np.std(channel)
    peaks, _ = find_peaks(channel, height=threshold)
    num_peaks = len(peaks)
    avg_height = np.mean(channel[peaks]) if num_peaks > 0 else 0
    avg_distance = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
    avg_prominence = np.mean(peak_prominences(channel, peaks)[0]) if num_peaks > 0 else 0
    avg_width = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
    peak_times = peaks / fs  # Convertir los índices de picos en tiempos
    return num_peaks, avg_height, avg_distance, avg_prominence, avg_width, peak_times


def calcular_firing_rate(peak_times, total_duration):
    return len(peak_times) / total_duration


def factor_fano(data):
    variance = np.var(data)
    mean = np.mean(data)
    print(f'Media: {mean}, Varianza: {variance}')
    if mean > 0:
        return variance / mean
    else:
        return np.nan
    return variance / mean if mean != 0 else 0


def plot_signals(data_for_plot, ch_names):
    plt.figure(figsize=(12, 6))

    max_amplitudes = [np.max(np.abs(ch_data)) for ch_data in data_for_plot]
    desplazamiento = np.max(max_amplitudes) * 1.2

    for i, channel_data in enumerate(data_for_plot):
        plt.plot(channel_data + i * desplazamiento, label=ch_names[i])

    plt.title("Señales EEG Filtradas")
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid()
    plt.show()


def plot_peaks(filtered_data, ch_names):
    for channel_index, channel_name in enumerate(ch_names):
        selected_channel = filtered_data[channel_index]
        threshold = np.mean(selected_channel) + np.std(selected_channel)
        peaks, _ = find_peaks(selected_channel, height=threshold)

        plt.figure(figsize=(12, 6))
        plt.plot(selected_channel, label=f"{channel_name} - Señal Filtrada", color="yellow")
        plt.plot(peaks, selected_channel[peaks], "x", label="Picos", color="red")
        plt.axhline(y=threshold, color="purple", linestyle="--", label=f"Umbral = {threshold:.2f}")
        plt.title(f"Picos en la Señal de {channel_name}")
        plt.xlabel("Tiempo (muestras)")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.grid()
        plt.show()


def save_filtered_data(filtered_signals, ch_names):
    subject_input = input('Por favor, ingresa el ID del sujeto ').split(' ')
    subject_ID = subject_input[0]
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    folder = f'S{subject_ID}_{datetime.datetime.now().strftime("%d%m%Y_%H%M")}'
    os.makedirs(folder, exist_ok=True)
    path_1 = os.path.join(folder, 'filtered_data.csv')

    data_filtered = pd.DataFrame({ch: filtered_signals[i] for i, ch in enumerate(ch_names)})

    try:
        data_filtered.to_csv(path_1, index=False)
        print("Datos filtrados guardados.")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        return folder, subject_ID


def autocorrelation(data):
    acorr = correlate(data - np.mean(data), data - np.mean(data), mode='full', method='auto')
    lags = correlation_lags(len(data), len(data))
    acorr /= np.max(acorr)
    return lags, acorr

def plot_autocorrelation(lags, corr, ch_name, lag_limit=None):
    plt.figure(figsize=(12, 6))
    plt.plot(lags, corr)
    plt.title(f'Autocorrelación {ch_name}')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelación')
    if lag_limit:
        plt.xlim(-lag_limit, lag_limit)
    plt.grid()
    plt.show()

def procesar_eeg(ruta_csv):
    T7, F8, Cz, P4 = leer_datos_csv(ruta_csv)
    fs = 200  # Frecuencia de muestreo
    T7_filtrado = filtrar_senal(T7, fs)
    F8_filtrado = filtrar_senal(F8, fs)
    Cz_filtrado = filtrar_senal(Cz, fs)
    P4_filtrado = filtrar_senal(P4, fs)

    filtered_signals = [T7_filtrado, F8_filtrado, Cz_filtrado, P4_filtrado]
    ch_names = ['T7', 'F8', 'Cz', 'P4']

    folder, subject_ID = save_filtered_data(filtered_signals, ch_names)

    plot_signals(filtered_signals, ch_names)
    plot_peaks(filtered_signals, ch_names)

    total_duration = len(T7_filtrado) / fs

    for i, ch_name in enumerate(ch_names):
        num_peaks, avg_height, avg_distance, avg_prominence, avg_width, peak_times = extract_peak_features(
            filtered_signals[i], fs)

        firing_rate = calcular_firing_rate(peak_times, total_duration)
        print(f"Tasa de Disparo ({ch_name}): {firing_rate:.2f} Hz")

        lag_limit = len(filtered_signals[i]) // 2
        lags, corr = autocorrelation(filtered_signals[i])
        plot_autocorrelation(lags, corr, ch_name, lag_limit=100)

    features = [extract_peak_features(channel, fs)[:5] for channel in filtered_signals]
    f_features = pd.DataFrame(
        features,
        index=ch_names,
        columns=["num_peaks", "avg_height", "avg_distance", "avg_prominence", "avg_width"]
    )
    print(f_features)

    fano_factors = {ch_name: factor_fano(filtered_signals[i]) for i, ch_name in enumerate(ch_names)}
    print("Factores de Fano:", fano_factors)


ruta_csv = '/home/r3i_00/Downloads/s01_ex01_s02.csv'
procesar_eeg(ruta_csv)
