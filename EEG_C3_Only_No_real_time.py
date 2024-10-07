from os.path import exists
import serial
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, find_peaks, peak_prominences, correlate,coherence
import matplotlib.pyplot as plt
import time
import os
import datetime

# Duración de la prueba
start_time = time.time()

# Filtros de Butterworth
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos

# Global data
filtered_C3 = []

def extract_peak_features(channel, threshold_factor=1):
    global num_peaks, avg_height, avg_distance, avg_prominence, avg_width
    threshold = np.mean(channel) + threshold_factor * np.std(channel)
    peaks, _ = find_peaks(channel, height=threshold)
    num_peaks = len(peaks)
    avg_height = np.mean(channel[peaks]) if num_peaks > 0 else 0
    avg_distance = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
    avg_prominence = np.mean(peak_prominences(channel, peaks)[0]) if num_peaks > 0 else 0
    avg_width = np.mean(np.diff(peaks)) if num_peaks > 1 else 0
    return num_peaks, avg_height, avg_distance, avg_prominence, avg_width

# Lectura de datos de Arduino
def arduino_uno():
    global filtered_C3
    beta_wave_C3 = []
    try:
        ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
        time.sleep(2)
        fs = 250  # Frecuencia de muestreo
        lowcut = 12.0
        highcut = 30.0
        sos = butter_bandpass(lowcut, highcut, fs)
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    # Leer y convertir datos
                    c3_value = float(line)  # Solo C3
                    beta_wave_C3.append(c3_value)
                    # Filtrar las señales en tiempo real
                    if len(beta_wave_C3) > 1:
                        filtered_C3 = sosfilt(sos, np.array(beta_wave_C3))
                        # Detectar picos en tiempo real
                        features_C3 = extract_peak_features(filtered_C3)
                        print(f"Características C3: {features_C3}")
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        if elapsed_time > 0:  # Evaluación de cómo responden las neuronas a diferentes estímulos
                            firing_rate_C3 = features_C3[0] / elapsed_time  # Tasa de disparo en C3
                            print(f"Tasa de disparo C3: {firing_rate_C3:.2f} Hz")

                except ValueError:
                    print(f"Error en los datos recibidos: {line}")
                    continue

            # Detener después de capturar 1000 muestras o break con interrupción
            if len(beta_wave_C3) >= 1000:
                break
    except KeyboardInterrupt:
        print("Interrupción")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        ser.close()
        end_time = time.time()
        trial_duration = end_time - start_time
        print(f"La prueba duró: {trial_duration:.2f} segundos")
        return (beta_wave_C3, filtered_C3, trial_duration)

def plot_signals(data_for_plot, ch_names):
    plt.figure(figsize=(12, 6))
    for i, channel_data in enumerate(data_for_plot):
        plt.plot(channel_data + i * 5, label=ch_names[i])
    plt.title("Señales EEG Filtradas")
    plt.xlabel("Tiempo (muestras)")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid()
    plt.show()

def picos_save(f_features, folder, subject_ID):
    path = os.path.join(folder, f"/{subject_ID}" + '_picos')
    f_features.to_csv(path, index=False)
    print(f"Características de picos guardadas en {path}")

def save_filtered_data(filtered_signals):
    subject_input = input('Please enter the subject ID and the number of repetition: ').split(' ')
    subject_ID = subject_input[0]
    subject_ID = '0' + subject_ID if int(subject_ID) < 10 else subject_ID
    folder = f'S{subject_ID}_{datetime.datetime.now().strftime("%d%m%Y_%H%M")}'
    os.makedirs(folder, exist_ok=True)
    path_1 = '/home/r3i_00/Documents/EEG_Proyect/' + folder + '/' + 'filtered_data.csv'
    data_filtered = pd.DataFrame({
        'Beta_wave_C3': filtered_signals[0]
    }, columns=['Beta_wave_C3'])
    try:
        data_filtered.to_csv(path_1, index=False)
        print("Datos filtrados guardados.")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        return folder, subject_ID

beta_wave_C3, filtered_C3, trial_duration = arduino_uno()
filtered_signals = [np.array(filtered_C3)]
folder, subject_ID = save_filtered_data(filtered_signals)

if beta_wave_C3 is not None:
    ch_names = ["C3"]
    data_for_plot = [filtered_C3]
    plot_signals(data_for_plot, ch_names)

    features = [extract_peak_features(filtered_C3)]
    f_features = pd.DataFrame(
        features,
        index=["C3"],
        columns=["num_peaks", "avg_height", "avg_distance", "avg_prominence", "avg_width"]
    )
    picos_save(f_features, folder, subject_ID)
    print(f_features)

def fano_calcular(spike_counts):
    if len(spike_counts) > 0:
        mean_spike_count = np.mean(spike_counts)
        var_spike_count = np.var(spike_counts)
        fano_factor = var_spike_count / mean_spike_count if mean_spike_count > 0 else 0
        print(f"Fano Factor: {fano_factor}")
    else:
        print("No hay picos")

# Determina la variabilidad en la tasa de disparo de una neurona
fano_calcular([])  # Cambié spike_counts_C3 a una lista vacía, ya que no se usa en este contexto.

def autocorrelation(data):
    n = len(data)
    acorr = correlate(data - np.mean(data), data - np.mean(data), mode='full',method='auto')
    lags = np.arange(-n + 1, n)  # intervalos de tiempo específicos
    acorr /= np.max(acorr)  # normalización
    return lags, acorr

lags_C3, corr_C3 = autocorrelation(filtered_C3)

def auto_corr(lags_C3, corr_C3):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(lags_C3, corr_C3)
    plt.title('Autocorrelación C3')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelación')
    plt.grid()
    plt.show()

auto_corr(lags_C3, corr_C3)

#Hay cohrencia entre la senales?

