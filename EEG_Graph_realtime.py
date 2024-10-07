from os.path import exists
import serial
import numpy as np
import pandas as pd
from fontTools.ttLib.ttGlyphSet import LerpGlyph
from scipy.signal import butter, sosfilt, find_peaks, peak_prominences,correlate
import matplotlib.pyplot as plt
import time
import os
import datetime
from matplotlib.animation import FuncAnimation
import time

#duracion de la prueba
start_time = time.time()
# Filtros de Butterworth
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs #frecuencia de nyq
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sos
#plot
fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_title("C3")
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_title("C4")
#Global data
filtered_C3 = []
filtered_C4 = []

def animation(frame):
    ax1.clear()
    ax2.clear()
    ax1.set_ylim(25, 33.5)
    ax2.set_ylim(-150, 150)
    ax1.plot(filtered_C3, color="orange")
    ax2.plot(filtered_C4, color="blue")

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
    global filtered_C3, filtered_C4
    spike_counts_C3 = []
    spike_counts_C4 = []
    try:
        ser = serial.Serial('/dev/ttyACM1', 9600, timeout=1)
        time.sleep(2)
        fs = 250  # Frecuencia de muestreo
        lowcut = 12.0
        highcut = 30.0
        sos = butter_bandpass(lowcut, highcut, fs)
        beta_wave_C3 = []
        beta_wave_C4 = []
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                try:
                    # Leer y convertir datos
                    c3_value, c4_value = map(float, line.split())  # Float y con separador
                    beta_wave_C3.append(c3_value)
                    beta_wave_C4.append(c4_value)

                    # Filtrar las señales en tiempo real
                    if len(beta_wave_C3) > 1:
                        filtered_C3 = sosfilt(sos, np.array(beta_wave_C3))
                        filtered_C4 = sosfilt(sos, np.array(beta_wave_C4))
                        # Detectar picos en tiempo real
                        features_C3 = extract_peak_features(filtered_C3)
                        features_C4 = extract_peak_features(filtered_C4)
                        spike_counts_C3.append(features_C3[0]) #Conteo de spikes
                        spike_counts_C4.append(features_C4[0]) #conteo de spikes
                        print(f"Características C3: {features_C3}")
                        print(f"Características C4: {features_C4}")
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        if elapsed_time > 0:#Evaluacion de cómo responden las neuronas a diferentes estímulos o condiciones
                            firing_rate_C3 = features_C3[0] / elapsed_time  # Tasa de disparo en C3
                            firing_rate_C4 = features_C4[0] / elapsed_time  # Tasa de disparo en C4
                            print(
                                f"Tasa de disparo C3: {firing_rate_C3:.2f} Hz, Tasa de disparo C4: {firing_rate_C4:.2f} Hz")

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
        return (beta_wave_C3, beta_wave_C4, filtered_C3, filtered_C4,trial_duration)

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

def plot_peaks(filtered_data, ch_names):
    for channel_index, channel_name in enumerate(ch_names):
        selected_channel = filtered_data[channel_index]
        threshold = np.mean(selected_channel) + np.std(selected_channel)
        peaks, _ = find_peaks(selected_channel, height=threshold)
        plt.figure(figsize=(12, 6))
        plt.plot(selected_channel, label=f"{channel_name} - Señal Filtrada", color="orange")
        plt.plot(peaks, selected_channel[peaks], "x", label="Picos Detectados")
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"Umbral = {threshold:.2f}")
        plt.title(f"Picos en la Señal de {channel_name}")
        plt.xlabel("Tiempo (muestras)")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.grid()
        plt.show()

def picos_save(f_features, folder,subject_ID):
    path = os.path.join(folder, f"/{subject_ID}"+'_picos')
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
        'Beta_wave_C3': filtered_signals[0],
        'Beta_wave_C4': filtered_signals[1]
    }, columns=['Beta_wave_C3', 'Beta_wave_C4'])
    try:
        data_filtered.to_csv(path_1, index=False)
        print("Datos filtrados guardados.")
    except Exception as e:
        print(f'Error: {e}')
    finally:
        return folder, subject_ID

anim = FuncAnimation(fig, animation, frames=None, interval=1000)
plt.show()
beta_wave_C3, beta_wave_C4, filtered_C3, filtered_C4, spike_counts_C3, spike_counts_C4, trial_duration = arduino_uno()
filtered_signals = [np.array(filtered_C3), np.array(filtered_C4)]
folder,subject_ID=save_filtered_data(filtered_signals)

if beta_wave_C3 is not None and beta_wave_C4 is not None:
    ch_names = ["C3", "C4"]
    data_for_plot = [filtered_C3, filtered_C4]
    plot_signals(data_for_plot, ch_names)
    filtered_signals = [np.array(filtered_C3), np.array(filtered_C4)]
    plot_peaks(filtered_signals, ch_names)
    features = [extract_peak_features(channel) for channel in filtered_signals]
    f_features = pd.DataFrame(
        features,
        index=["C3", "C4"],
        columns=["num_peaks", "avg_height", "avg_distance", "avg_prominence", "avg_width"]
    )
    picos_save(f_features,folder,subject_ID)
    print(f_features)


def fano_calcular(spike_counts):
    if len(spike_counts) > 0:
        mean_spike_count = np.mean(spike_counts)
        var_spike_count = np.var(spike_counts)
        fano_factor = var_spike_count / mean_spike_count if mean_spike_count > 0 else 0
        print(f"Fano Factor: {fano_factor}")
    else:
        print("No hay picos")

#determina la variabilidad en la tasa de disparo de una neurona
#Un Fano Factor mayor que 1 indica mayor variabilidad que la esperada
fano_calcular(spike_counts_C3)
fano_calcular(spike_counts_C4)

def autocorrelation(data):
    n = len(data)
    acorr = correlate(data - np.mean(data), data - np.mean(data), mode='full',method='auto')# sacamos que tanta difrencia
    # hay con el promedio de la senal dicienos que tanta vriacion hay
    lags = correlation_lags(len(signal), len(signal)) #intervalo de tiempo específico para ver como varia a lo largo del tiempo
    corr /= np.max(acorr) #normalizacion
    return lags, corr

lags_C3, corr_C3 = autocorrelation(filtered_C3)
lags_C4, corr_C4 = autocorrelation(filtered_C4)

def auto_corr(lags_C3,corr_C3,lags_C4,corr_C4):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(lags_C3, corr_C3)
    plt.title('Autocorrelación C3')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelación')

    plt.subplot(2, 1, 2)
    plt.plot(lags_C4, corr_C4)
    plt.title('Autocorrelación C4')
    plt.xlabel('Lags')
    plt.ylabel('Autocorrelación')

    plt.tight_layout()
    plt.show()

auto_corr(lags_C3,corr_C3,lags_C4,corr_C4)

