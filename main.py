import numpy as np
import pandas as pd
import matplotlib.pyplot as matplt
from scipy.signal import butter, filtfilt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# loading signals, labels, and started sample time for given subject identfier
def load_data(subject_id):
    signals = pd.read_csv(f'Subject{subject_id}_Signals.csv', header=None).values
    labels = pd.read_csv(f'Subject{subject_id}_Labels.csv', header=None).values.flatten()
    sample_time = pd.read_csv(f'Subject{subject_id}_Trial.csv', header=None).values.flatten()
    return signals, labels, sample_time

# apply the common average filter on given signal
def CAR_filter(signals):
    car_signals = signals - np.mean(signals, axis=1, keepdims=True)
    return car_signals

# plot the first three channels before and after CAR filter for subject 1 as given in requirments
def plot_CAR_filter_effect(signals, car_signals, sampling_rate):
    num_channels = 3
    time = np.arange(signals.shape[0]) / sampling_rate
    
    fig, axs = matplt.subplots(num_channels, 2, figsize=(11, 7))
    fig.subplots_adjust(hspace=1)
    
    for i in range(num_channels):
        axs[i, 0].plot(time, signals[:, i])
        axs[i, 0].set_title(f'Subject 1: Channel {i+1} - Before CAR')
        axs[i, 0].set_xlabel('Time (s)')
        axs[i, 0].set_ylabel('Amplitude')
        
        axs[i, 1].plot(time, car_signals[:, i])
        axs[i, 1].set_title(f'Subject 1: Channel {i+1} - After CAR')
        axs[i, 1].set_xlabel('Time (s)')
        axs[i, 1].set_ylabel('Amplitude')
    
    matplt.show()


# implement bandpass filter 
def bandpass_filter(signal, sampling_rate, lowcut, highcut):
    nyquist = 0.5 * sampling_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(5, [low, high], btype='band')# order 4 and 5 simulated 
    filtered_signal = filtfilt(b, a, signal, axis=0)
    return filtered_signal

# Function to compute relative change in the band for each trial
def compute_relative_change(signal, trial_start, pre_onset_start, sampling_rate):
    trial_samples = int(5 * sampling_rate)  # 5 seconds trial duration
    pre_onset_samples = int(5 * sampling_rate)  # 5 seconds pre-onset duration
    
    trial_signal = signal[trial_start:trial_start + trial_samples]
    pre_onset_signal = signal[pre_onset_start:pre_onset_start + pre_onset_samples]
    
    relative_change = (np.mean(trial_signal, axis=0) - np.mean(pre_onset_signal, axis=0)) / np.mean(pre_onset_signal, axis=0)
    
    return relative_change

mu_band = (8, 13)  
beta_band = (13, 30) 

def KNN_classify(signals, labels, sample_time, sampling_rate,concat):
    num_trials = len(sample_time)
    num_channels = signals.shape[1]
    pre_onset_starts = sample_time - int(5 * sampling_rate)  # 5 seconds pre-onset duration
    
    min_error_mu = float('inf')
    min_error_beta = float('inf')
    best_electrode_mu = None
    best_electrode_beta = None
    best_k_mu = None
    best_k_beta = None
    
    for electrode in range(num_channels):
        mu_band_rel_changes = []
        beta_band_rel_changes = []
        all_features_mu = []  
        all_features_beta = []  
        
        for i in range(num_trials):
            trial_start = int(sample_time[i])
            pre_onset_start = int(pre_onset_starts[i])
            
            mu_band_filtered = bandpass_filter(signals[:, electrode], sampling_rate, mu_band[0], mu_band[1])
            mu_rel_change = compute_relative_change(mu_band_filtered, trial_start, pre_onset_start, sampling_rate)
            mu_band_rel_changes.append(mu_rel_change)
            
            beta_band_filtered = bandpass_filter(signals[:, electrode], sampling_rate, beta_band[0], beta_band[1])
            beta_rel_change = compute_relative_change(beta_band_filtered, trial_start, pre_onset_start, sampling_rate)
            beta_band_rel_changes.append(beta_rel_change)
        
        mu_band_rel_changes = np.array(mu_band_rel_changes)
        beta_band_rel_changes = np.array(beta_band_rel_changes)
        
        if mu_band_rel_changes.ndim == 1:
            mu_band_rel_changes = mu_band_rel_changes[:, np.newaxis]
        if beta_band_rel_changes.ndim == 1:
            beta_band_rel_changes = beta_band_rel_changes[:, np.newaxis]
        if concat==1:
             all_features_mu.append(mu_band_rel_changes)
             all_features_beta.append(beta_band_rel_changes)
             all_features_mu = np.concatenate(all_features_mu, axis=1)
             all_features_beta = np.concatenate(all_features_beta, axis=1)
             scaler_mu = StandardScaler()
             scaler_beta = StandardScaler()
             features_mu = scaler_mu.fit_transform(all_features_mu)
             features_beta = scaler_beta.fit_transform(all_features_beta)
             
        else :
            scaler_mu = StandardScaler()
            scaler_beta = StandardScaler()
            features_mu = scaler_mu.fit_transform(mu_band_rel_changes)
            features_beta = scaler_beta.fit_transform(beta_band_rel_changes)
        
        for k in range(1, 11):  
            knn_mu = KNeighborsClassifier(n_neighbors=k)
            scores_mu = cross_val_score(knn_mu, features_mu, labels, cv=10, scoring='accuracy')
            error_mu = 1 - np.mean(scores_mu)
            
            if error_mu < min_error_mu:
                min_error_mu = error_mu
                best_electrode_mu = electrode
                best_k_mu = k
            
          
            knn_beta = KNeighborsClassifier(n_neighbors=k)
            scores_beta = cross_val_score(knn_beta, features_beta, labels, cv=10, scoring='accuracy')
            error_beta = 1 - np.mean(scores_beta)
            
            if error_beta < min_error_beta:
                min_error_beta = error_beta
                best_electrode_beta = electrode
                best_k_beta = k
        
    
    return min_error_mu, best_electrode_mu, best_k_mu, min_error_beta, best_electrode_beta, best_k_beta

sign1, labl1, sample_time1 = load_data(1)
sign2, labl2, sample_time2 = load_data(2)
sign3, labl3, sample_time3 = load_data(3)
sampling_rate = 512
car_sign1 = CAR_filter(sign1)
car_sign2 = CAR_filter(sign2)
car_sign3 = CAR_filter(sign3)

plot_CAR_filter_effect(sign1, car_sign1, sampling_rate)

mu_err1, mu_electr1, k1_mu, beta_err1, beta_electr1, k1_beta = KNN_classify(car_sign1, labl1, sample_time1, sampling_rate,0)
mu_err2, mu_electr2, k2_mu, beta_err2, beta_electr2, k2_beta = KNN_classify(car_sign2, labl2, sample_time2, sampling_rate,0)
mu_err3, mu_electr3, k3_mu, beta_err3, beta_electr3, k3_beta = KNN_classify(car_sign3, labl3, sample_time3, sampling_rate,0)

avg_mu_err = (mu_err1 + mu_err2 + mu_err3) / 3
avg_beta_err = (beta_err1 + beta_err2 + beta_err3) / 3

print(f"Subject 1: Mu band - Best electrode: {mu_electr1}, K: {k1_mu}, Error: {mu_err1}")
print(f"Subject 1: Beta band - Best electrode: {beta_electr1}, K: {k1_beta}, Error: {beta_err1}")

print(f"Subject 2: Mu band - Best electrode: {mu_electr2}, K: {k2_mu}, Error: {mu_err2}")
print(f"Subject 2: Beta band - Best electrode: {beta_electr2}, K: {k2_beta}, Error: {beta_err2}")

print(f"Subject 3: Mu band - Best electrode: {mu_electr3}, K: {k3_mu}, Error: {mu_err3}")
print(f"Subject 3: Beta band - Best electrode: {beta_electr3}, K: {k3_beta}, Error: {beta_err3}")

print(f"Average across subjects: Mu band - Error: {avg_mu_err}")
print(f"Average across subjects: Beta band - Error: {avg_beta_err}")

mu_err1_concat,mu_electr1_conc, k1_mu_concat, beta_err1_concat,beta_electr1_conc, k1_beta_concat = KNN_classify(car_sign1, labl1, sample_time1, sampling_rate,1)
mu_err2_concat,mu_electr2_conc, k2_mu_concat, beta_err2_concat,beta_electr2_conc, k2_beta_concat = KNN_classify(car_sign2, labl2, sample_time2, sampling_rate,1)
mu_err3_concat,mu_electr3_conc, k3_mu_concat, beta_err3_concat,beta_electr3_conc, k3_beta_concat = KNN_classify(car_sign3, labl3, sample_time3, sampling_rate,1)

print(f"Subject 1 (Concate): Mu band - Best electrode: {mu_electr1_conc}, K: {k1_mu_concat}, Error: {mu_err1_concat}")
print(f"Subject 1 (Concate): Beta band - Best electrode: {beta_electr1_conc}, K: {k1_beta_concat}, Error: {beta_err1_concat}")

print(f"Subject 2 (Concate): Mu band - Best electrode: {mu_electr2_conc}, K: {k2_mu_concat}, Error: {mu_err2_concat}")
print(f"Subject 2 (Concate): Beta band - Best electrode: {beta_electr2_conc}, K: {k2_beta_concat}, Error: {beta_err2_concat}")

print(f"Subject 3 (Concate): Mu band - Best electrode: {mu_electr3_conc}, K: {k3_mu_concat}, Error: {mu_err3_concat}")
print(f"Subject 3 (Concate): Beta band - Best electrode: {beta_electr3_conc}, K: {k3_beta_concat}, Error: {beta_err3_concat}")
