# Ahmed's Helper Functions for ECG Project
# These functions help process and visualize heart signal data

import os
# Turn off TensorFlow warnings because they're annoying
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

import wfdb  # For reading heart signal files
import pywt   # For signal processing
import seaborn  # For making nice plots
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def clean_signal_step(measurement, previous_estimate, previous_uncertainty, process_noise, measurement_noise):
    """
    This function does one step of signal cleaning using Kalman filter
    """
    # Predict where the signal should be
    predicted_value = previous_estimate
    predicted_uncertainty = previous_uncertainty + process_noise
    
    # Calculate how much we trust the new measurement
    trust_factor = predicted_uncertainty / (predicted_uncertainty + measurement_noise)
    
    # Update our estimate based on new measurement
    new_estimate = predicted_value + trust_factor * (measurement - predicted_value)
    new_uncertainty = (1 - trust_factor) * predicted_uncertainty
    
    return new_estimate, new_uncertainty

def clean_signal_twice(heart_signal, noise_level1, measurement_noise1, noise_level2, measurement_noise2):
    """
    Clean the heart signal twice to make it super clear
    First cleaning removes big noise, second cleaning removes small noise
    """
    signal_length = len(heart_signal)
    
    # Arrays to store cleaned signals
    cleaned_signal1 = np.zeros(signal_length)
    uncertainty1 = np.full(signal_length, 1.0)
    cleaned_signal2 = np.zeros(signal_length)
    uncertainty2 = np.full(signal_length, 1.0)
    
    # Start with the first measurement
    cleaned_signal1[0] = heart_signal[0]
    cleaned_signal2[0] = heart_signal[0]
    
    # Clean the signal twice
    for i in range(1, signal_length):
        # First cleaning
        cleaned_signal1[i], uncertainty1[i] = clean_signal_step(
            heart_signal[i], 
            cleaned_signal1[i-1], 
            uncertainty1[i-1], 
            noise_level1, 
            measurement_noise1
        )
        # Second cleaning
        cleaned_signal2[i], uncertainty2[i] = clean_signal_step(
            cleaned_signal1[i], 
            cleaned_signal2[i-1], 
            uncertainty2[i-1], 
            noise_level2, 
            measurement_noise2
        )
    
    return cleaned_signal2

def read_patient_data(patient_number, signal_data, labels):
    """
    Read and process ECG data for one patient
    """
    # These are the different types of heartbeats we're looking for
    heart_beat_types = ['N', 'A', 'V', 'L', 'R']
    print("Reading heart signals from patient " + patient_number)
    
    # Read the patient's ECG data
    my_folder = "./"
    heart_record = wfdb.rdrecord(my_folder+'mit-bih-arrhythmia-database/' + patient_number, 
                                channel_names=['MLII'])
    raw_signal = heart_record.p_signal.flatten()
    
    # Clean up the signal
    noise_level1, measuring_noise1 = 0.001, 10  # For first cleaning
    noise_level2, measuring_noise2 = 0.001, 1   # For second cleaning
    clean_signal = clean_signal_twice(raw_signal, noise_level1, measuring_noise1, 
                                    noise_level2, measuring_noise2)
    
    # Get the locations of important heart beats
    heart_beats = wfdb.rdann('mit-bih-arrhythmia-database/' + patient_number, 'atr')
    beat_locations = heart_beats.sample
    beat_types = heart_beats.symbol
    
    # Skip first 10 and last 5 beats because they might be messy
    start_beat = 10
    end_beat = 5
    current_beat = start_beat
    last_beat = len(beat_types) - end_beat
    
    # Process each heartbeat
    while current_beat < last_beat:
        try:
            # Figure out what type of heartbeat it is
            beat_label = heart_beat_types.index(beat_types[current_beat])
            
            # Get the signal around this heartbeat (300 points)
            beat_signal = clean_signal[beat_locations[current_beat] - 99:
                                     beat_locations[current_beat] + 201]
            
            # Save it if it's the right length
            if len(beat_signal) == 300:
                signal_data.append(beat_signal)
                labels.append(beat_label)
            current_beat += 1
        except ValueError:
            # Skip beats we don't care about
            current_beat += 1
    return

def load_data(test_size, random_number):
    """
    Load and prepare all the ECG data for training
    """
    # List of all patients we're looking at
    patient_list = ['100', '101', '103', '105', '106', '107', '108', '109', '111', 
                   '112', '113', '114', '115', '116', '117', '119', '121', '122', 
                   '123', '124', '200', '201', '202', '203', '205', '208', '210', 
                   '212', '213', '214', '215', '217', '219', '220', '221', '222', 
                   '223', '228', '230', '231', '232', '233', '234']
    
    # Lists to store all our data
    all_signals = []
    all_labels = []
    
    # Get data from each patient
    for patient in patient_list:
        read_patient_data(patient, all_signals, all_labels)
    
    # Prepare data for training
    all_signals = np.array(all_signals).reshape(-1, 300)
    all_labels = np.array(all_labels).reshape(-1)
    
    # Split into training and testing sets
    train_signals, test_signals, train_labels, test_labels = train_test_split(
        all_signals, all_labels, 
        test_size=test_size, 
        random_state=random_number
    )
    
    return train_signals, test_signals, train_labels, test_labels

def show_results_heatmap(true_labels, predicted_labels):
    """
    Show how well our model did using a colored grid
    """
    # Calculate how many of each type we got right
    results_grid = confusion_matrix(true_labels, predicted_labels)
    
    # Make a nice colored plot
    plt.figure(figsize=(8, 8))
    seaborn.heatmap(results_grid, annot=True, fmt='.20g', cmap='Blues')
    plt.ylim(0, 5)
    plt.xlabel('What our model predicted')
    plt.ylabel('Actual heartbeat types')
    plt.title('How Well Our Model Did')
    plt.savefig('imgs/confusion_matrix.png')
    plt.show()

def plot_tensorflow_results(training_history):
    """
    Show how well we did during training (TensorFlow version)
    """
    # Plot accuracy over time
    plt.figure(figsize=(8, 8))
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('How Accurate Our Model Got')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Round')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig('imgs/accuracy.png')
    plt.show()
    
    # Plot loss over time
    plt.figure(figsize=(8, 8))
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('How Much Our Model Was Wrong')
    plt.ylabel('Loss')
    plt.xlabel('Training Round')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig('imgs/loss.png')
    plt.show()

def plot_pytorch_results(training_history):
    """
    Show how well we did during training (PyTorch version)
    """
    # Plot accuracy over time
    plt.figure(figsize=(8, 8))
    plt.plot(training_history['train_acc'])
    plt.plot(training_history['test_acc'])
    plt.title('How Accurate Our Model Got')
    plt.ylabel('Accuracy')
    plt.xlabel('Training Round')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig('imgs/accuracy.png')
    plt.show()
    
    # Plot loss over time
    plt.figure(figsize=(8, 8))
    plt.plot(training_history['train_loss'])
    plt.plot(training_history['test_loss'])
    plt.title('How Much Our Model Was Wrong')
    plt.ylabel('Loss')
    plt.xlabel('Training Round')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.savefig('imgs/loss.png')
    plt.show()