# Ahmed's ECG Heart Signal Classification Project
# Date: October 2024
# This program classifies ECG signals into different categories using machine learning

import os
import datetime
import sys
# Add my code folder to Python so it can find my helper functions
my_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(my_folder)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parameter import Parameter
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from utils import load_data, plot_pytorch_results, show_results_heatmap

# Set up folders for saving stuff
my_project_folder = "./"
# Create a folder to save training progress with current date/time
training_logs_folder = my_project_folder + "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
saved_model_file = my_project_folder + "models/heart_signal_model.pt"

# Check if we can use GPU, otherwise use CPU
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("I'm using {} for training".format(my_device))

# Make a class to handle our ECG data
class HeartSignalData(Dataset):
    def __init__(self, signal_data, labels):
        self.signal_data = signal_data
        self.labels = labels

    def __getitem__(self, idx):
        single_signal = torch.tensor(self.signal_data[idx], dtype=torch.float32)
        single_label = torch.tensor(self.labels[idx], dtype=torch.long)
        return single_signal, single_label

    def __len__(self):
        return len(self.signal_data)

# My neural network for classifying heart signals
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer to process the signal
        self.first_conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding='same')
        self.first_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Second layer to find more patterns
        self.second_conv = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding='same')
        self.second_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Third layer for complex features
        self.third_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same')
        self.third_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        
        # Fourth layer for final signal processing
        self.fourth_conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding='same')
        
        # Make the data flat for the neural network
        self.make_flat = nn.Flatten()
        
        # Final layers to make the prediction
        self.first_dense = nn.Linear(64 * 38, 128)
        self.dropout_layer = nn.Dropout(0.2)
        self.final_dense = nn.Linear(128, 5)

    def forward(self, x):
        # Reshape the input signal
        x = x.reshape(-1, 1, 300)
        
        # Apply each layer one by one
        x = F.relu(self.first_conv(x))
        x = self.first_pool(x)
        x = F.relu(self.second_conv(x))
        x = self.second_pool(x)
        x = F.relu(self.third_conv(x))
        x = self.third_pool(x)
        x = F.relu(self.fourth_conv(x))
        x = self.make_flat(x)
        x = F.relu(self.first_dense(x))
        x = self.dropout_layer(x)
        x = self.final_dense(x)
        return x

# My advanced network that combines CNN and LSTM
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN layers to process the signal
        self.first_conv = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=21, stride=1, padding='same')
        self.first_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.second_conv = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=23, stride=1, padding='same')
        self.second_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.third_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=25, stride=1, padding='same')
        self.third_pool = nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
        self.fourth_conv = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=27, stride=1, padding='same')
        
        # LSTM layer to understand signal patterns over time
        self.lstm_layer = nn.LSTM(input_size=64, hidden_size=50, num_layers=1, batch_first=True)
        
        # Final layers for classification
        self.first_dense = nn.Linear(50, 128)
        self.dropout_layer = nn.Dropout(0.2)
        self.final_dense = nn.Linear(128, 5)

    def forward(self, x):
        # Process the signal through CNN layers
        x = x.reshape(-1, 1, 300)
        x = F.relu(self.first_conv(x))
        x = self.first_pool(x)
        x = F.relu(self.second_conv(x))
        x = self.second_pool(x)
        x = F.relu(self.third_conv(x))
        x = self.third_pool(x)
        x = F.relu(self.fourth_conv(x))
        
        # Prepare data for LSTM
        x = x.permute(0, 2, 1)
        
        # Process through LSTM
        x, (h_n, c_n) = self.lstm_layer(x)
        x = x[:, -1, :]
        
        # Make final prediction
        x = F.relu(self.first_dense(x))
        x = self.dropout_layer(x)
        x = self.final_dense(x)
        return x

# Function to train the model for one epoch
def train_one_epoch(progress_bar, model, loss_function, optimizer):
    losses_in_epoch = []
    accuracies_in_epoch = []
    model.train()
    
    for batch_num, (signals, labels) in progress_bar:
        # Move data to GPU if available
        signals = signals.to(my_device)
        labels = labels.to(my_device)
        
        # Make predictions
        predictions = model(signals)
        current_loss = loss_function(predictions, labels)
        
        # Update the model
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        current_loss = current_loss.item()
        losses_in_epoch.append(current_loss)
        predicted_classes = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        true_labels = labels.detach().cpu().numpy()
        current_accuracy = accuracy_score(true_labels, predicted_classes)
        accuracies_in_epoch.append(current_accuracy)
        
        # Show progress
        progress_bar.set_postfix(loss=current_loss, acc=current_accuracy)
    
    return {
        "loss": np.mean(losses_in_epoch),
        "acc": np.mean(accuracies_in_epoch)
    }

# Function to test the model
def test_model(progress_bar, model, loss_function):
    test_losses = []
    test_accuracies = []
    model.eval()
    
    with torch.no_grad():
        for batch_num, (signals, labels) in progress_bar:
            # Move data to GPU if available
            signals = signals.to(my_device)
            labels = labels.to(my_device)
            
            # Make predictions
            predictions = model(signals)
            current_loss = loss_function(predictions, labels).item()
            
            # Calculate accuracy
            test_losses.append(current_loss)
            predicted_classes = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            true_labels = labels.detach().cpu().numpy()
            current_accuracy = accuracy_score(true_labels, predicted_classes)
            test_accuracies.append(current_accuracy)
            
            # Show progress
            progress_bar.set_postfix(loss=current_loss, acc=current_accuracy)
    
    return {
        "loss": np.mean(test_losses),
        "acc": np.mean(test_accuracies)
    }

# Function to train the model for multiple epochs
def train_model(train_data, test_data, model, loss_function, optimizer, settings, logger):
    total_epochs = settings['num_epochs']
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for current_epoch in range(total_epochs):
        # Set up progress bars
        train_progress = tqdm(enumerate(train_data), total=len(train_data))
        test_progress = tqdm(enumerate(test_data), total=len(test_data))
        train_progress.set_description(f'Training Epoch [{current_epoch + 1}/{total_epochs}]')
        test_progress.set_description(f'Testing Epoch [{current_epoch + 1}/{total_epochs}]')
        
        # Train and test for this epoch
        train_results = train_one_epoch(train_progress, model, loss_function, optimizer)
        test_results = test_model(test_progress, model, loss_function)
        
        # Save results
        train_losses.append(train_results['loss'])
        train_accuracies.append(train_results['acc'])
        test_losses.append(test_results['loss'])
        test_accuracies.append(test_results['acc'])
        
        # Print progress
        print(f'Epoch {current_epoch + 1} Results:')
        print(f'Training - Loss: {train_results["loss"]}, Accuracy: {train_results["acc"]}')
        print(f'Testing  - Loss: {test_results["loss"]}, Accuracy: {test_results["acc"]}')
        
        # Log results for TensorBoard
        logger.add_scalar('training/loss', train_results['loss'], current_epoch)
        logger.add_scalar('training/accuracy', train_results['acc'], current_epoch)
        logger.add_scalar('testing/loss', test_results['loss'], current_epoch)
        logger.add_scalar('testing/accuracy', test_results['acc'], current_epoch)
    
    return {
        'train_loss': train_losses,
        'train_acc': train_accuracies,
        'test_loss': test_losses,
        'test_acc': test_accuracies
    }

# Main function to run everything
def run_program():
    # My settings for the program
    my_settings = {
        'seed': 42,  # For consistent results
        'test_size': 0.3,  # Use 30% of data for testing
        'num_epochs': 30,  # Train for 30 epochs
        'batch_size': 128,  # Process 128 signals at once
        'learning_rate': 0.001,  # How fast to learn
    }

    # Load and prepare the data
    train_signals, test_signals, train_labels, test_labels = load_data(
        my_settings['test_size'], 
        my_settings['seed']
    )
    
    # Create datasets
    train_dataset = HeartSignalData(train_signals, train_labels)
    test_dataset = HeartSignalData(test_signals, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=my_settings['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=my_settings['batch_size'], shuffle=False)

    # Create the model
    # model = CNN()  # Simple version
    model = LSTM()  # Advanced version with LSTM
    
    # Check if we have a saved model
    if os.path.exists(saved_model_file):
        print('Found a saved model! Loading it instead of training a new one')
        model.load_state_dict(torch.load(saved_model_file))
        model.eval()
    else:
        # Set up the model for training
        model = model.to(my_device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=my_settings['learning_rate'])

        # Show model structure
        summary(model, (my_settings['batch_size'], train_signals.shape[1]), 
               col_names=["input_size", "kernel_size", "output_size"],
               verbose=2)

        # Start training
        logger = SummaryWriter(log_dir=training_logs_folder)
        training_history = train_model(
            train_loader, 
            test_loader, 
            model, 
            loss_function, 
            optimizer, 
            my_settings, 
            logger
        )
        logger.close()
        
        # Save the trained model
        torch.save(model.state_dict(), saved_model_file)
        
        # Show training results
        plot_pytorch_results(training_history)

    # Test the model on all test data
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for batch_num, (signals, labels) in enumerate(test_loader):
            signals = signals.to(my_device)
            labels = labels.to(my_device)
            predictions = model(signals)
            batch_predictions = torch.argmax(predictions, dim=1).detach().cpu().numpy()
            all_predictions.extend(batch_predictions)
    
    # Show results as a heat map
    show_results_heatmap(test_labels, all_predictions)

# Run the program
if __name__ == '__main__':
    run_program()