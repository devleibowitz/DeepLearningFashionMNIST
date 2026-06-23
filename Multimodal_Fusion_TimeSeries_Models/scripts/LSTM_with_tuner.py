import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import os

# Define your LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Define your dataset
class CTGDataset(Dataset):
    def __init__(self, data_tensor, labels_csv):
        self.data = data_tensor
        self.labels = pd.read_csv(labels_csv)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.tensor(self.labels.iloc[idx]['label'], dtype=torch.long)
        return sample, label

# Load data
def load_data(data_path, labels_path):
    data_tensor = torch.load(data_path)
    labels_df = pd.read_csv(labels_path)
    return data_tensor, labels_df

# Prepare data
def prepare_data(data_path, labels_path, batch_size=32):
    data_tensor, labels_df = load_data(data_path, labels_path)
    
    dataset = CTGDataset(data_tensor, labels_path)
    
    # Calculate class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels_df['label']), y=labels_df['label'])
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Create a sampler with class weights
    sample_weights = [class_weights[label] for label in labels_df['label']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(labels_df), replacement=True)
    
    # Create a DataLoader with the sampler
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    
    return data_loader

# Define a function to train the model
def train_model(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_data, batch_labels in data_loader:
        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)
    
    accuracy = correct / total
    return total_loss / len(data_loader), accuracy

# Define a function to evaluate the model
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    report = classification_report(all_labels, all_predictions)
    return total_loss / len(data_loader), accuracy, report

# Grid search parameters
param_grid = {
    'input_size': [21620],
    'hidden_size': [64, 128],
    'num_layers': [1, 2],
    'num_classes': [2],
    'batch_size': [32, 64],
    'learning_rate': [0.001, 0.01]
}

# Perform grid search
def perform_grid_search(data_path, labels_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    best_accuracy = 0
    best_params = None
    
    for input_size in param_grid['input_size']:
        for hidden_size in param_grid['hidden_size']:
            for num_layers in param_grid['num_layers']:
                for num_classes in param_grid['num_classes']:
                    for batch_size in param_grid['batch_size']:
                        for learning_rate in param_grid['learning_rate']:
                            model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
                            criterion = nn.CrossEntropyLoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                            
                            data_loader = prepare_data(data_path, labels_path, batch_size)
                            
                            # Train the model
                            train_loss, train_accuracy = train_model(model, data_loader, criterion, optimizer, device)
                            
                            # Evaluate the model
                            eval_loss, eval_accuracy, report = evaluate_model(model, data_loader, criterion, device)
                            
                            print(f"Params: {input_size, hidden_size, num_layers, num_classes, batch_size, learning_rate}")
                            print(f"Train Accuracy: {train_accuracy}, Eval Accuracy: {eval_accuracy}")
                            print(report)
                            
                            if eval_accuracy > best_accuracy:
                                best_accuracy = eval_accuracy
                                best_params = (input_size, hidden_size, num_layers, num_classes, batch_size, learning_rate)
    
    print(f"Best Parameters: {best_params}")
    print(f"Best Accuracy: {best_accuracy}")

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(ROOT_DIR)

    # Set other directories
    MODEL_DIR = os.path.join(ROOT_DIR, 'models')
    DATA_DIR = os.path.join(ROOT_DIR, 'data')

    # Data paths
    data_path = os.path.join(DATA_DIR, "processed", "ctg_tensor.pt")
    labels_path = os.path.join(DATA_DIR, "processed", "labels.csv")
    perform_grid_search(data_path, labels_path)
