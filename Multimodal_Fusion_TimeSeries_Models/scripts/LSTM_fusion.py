import torch
import torch.nn as nn

class LSTMFusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, weight_dim):
        super(LSTMFusionModel, self).__init__()

        # LSTM for time series input
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm_fc = nn.Linear(hidden_dim, 64)  # Reduce LSTM output dimension

        # Fully connected layer for mother's weight
        self.weight_fc = nn.Linear(weight_dim, 64)

        # Fusion layer + classification
        self.fusion_fc = nn.Linear(64 + 64, 32)  # Combine both inputs
        self.output = nn.Linear(32, 1)  # Binary classification

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Output probability

    def forward(self, time_series_input, weight_input):
        # LSTM branch
        lstm_out, _ = self.lstm(time_series_input)  
        lstm_out = lstm_out[:, -1, :]  # Take last time step
        lstm_out = self.relu(self.lstm_fc(lstm_out))

        # Weight branch
        weight_out = self.relu(self.weight_fc(weight_input))

        # Concatenate both features
        fusion = torch.cat((lstm_out, weight_out), dim=1)
        fusion = self.relu(self.fusion_fc(fusion))

        # Output layer
        output = self.sigmoid(self.output(fusion))
        return output
