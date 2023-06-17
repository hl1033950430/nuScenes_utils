import torch
from torch import nn


class LaneChangeModel(nn.Module):

    def __init__(self, embedding_size, lstm_hidden_size, num_stacked_layers, fc_size, time_horizon):
        super().__init__()
        self.embedding_size = embedding_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.time_horizon = time_horizon
        self.surrounding_feature_size = 15
        self.target_feature_size = 14

        self.lstm_surrounding = nn.LSTM(self.surrounding_feature_size, lstm_hidden_size, num_stacked_layers, batch_first=True)
        self.lstm_target = nn.LSTM(self.target_feature_size, lstm_hidden_size, num_stacked_layers, batch_first=True)
        self.fc1 = nn.Linear(6 * lstm_hidden_size, fc_size[0])
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.fc3 = nn.Linear(fc_size[1], fc_size[2])
        self.fc4 = nn.Linear(fc_size[2], 3)
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax()

    def forward(self, target_vehicle_vector, surrounding_vehicle_vector, device):
        batch_size = len(target_vehicle_vector)

        # target vehicle lstm encoding
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.lstm_hidden_size).to(device)
        lstm_target_out, _ = self.lstm_target(target_vehicle_vector, (h0, c0))
        lstm_target_out = lstm_target_out[:, -1, :].view(batch_size, self.lstm_hidden_size)

        # surrounding vehicle lstm encoding
        h0 = torch.zeros(self.num_stacked_layers, batch_size * 5, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size * 5, self.lstm_hidden_size).to(device)
        surrounding_vehicle_vector = surrounding_vehicle_vector.view(batch_size * 5, self.time_horizon, 15)
        lstm_surrounding_out, _ = self.lstm_surrounding(surrounding_vehicle_vector, (h0, c0))
        lstm_surrounding_out = lstm_surrounding_out[:, -1, :]
        lstm_surrounding_out = lstm_surrounding_out.view(batch_size, self.lstm_hidden_size * 5)

        # ATT (waiting to do)

        # FC
        input_encoding = torch.cat([lstm_target_out, lstm_surrounding_out], dim=1)
        output = self.activation(self.fc1(input_encoding))
        output = self.activation(self.fc2(output))
        output = self.activation(self.fc3(output))
        output = self.softmax(self.fc4(output))

        return output