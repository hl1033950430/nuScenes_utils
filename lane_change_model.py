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

        self.target_vehicle_encoding = nn.Sequential(nn.Linear(self.target_feature_size, embedding_size), nn.BatchNorm1d(embedding_size), nn.LeakyReLU())
        self.surrounding_vehicle_encoding = nn.Sequential(nn.Linear(self.surrounding_feature_size, embedding_size), nn.BatchNorm1d(embedding_size), nn.LeakyReLU())
        self.lstm_surrounding = nn.LSTM(self.embedding_size, lstm_hidden_size, num_stacked_layers, batch_first=True)
        self.lstm_target = nn.LSTM(self.embedding_size, lstm_hidden_size, num_stacked_layers, batch_first=True)
        self.att = nn.MultiheadAttention(self.lstm_hidden_size, 16, 0.5, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(lstm_hidden_size, fc_size[0]), nn.BatchNorm1d(fc_size[0]), nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(fc_size[0], fc_size[1]), nn.BatchNorm1d(fc_size[1]), nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(fc_size[1], fc_size[2]), nn.BatchNorm1d(fc_size[2]), nn.LeakyReLU())
        self.fc4 = nn.Linear(fc_size[2], 3)
        self.activation = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, target_vehicle_vector, surrounding_vehicle_vector, device):
        batch_size = len(target_vehicle_vector)

        # target vehicle embedding
        target_vehicle_vector = target_vehicle_vector.view(batch_size * self.time_horizon, self.target_feature_size)
        target_vehicle_vector = self.target_vehicle_encoding(target_vehicle_vector)
        target_vehicle_vector = target_vehicle_vector.view(batch_size, self.time_horizon, self.embedding_size)

        # target vehicle lstm encoding
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.lstm_hidden_size).to(device)
        lstm_target_out, _ = self.lstm_target(target_vehicle_vector, (h0, c0))
        lstm_target_out = lstm_target_out[:, -1, :].view(batch_size, 1, self.lstm_hidden_size)

        # surrounding vehicle embedding
        surrounding_vehicle_vector = surrounding_vehicle_vector.view(batch_size * self.time_horizon * 5, self.surrounding_feature_size)
        surrounding_vehicle_vector = self.surrounding_vehicle_encoding(surrounding_vehicle_vector)
        surrounding_vehicle_vector = surrounding_vehicle_vector.view(batch_size, self.time_horizon, 5, self.embedding_size)

        # surrounding vehicle lstm encoding
        h0 = torch.zeros(self.num_stacked_layers, batch_size * 5, self.lstm_hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size * 5, self.lstm_hidden_size).to(device)
        surrounding_vehicle_vector = surrounding_vehicle_vector.view(batch_size * 5, self.time_horizon, self.embedding_size)
        lstm_surrounding_out, _ = self.lstm_surrounding(surrounding_vehicle_vector, (h0, c0))
        lstm_surrounding_out = lstm_surrounding_out[:, -1, :]
        lstm_surrounding_out = lstm_surrounding_out.view(batch_size, 5, self.lstm_hidden_size)

        # ATT (waiting to do)
        q = lstm_target_out
        k, v = lstm_surrounding_out, lstm_surrounding_out
        att_output, _ = self.att(q, k, v)

        # FC
        input_encoding = att_output.view(batch_size, self.lstm_hidden_size)
        output = self.fc1(input_encoding)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.softmax(self.fc4(output))

        return output