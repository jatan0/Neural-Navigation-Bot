import torch
import torch.nn as nn
import torch.optim as optim
from preprocess_data import X_train_stochastic, y_train_stochastic
from preprocess_data import X_train_stationary, y_train_stationary


class BotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(BotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


input_size = X_train_stochastic.shape[1]
output_size = 5  # up, down, left, right, sense
model = BotModel(input_size, output_size)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
