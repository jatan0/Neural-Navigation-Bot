import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

with open("data_stochastic.pkl", "rb") as f:
    data_stochastic = pickle.load(f)

with open("data_stationary.pkl", "rb") as f:
    data_stationary = pickle.load(f)

action_mapping = {"up": 0, "down": 1, "left": 2, "right": 3, "sense": 4, "stay": 4}


def preprocess_data(data):
    X = []
    y = []

    for item in data:
        grid, bot_pos, sensor_data, actions = item
        grid_flat = grid.flatten()
        sensor_data_flat = sensor_data

        scaler = MinMaxScaler()
        grid_normalized = scaler.fit_transform(grid_flat.reshape(-1, 1)).flatten()
        bot_pos_normalized = scaler.fit_transform(
            np.array(bot_pos).reshape(-1, 1)
        ).flatten()

        features = np.hstack((grid_normalized, bot_pos_normalized, sensor_data_flat))

        for action in actions:
            if isinstance(action, tuple) and len(action) > 0:
                mapped_action = action_mapping.get(action[0], -1)
                if mapped_action != -1:
                    X.append(features)
                    y.append(mapped_action)
                else:
                    print(f"unknown output: {action}")

    X = np.array(X)
    y = np.array(y)

    print(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y


X_stochastic, y_stochastic = preprocess_data(data_stochastic)
X_stationary, y_stationary = preprocess_data(data_stationary)
X_train_stochastic, X_temp_stochastic, y_train_stochastic, y_temp_stochastic = (
    train_test_split(X_stochastic, y_stochastic, test_size=0.3, random_state=42)
)
X_val_stochastic, X_test_stochastic, y_val_stochastic, y_test_stochastic = (
    train_test_split(
        X_temp_stochastic, y_temp_stochastic, test_size=0.5, random_state=42
    )
)

X_train_stationary, X_temp_stationary, y_train_stationary, y_temp_stationary = (
    train_test_split(X_stationary, y_stationary, test_size=0.3, random_state=42)
)
X_val_stationary, X_test_stationary, y_val_stationary, y_test_stationary = (
    train_test_split(
        X_temp_stationary, y_temp_stationary, test_size=0.5, random_state=42
    )
)

print(f"stochastic training set: {X_train_stochastic.shape}, {len(y_train_stochastic)}")
print(f"stochastic validation set: {X_val_stochastic.shape}, {len(y_val_stochastic)}")
print(f"stochastic test set: {X_test_stochastic.shape}, {len(y_test_stochastic)}")

print(f"stationary training set: {X_train_stationary.shape}, {len(y_train_stationary)}")
print(f"stationary validation set: {X_val_stationary.shape}, {len(y_val_stationary)}")
print(f"stationary test set: {X_test_stationary.shape}, {len(y_test_stationary)}")
