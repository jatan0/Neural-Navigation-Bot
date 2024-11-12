import torch
from torch.utils.data import DataLoader, TensorDataset
from model import model, criterion, optimizer
import numpy as np
import matplotlib.pyplot as plt

from preprocess_data import (
    X_train_stochastic,
    y_train_stochastic,
    X_val_stochastic,
    y_val_stochastic,
    X_test_stochastic,
    y_test_stochastic,
)


def get_data_loader(X, y, batch_size=32):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


val_loader_full = get_data_loader(X_val_stochastic, y_val_stochastic)
test_loader_full = get_data_loader(X_test_stochastic, y_test_stochastic)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))

        print(
            f"epoch: {epoch+1}/{num_epochs}, train loss: {train_loss/len(train_loader)}, val Loss: {val_loss/len(val_loader)}"
        )

    return train_losses, val_losses


def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    print(f"loss: {avg_loss}, acc: {accuracy}")


subset_size = 5000
indices = np.random.choice(len(X_train_stochastic), subset_size, replace=False)
X_train_subset = X_train_stochastic[indices]
y_train_subset = y_train_stochastic[indices]
train_loader_subset = get_data_loader(X_train_subset, y_train_subset, batch_size=64)
train_losses, val_losses = train_model(
    model, train_loader_subset, val_loader_full, criterion, optimizer, num_epochs=20
)

print("eval on full val set:")
evaluate_model(model, val_loader_full, criterion)
print("eval on full test set:")
evaluate_model(model, test_loader_full, criterion)
torch.save(model.state_dict(), "bot_model.pth")


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()


plot_loss(train_losses, val_losses)
