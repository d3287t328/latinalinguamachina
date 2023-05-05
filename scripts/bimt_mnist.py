import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class BIMTLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(in_features, out_features))
        self.biases = nn.Parameter(torch.randn(out_features))
        self.register_buffer('coordinates', torch.randn(in_features, out_features))

    def forward(self, x):
        return torch.matmul(x, self.weights) + self.biases

    def connection_lengths(self):
        return torch.norm(self.coordinates, dim=0)

class BIMTNetwork(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList(
            BIMTLayer(in_features, out_features)
            for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:])
        )

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        x = self.layers[-1](x)
        return x

    def connection_cost(self, scale_factor):
        cost = 0
        for layer in self.layers:
            cost += torch.sum(layer.connection_lengths())
        return cost * scale_factor

    def flatten(self, x):
        return x.view(x.size(0), -1)

def train_bimt_network(net, train_loader, val_loader, epochs, learning_rate, connection_scale_factor):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        net.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) + net.connection_cost(connection_scale_factor)
            loss.backward()
            optimizer.step()

        net.eval()
        train_loss, train_accuracy = evaluate(net, train_loader, criterion)
        val_loss, val_accuracy = evaluate(net, val_loader, criterion)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n")

def evaluate(net, data_loader, criterion):
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return average_loss, accuracy

if __name__ == '__main__':
    # Load the MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    val_data = datasets.MNIST(root='.', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

    # Define the layer_sizes and training parameters
    layer_sizes = [28 * 28, 128, 64, 10]
    epochs = 10
    learning_rate = 0.001
    connection_scale_factor = 0.01

    # Create and train the BIMTNetwork
    net = BIMTNetwork(layer_sizes)
    train_bimt_network(net, train_loader, val_loader, epochs, learning_rate, connection_scale_factor)
