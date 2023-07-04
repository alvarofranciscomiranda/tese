import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

filename = 'resources/dataset_seconds.csv'

data = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
    
# Extract the features and target from the dataset
X = data[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values
y = data['Resultado'].values


# Preprocess the data using scikit-learn
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the neural network architecture using PyTorch
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Instantiate the neural network model
input_size = X_train.shape[1]
hidden_size = 64
num_classes = len(set(y))

model = NeuralNet(input_size, hidden_size, num_classes)

# Define the loss function, optimizer, and learning rate
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

y_train_array = np.array(y_train)

# Train the neural network using PyTorch
num_epochs = 10
batch_size = 32


for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = torch.tensor(X_train[i:i+batch_size], dtype=torch.float32)
        labels = torch.tensor(y_train[i:i+batch_size], dtype=torch.long)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the neural network using scikit-learn metrics
from sklearn.metrics import accuracy_score

with torch.no_grad():
    model.eval()
    inputs = torch.tensor(X_test, dtype=torch.float32)
    labels = torch.tensor(y_test, dtype=torch.long)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    accuracy = accuracy_score(labels, predicted)
    print(f'Test Accuracy: {accuracy:.4f}')
