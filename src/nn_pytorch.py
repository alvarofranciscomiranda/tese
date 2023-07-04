import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def run_nn_pytorch():
    filename = 'resources/dataset_seconds.csv'

    data = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
        
    # Extract the features and target from the dataset
    X = data[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values
    y = data['Resultado'].values

    # Preprocess the data using scikit-learn
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

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

    # Define the number of folds
    k_folds = 5

    # Initialize the KFold object
    kf = KFold(n_splits=k_folds, shuffle=True)

    # Initialize lists to store the evaluation metrics for each fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    # Perform k-fold cross-validation
    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        #print(f"Fold {fold + 1}")

        # Split the data into train and test sets for this fold
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Instantiate the neural network model
        input_size = X_train.shape[1]
        hidden_size = 64
        num_classes = len(set(y))

        model = NeuralNet(input_size, hidden_size, num_classes)

        # Define the loss function, optimizer, and learning rate
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

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

            #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluate the neural network using scikit-learn metrics
        with torch.no_grad():
            model.eval()
            inputs = torch.tensor(X_test, dtype=torch.float32)
            labels = torch.tensor(y_test, dtype=torch.long)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # Calculate the evaluation metrics
            accuracy = accuracy_score(labels, predicted)
            precision = precision_score(labels, predicted, average='macro', zero_division=0)
            recall = recall_score(labels, predicted, average='macro')
            f1 = f1_score(labels, predicted, average='macro')

            # Store the evaluation metrics for this fold
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

    # Calculate the average metrics across all folds
    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)

    # Print the average metrics
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")