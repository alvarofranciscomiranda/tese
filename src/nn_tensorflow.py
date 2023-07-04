import tensorflow as tf
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

def run_nn_tensorflow():
    filename = 'resources/dataset_seconds.csv'

    data = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
        
    # Extract the features and target from the dataset
    X = data[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values
    y = data['Resultado'].values

    # Preprocess the data using scikit-learn
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data into training and test sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Instantiate the neural network model
    input_size = X.shape[1]
    hidden_size = 64
    num_classes = len(set(y))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=hidden_size, activation='relu', input_shape=(input_size,)),
        tf.keras.layers.Dense(units=num_classes, activation='softmax')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    num_epochs = 10
    batch_size = 32

    # Define a function to evaluate the model
    def evaluate_model(X, y):
        scores = []
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-fold cross-validation
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=0)
            _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

            # Predict labels for the test set
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

            # Calculate precision, recall, and F1 score
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')

            scores.append((test_accuracy, precision, recall, f1))

        return scores

    scores = evaluate_model(X, y)

    test_accuracy_avg = np.mean([score[0] for score in scores])
    precision_avg = np.mean([score[1] for score in scores])
    recall_avg = np.mean([score[2] for score in scores])
    f1_avg = np.mean([score[3] for score in scores])

    
    print(f'Test Accuracy: {test_accuracy_avg:.4f}')
    print(f'Precision: {precision_avg:.4f}')
    print(f'Recall: {recall_avg:.4f}')
    print(f'F1 Score: {f1_avg:.4f}')