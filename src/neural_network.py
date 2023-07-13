from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import time



def run_neural_network(x_train, x_test, y_test, y_train):
    
    start_time = time.time()  # Record the start time
    
    # Instantiate the MLPClassifier model
    #model = MLPClassifier(hidden_layer_sizes=(10), max_iter=5000, alpha=0.0001, solver='adam', activation= 'logistic', random_state=10, tol=0.000000001)   #neural nets classifier
    model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, max_iter=200)
    #model = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', random_state=42)
                    
    # Note: You can adjust the hyperparameters, such as the hidden_layer_sizes, activation function, and solver, based on your specific requirements.

    #model.fit(x_train, y_train)   #network training
    #y_pred = model.predict(x_test)  #prediction
    
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)

    # Calculate metrics
    #accuracy = accuracy_score(y_test, y_pred)
    #precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    #recall = recall_score(y_test, y_pred, average='macro')
    #f1 = f1_score(y_test, y_pred, average='macro')

    # Calculate metrics when cross
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='macro')
    recall = recall_score(y_train, y_pred, average='macro')
    f1 = f1_score(y_train, y_pred, average='macro')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    #print(classification_report(y_test, y_pred, zero_division=0))  #print classifier report
    print(classification_report(y_train, y_pred, zero_division=0))  #print classifier report when cross validation used
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print("Execution Time: {:.2f} seconds".format(execution_time))