from sklearn import metrics
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
import time


#Support Vector Machine
def run_svm(x_train, x_test, y_test, y_train):
    start_time = time.time()  # Record the start time

    model = svm.SVC()
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    
    # Compute additional metrics using cross_val_predict
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)  # Get predicted labels
    
    # Calculate metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='macro')
    recall = recall_score(y_train, y_pred, average='macro')
    f1 = f1_score(y_train, y_pred, average='macro')

    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Print the cross-validation scores
    print(f'Cross-Validation Scores: {cv_scores}')
    print(f'Average Score: {cv_scores.mean():.4f}')
    
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print("Execution Time: {:.2f} seconds".format(execution_time))