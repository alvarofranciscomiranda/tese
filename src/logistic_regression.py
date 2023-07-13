from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
from src.train_and_test import get_train_and_test
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

def run_logistic_regressioncv(x,y,test_size, x_train, x_test, y_test, y_train):
    start_time = time.time()  # Record the start time
    
    #Logistic Regression
    valid = True
    while valid:
        try:  
            model = LogisticRegressionCV(cv=5, multi_class="multinomial", random_state=0, max_iter=5000) #5 fold cross validation
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print("Logistic Regression Precision: ", format(model.score(x_train, y_train)))
            print("Logistic Regression Accuracy: ", metrics.accuracy_score(y_test, y_pred))
            valid = False
        except ValueError:
            x_train, x_test, y_test, y_train = get_train_and_test(x,y,test_size)
            
    end_time = time.time()  # Record the end time
    execution_time = end_time - start_time  # Calculate the execution time
    print("Execution Time: {:.2f} seconds".format(execution_time))
            
            
def run_logistic_regression(x,y,test_size, x_train, x_test, y_test, y_train):
    start_time = time.time()  # Record the start time
    
    model = LogisticRegression()
    
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
    