from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
from src.train_and_test import get_train_and_test

def run_logistic_regression(x,y,test_size, x_train, x_test, y_test, y_train):
    #Logistic Regression
    valid = True
    while valid:
        try:  
            model = LogisticRegressionCV(cv=10, random_state=0, max_iter=5000) #10 fold cross validation
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            print("Logistic Regression Precision: ", format(model.score(x_train, y_train)))
            print("Logistic Regression Accuracy: ", metrics.accuracy_score(y_test, y_pred))
            valid = False
        except ValueError:
            x_train, x_test, y_test, y_train = get_train_and_test(x,y,test_size)
    