from sklearn import metrics

#Support Vector Machine
def run_svm(x_train, x_test, y_test, y_train):

    from sklearn import svm
    
    model2 = svm.SVC()
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    print("VSM Precision: ", format(model2.score(x_train, y_train)))
    print("VSM Accuracy: ", metrics.accuracy_score(y_test, y_pred))