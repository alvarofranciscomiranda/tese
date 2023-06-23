from sklearn import metrics

def run_naive_bayes(x_train, x_test, y_test, y_train):
    #Naive Bayes    
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import precision_score, recall_score, f1_score
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    
    # Calculate accuracy
    print("Naive Bayes Precision: ", format(gnb.score(x_train, y_train)))
     
    #Calculate precision
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)

    # Calculate recall
    #recall = recall_score(y_test, y_pred)

    # Calculate F1 score
    #f1 = f1_score(y_test, y_pred)
    
    print("Naive-Bayes Metrics: ", metrics.accuracy_score(y_test, y_pred))
    #print("Naive-Bayes Accuracy: ", accuracy)
    print("Naive-Bayes Precision: ", precision)
    #print("Naive-Bayes Recall: ", recall)
    #print("Naive-Bayes F1: ", f1)