from sklearn import metrics

def run_neural_network(x_train, x_test, y_test, y_train):
    
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=5000, alpha=0.0001,
                        solver='adam',activation= 'logistic', random_state=10,
                        tol=0.000000001)   #neural nets classifier

    mlp.fit(x_train, y_train)   #network training
    y_pred = mlp.predict(x_test)  #prediction

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))  #print classifier report

    #Accuracy NN
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score (y_test, y_pred)
    print("NN Accuracy= ", accuracy)