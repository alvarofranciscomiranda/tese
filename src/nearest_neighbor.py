from sklearn import metrics

def run_nearest_neighbor(x_train, x_test, y_test, y_train):

#Nearest Neighbor - KNN
    from sklearn.neighbors import KNeighborsClassifier
    model3 = KNeighborsClassifier(n_neighbors=5)
    model3.fit(x_train, y_train)
    y_pred = model3.predict(x_test)
    print("KNN Precision: ", format(model3.score(x_train, y_train)))
    print("KNN Accuracy: ", metrics.accuracy_score(y_test, y_pred))