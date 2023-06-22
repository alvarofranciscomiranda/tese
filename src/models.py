import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import csv
sns.set_style("darkgrid")

def read_csv(filename):
    
    with open(filename) as csvfile:
        
        data = []

        reader = csv.DictReader(csvfile)
        
        for row in reader:
            
            data.append(row)
        
        return pd.DataFrame(data)

def get_train_and_test(x,y,size):
    
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size)  #partition in training and test set
    
    return x_train, x_test, y_test, y_train


def run_models():
    
    filename = 'resources/dataset_seconds.csv'
    
    #data = read_csv(filename)
    #print(data)
    
    #read data in csv
    #dateparse = lambda x: pd.datetime.strptime(x, '%H:%M:%S').time()
    #data = pd.read_csv('resources/dataset.csv', parse_dates=['P4','P7'], date_parser=dateparse)
    data = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
    #data = pd.read_excel('dataset.xlsx')
    
    x = data[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values  #selección de variables de entrada
    y = data['Resultado']  #select target
    test_size = 0.2


    from sklearn.model_selection import train_test_split
    x_train, x_test, y_test, y_train = get_train_and_test(x,y,test_size)

    
    #Logistic Regression
    from sklearn.linear_model import LogisticRegressionCV
    valid = True
    while valid:
        try:  
            model5 = LogisticRegressionCV(cv=10, random_state=0, max_iter=5000) #10 fold cross validation
            model5.fit(x_train, y_train)
            y_pred = model5.predict(x_test)
            print("Logistic Regression Precision: ", format(model5.score(x_train, y_train)))
            print("Logistic Regression Accuracy: ", metrics.accuracy_score(y_test, y_pred))
            valid = False
        except ValueError:
            x_train, x_test, y_test, y_train = get_train_and_test(x,y,test_size)
    
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(10), max_iter=5000, alpha=0.0001,
                        solver='adam',activation= 'logistic', random_state=25,
                        tol=0.000000001)   #neural nets classifier

    mlp.fit(x_train, y_train)   #network training
    y_pred = mlp.predict(x_test)  #prediction

    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))  #print classifier report

    #Accuracy NN
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score (y_test, y_pred)
    print("NN Accuracy= ", accuracy)


    #Vector Support Machine . VSM
    from sklearn import svm
    model2 = svm.SVC()
    model2.fit(x_train, y_train)
    y_pred = model2.predict(x_test)
    print("VSM Precision: ", format(model2.score(x_train, y_train)))
    print("VSM Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    #Nearest Neighbor - KNN
    from sklearn.neighbors import KNeighborsClassifier
    model3 = KNeighborsClassifier(n_neighbors=5)
    model3.fit(x_train, y_train)
    y_pred = model3.predict(x_test)
    print("KNN Precision: ", format(model3.score(x_train, y_train)))
    print("KNN Accuracy: ", metrics.accuracy_score(y_test, y_pred))


    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    model4 = DecisionTreeClassifier()
    model4.fit(x_train, y_train)
    y_pred = model4.predict(x_test)
    print("Decision Trees Accuracy: ", metrics.accuracy_score(y_test, y_pred))

