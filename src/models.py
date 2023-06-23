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


def run_models(filename):
            
    #read data in csv
    #data = pd.read_csv('resources/dataset.csv', parse_dates=['P4','P7'], date_parser=dateparse)
    data = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")

    
    x = data[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values  #selección de variables de entrada
    y = data['Resultado']  #select target
    test_size = 0.2

    x_train, x_test, y_test, y_train = get_train_and_test(x,y,test_size)
    
    from src.logistic_regression import run_logistic_regression
    
    run_logistic_regression(x,y,test_size,x_train, x_test, y_test, y_train)      
 
    from src.neural_network import run_neural_network

    run_neural_network(x_train, x_test, y_test, y_train)      

    from src.svm import run_svm

    run_svm(x_train, x_test, y_test, y_train)      

    from src.nearest_neighbor import run_nearest_neighbor

    run_nearest_neighbor(x_train, x_test, y_test, y_train) 

    from src.decision_tree import run_decision_tree

    run_decision_tree(x_train, x_test, y_test, y_train)
    
    from src.naive_bayes import run_naive_bayes

    run_naive_bayes(x_train, x_test, y_test, y_train)  