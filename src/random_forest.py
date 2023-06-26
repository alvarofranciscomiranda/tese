from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def run_random_forest(x_train, x_test, y_test, y_train):
    #Decision Tree
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Random Forest Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    
def random_forest_live(df, x_test):
    #Decision Tree
    filename = 'resources/dataset_seconds.csv'
    train_df = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
    
    x = train_df[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values
    y = train_df['Resultado']  #select target   
       
    model = RandomForestClassifier()
    model.fit(x, y)
    y_pred = model.predict(x_test)
    df['RandomForestClassifier'] = y_pred
    df.to_csv('results/RandomForestClassifierResults.csv')