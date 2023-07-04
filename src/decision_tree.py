from sklearn import metrics
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import export_graphviz
import graphviz

def run_decision_tree(x_train, x_test, y_test, y_train):
    #Decision Tree
    model = DecisionTreeClassifier()
    
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
        
def decision_tree_live(df, x_test):
    #Decision Tree
    filename = 'resources/dataset_seconds.csv'
    train_df = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
    
    x = train_df[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values
    y = train_df['Resultado']  #select target   
       
    model = DecisionTreeClassifier()
    model.fit(x, y)
    y_pred = model.predict(x_test)
    df['DecisionTreeClassifier'] = y_pred
    df.to_csv('results/DecisionTreeClassifierResults.csv')