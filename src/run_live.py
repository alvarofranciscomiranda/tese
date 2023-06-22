import requests
import pandas as pd
from io import BytesIO
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.tree import DecisionTreeClassifier


def run_live():
    
    url = 'https://docs.google.com/spreadsheets/d/1krIL6WcgFVACUQdfvwhcw9VHjJGY4-9tGHXNMeS0DR4'
    
    # Define the scope and credentials
    scope = ['https://spreadsheets.google.com/feeds',
            'https://www.googleapis.com/auth/drive']
    credentials = ServiceAccountCredentials.from_json_keyfile_name('resources/credentials.json', scope)

    # Authenticate with the Google Sheets API
    gc = gspread.authorize(credentials)

    # Open the Google Sheet by its URL or title
    sheet = gc.open_by_url(url)  # Alternatively, use sheet = gc.open('your_sheet_title')

    # Select the desired worksheet by its index or title
    worksheet = sheet.get_worksheet(1)  # Assuming the first worksheet is to be used

    # Get all values from the worksheet
    data = worksheet.get_all_values()

    # Create a DataFrame from the data
    df = pd.DataFrame(data[1:], columns=data[0])  # Assuming the first row contains column headers

    df = df.drop('Questionaire', axis=1)
   
    # Print the DataFrame
    print(df)
    x_test = df[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values  #selección de variables de entrada
    
    #Decision Tree
    filename = 'resources/dataset_seconds.csv'
    train_df = pd.read_csv(filename, encoding = "utf-8", delimiter = ",")
    
    x = train_df[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values  #selección de variables de entrada
    y = train_df['Resultado']  #select target   
       
    model4 = DecisionTreeClassifier()
    model4.fit(x, y)
    y_pred = model4.predict(x_test)
    df['DecisionTreeClassifier'] = y_pred
    #df['NeuralNetwork'] = y_pred
    df.to_csv('results/DecisionTreeClassifierResults.csv')
    #print("Decision Trees Accuracy: ", metrics.accuracy_score(y_test, y_pred))

    #r = pd.read_csv(url)
    #r = requests.get(url)
    #print(r.content)
    #data = r.content
    
    #x = r[['P1','P2','P3','P4','P4.1','P5','P5.1','P5.2','P6','P7','P8','P9','P10','P11','P12','P13','P14','P15','P16','P17','P18','P19','P20','P21','P22','P23','P24','P25','P26','P27','P28','P29','P30','P31','P32','P33','P34','P35','P36','P37','P38','P39','P40']].values  #selección de variables de entrada
    #y = r['Resultado']  #select target
    
    #df = pd.read_csv(BytesIO(r.content))
    