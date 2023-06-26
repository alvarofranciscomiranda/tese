from sklearn.model_selection import train_test_split

def get_train_and_test(x,y,size):
    

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = size)  #partition in training and test set
    
    return x_train, x_test, y_test, y_train
