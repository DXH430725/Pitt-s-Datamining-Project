import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def DecisionTree(train_x, train_y):

    x = train_x
    y = train_y


    predict_model = DecisionTreeRegressor(random_state=1)
    predict_model.fit(x, y)
    return predict_model

