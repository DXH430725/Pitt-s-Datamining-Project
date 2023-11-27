import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def RandomForest(train_x, train_y):

    x = train_x
    y = train_y


    predict_model = RandomForestClassifier(random_state=1)
    predict_model.fit(x, y)
    return predict_model
