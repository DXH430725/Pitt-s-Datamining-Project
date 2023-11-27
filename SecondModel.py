import pandas as pd
from sklearn.svm import SVC


def SupportVector(train_x, train_y):

    x = train_x
    y = train_y


    predict_model = SVC(random_state=1)
    predict_model.fit(x, y)
    return predict_model
