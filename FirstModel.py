import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Updated import for classifier

def DecisionTree(train_x, train_y):
    x = train_x
    y = train_y

    # Use DecisionTreeClassifier instead of DecisionTreeRegressor
    predict_model = DecisionTreeClassifier(random_state=1)
    predict_model.fit(x, y)
    return predict_model
