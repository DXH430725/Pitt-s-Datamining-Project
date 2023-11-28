import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def KNNClassifier(train_x, train_y):

    x = train_x
    y = train_y

    # Create K-Nearest Neighbors classifier
    knn_model = KNeighborsClassifier()
    
    # Train the KNN model
    knn_model.fit(x, y)

    return knn_model
