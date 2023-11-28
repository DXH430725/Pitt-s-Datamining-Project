import pandas as pd
from sklearn.model_selection import train_test_split

def split(dataset, features=None, test_size=50):

    if features is None:
        features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    x = dataset[features]
    y = dataset.num

    train_x = x.iloc[test_size:]
    train_y = y.iloc[test_size:]
    test_x = x.iloc[:test_size]
    test_y = y.iloc[:test_size]

    return train_x, test_x, train_y, test_y, x, y

if __name__ == "__main__":
    train_x, test_x, train_y, test_y = split()
    print("Train X:")
    print(train_x)