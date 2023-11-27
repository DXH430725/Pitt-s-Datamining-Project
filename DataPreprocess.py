import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno


def get_numerical_summary(df):
    total = df.shape[0]
    
    missing_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    
    missing_percent = {}
    
    for col in missing_columns:
        null_count = df[col].isnull().sum()
        
        per = (null_count / total) * 100
        
        missing_percent[col] = per
        
        print("{} : {} ({}%)".format(col, null_count, round(per, 3)))
    
    return missing_percent

def DataAnalyse(dataset):
    get_numerical_summary(dataset)

    plt.figure(figsize = (15,9))
    msno.matrix(dataset)
    msno.heatmap(dataset, labels = True)
    plt.show()

    print(dataset.info())

def DataProcess(dataset):
    median_ca = dataset['ca'].median()
    dataset['ca'].fillna(median_ca, inplace=True)
    mode_thal = dataset['thal'].mode()[0]
    dataset['thal'].fillna(mode_thal, inplace=True)

    return dataset




if __name__ == '__main__':
    dataset = pd.read_csv("dataset.csv")
    DataAnalyse(dataset)
    DataProcess(dataset)
    DataAnalyse(dataset)