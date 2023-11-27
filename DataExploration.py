import pandas as pd

dataset = pd.read_csv("dataset.csv")

result = dataset.describe()

for column in dataset.columns:
    print(f"For {column}, the count = {dataset[column].count()}; missing count = {303-dataset[column].count()}; mean = {dataset[column].mean()}; min = {dataset[column].min()}; max = {dataset[column].max()};")
