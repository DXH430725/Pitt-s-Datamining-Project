import pandas as pd

from DataSplit import split
from DataPreprocess import DataProcess,DataAnalyse

from FirstModel import DecisionTree
from SecondModel import SupportVector
from ThirdModel import RandomForest

from OptimizeDT import optimize_decision_tree
from OptimizeSVM import optimize_svm
from OptimizeRF import optimize_random_forest

from ModelPerform import evaluate_model


dataset = pd.read_csv("dataset.csv")

DataProcess(dataset)

#DataAnalyse(dataset)

train_x, test_x, train_y, test_y = split(dataset)

#Classifier————Decision Tree Regressor
Model_1 = DecisionTree(train_x, train_y)

#Classifier————Support Vector Machines (SVM)
Model_2 = SupportVector(train_x, train_y)

#Classifier————Random Forest
Model_3 = RandomForest(train_x, train_y)

model_names = ["Decision Tree Regressor", "Support Vector Machines (SVM)", "Random Forest"]
evaluate_model(Model_1, Model_2, Model_3, test_x, test_y, model_names)

#optimize_decision_tree(train_x, train_y)

#optimize_svm(train_x, train_y)

optimize_random_forest(train_x, train_y)