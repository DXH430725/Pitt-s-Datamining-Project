import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from DataSplit import split
from DataPreprocess import DataProcess,DataAnalyse

from FirstModel import DecisionTree
from SecondModel import KNNClassifier
from ThirdModel import RandomForest

from OptimizeDT import optimize_decision_tree
from OptimizeKNN import optimize_knn
from OptimizeRF import optimize_random_forest

from RFEDecisionTree import train_decision_tree_with_rfe
from SelectKBestDecisionTree import train_decision_tree_with_SelectKBest

from DimensionReductionKNN import optimize_knn_with_reduction

from ModelPerform import evaluate_model

# 'do not disturbe' mode
import warnings                                  
warnings.filterwarnings('ignore')


fig = None


dataset = pd.read_csv("dataset.csv")

dataset = DataProcess(dataset)

#DataAnalyse(dataset)

train_x, test_x, train_y, test_y, x, y = split(dataset)

#Classifier————Decision Tree Regressor
Model_1 = DecisionTree(train_x, train_y)

#Classifier————K-Nearest Neighbors (KNN)
Model_2 = KNNClassifier(train_x, train_y)

#Classifier————RandomForest
Model_3 = RandomForest(train_x, train_y)

#model_names = ["Decision Tree Regressor", "K-Nearest Neighbors (KNN)", "RandomForest"]
#models = [Model_1, Model_2, Model_3]

#for idx, (model, name) in enumerate(zip(models, model_names), 1):
#    print(f"\nEvaluate Model {idx} ({name}):")
#    evaluate_model(model, test_x, test_y)



optimize_decision_tree(train_x, train_y,test_x, test_y, fig)

optimize_knn(train_x, train_y, test_x, test_y, fig)

optimize_random_forest(train_x, train_y,test_x, test_y, fig)

Model_4 = train_decision_tree_with_rfe(x, y, fig)

Model_5 = train_decision_tree_with_SelectKBest(train_x, train_y, 5, fig)

# Use PCA for dimensionality reduction
pca = PCA(n_components=2)  # Adjust the number of components as needed
best_model_pca = optimize_knn_with_reduction(x, y, fig, pca)

# Use t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)  # Adjust parameters as needed
best_model_tsne = optimize_knn_with_reduction(x, y, fig, tsne)