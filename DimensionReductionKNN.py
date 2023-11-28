import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Function to optimize KNN with dimensionality reduction
def optimize_knn_with_reduction(x, y, fig, dimensionality_reduction_algorithm):
    # Apply dimensionality reduction
    x_reduced = dimensionality_reduction_algorithm.fit_transform(x)

    print(f"For DecisionTreeClassifier with {dimensionality_reduction_algorithm}:")

    # Convert the reduced array back to a Pandas DataFrame
    x_reduced_df = pd.DataFrame(x_reduced, columns=[f'component_{i}' for i in range(x_reduced.shape[1])])

    # Select the first 50 rows as the test set
    test_x = x_reduced_df.iloc[:50, :]
    test_y = y.iloc[:50]

    # Select the remaining rows as the training set
    train_x = x_reduced_df.iloc[50:, :]
    train_y = y.iloc[50:]

    # Define KNN hyperparameter search space
    param_dist = {
        'n_neighbors': randint(1, 20),
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan distance (L1), 2 for Euclidean distance (L2)
    }

    # Create KNN model
    knn_model = KNeighborsClassifier()  # Use KNeighborsClassifier

    # Use RandomizedSearchCV for random search
    random_search = RandomizedSearchCV(
        knn_model,
        param_distributions=param_dist,
        n_iter=10,  # Set the number of search iterations
        scoring='accuracy',  # Use accuracy for classification problems
        cv=5,  # Number of cross-validation folds
        random_state=42
    )

    # Perform the search on reduced features
    random_search.fit(train_x, train_y)

    # Extract the results from the random search
    results_df = pd.DataFrame(random_search.cv_results_)

    # Output the best parameters
    best_params = random_search.best_params_
    print("Best Parameters:", best_params)

    # Output the performance of the best model
    best_model = random_search.best_estimator_
    print("Best Model Performance (accuracy):", random_search.best_score_)
    print("")

    if fig:
        # Visualize the relationship between 'n_neighbors' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_n_neighbors'], results_df['mean_test_score'])
        plt.title('KNN Hyperparameter Tuning with Dimensionality Reduction')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        plt.show()

    # Make predictions on the test set
    predictions = best_model.predict(test_x)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_y, predictions, average='weighted', zero_division=0)

    # Calculate AUC
    if best_model.classes_.shape[0] > 2:  # For multiclass classification
        auc = roc_auc_score(pd.get_dummies(test_y), best_model.predict_proba(test_x), multi_class='ovr')
    else:  # For binary classification
        auc = roc_auc_score(test_y, best_model.predict_proba(test_x)[:, 1])

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("")

    if plt:
        # Visualize the relationship between 'n_neighbors' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_n_neighbors'], results_df['mean_test_score'])
        plt.title('KNN Hyperparameter Tuning with Dimensionality Reduction')
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Accuracy')
        plt.show()

    return best_model