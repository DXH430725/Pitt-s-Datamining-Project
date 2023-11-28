import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from scipy.stats import randint

def train_decision_tree_with_SelectKBest(x, y, k, fig):
    # Create a Decision Tree classifier
    decision_tree = DecisionTreeClassifier(random_state=1)

    # Feature selection using SelectKBest with chi-squared
    select_k_best = SelectKBest(chi2, k=k)
    x_selected = select_k_best.fit_transform(x, y)

    # Print selected features
    selected_features = x.columns[select_k_best.get_support()]
    print("Selected Features:", selected_features)
    print("For DecisionTreeClassifier with SelectKBest feature selection:")
    
    # Split the data into training and testing sets
    x_split, test_x, y_split, test_y = train_test_split(x_selected, y, test_size=0.165, random_state=42)

    # Define Decision Tree hyperparameter search space
    param_dist = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['auto', 'sqrt', 'log2', None]
    }

    # Use RandomizedSearchCV for random search
    random_search = RandomizedSearchCV(
        decision_tree,
        param_distributions=param_dist,
        n_iter=10,
        scoring='accuracy',  # Use accuracy for classification problems
        cv=5,
        random_state=42
    )

    # Perform the search
    random_search.fit(x_split, y_split)

    # Extract the results from the random search
    results_df = pd.DataFrame(random_search.cv_results_)

    # Output the best parameters
    best_params = random_search.best_params_
    print("Best Parameters:", best_params)

    # Output the performance of the best model
    best_model = random_search.best_estimator_
    print("Best Model Performance (accuracy):", random_search.best_score_)

    # Make predictions
    predictions = best_model.predict(test_x)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_y, predictions, average='weighted', zero_division=0)
    
    # Calculate AUC
    if best_model.classes_.shape[0] > 2:
        auc = roc_auc_score(pd.get_dummies(test_y), best_model.predict_proba(test_x), multi_class='ovr')
    else:
        auc = roc_auc_score(test_y, best_model.predict_proba(test_x)[:, 1])

    # Print other metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("")

    if fig:
        # Visualize the relationship between 'max_depth' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_max_depth'], results_df['mean_test_score'])
        plt.title('Decision Tree Hyperparameter Tuning with Feature Selection')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.show()

        # Visualize the relationship between 'min_samples_split' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_min_samples_split'], results_df['mean_test_score'])
        plt.title('Decision Tree Hyperparameter Tuning with Feature Selection')
        plt.xlabel('Min Samples Split')
        plt.ylabel('Accuracy')
        plt.show()

        # Visualize the relationship between 'min_samples_leaf' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_min_samples_leaf'], results_df['mean_test_score'])
        plt.title('Decision Tree Hyperparameter Tuning with Feature Selection')
        plt.xlabel('Min Samples Leaf')
        plt.ylabel('Accuracy')
        plt.show()

    return best_model
