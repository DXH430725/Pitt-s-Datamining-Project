import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier  # Updated import for classifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def optimize_decision_tree(train_x, train_y,test_x, test_y, fig):
    # Define Decision Tree hyperparameter search space
    param_dist = {
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'max_features': ['log2', 'sqrt', None],
    }

    # Create Decision Tree model
    decision_tree = DecisionTreeClassifier(random_state=1)  # Use DecisionTreeClassifier

    # Use RandomizedSearchCV for random search
    random_search = RandomizedSearchCV(
        decision_tree, 
        param_distributions=param_dist, 
        n_iter=10,  # Set the number of search iterations
        scoring='accuracy',  # Use accuracy for classification problems
        cv=5,  # Number of cross-validation folds
        random_state=42
    )

    # Perform the search
    random_search.fit(train_x, train_y)

    # Extract the results from the random search
    results_df = pd.DataFrame(random_search.cv_results_)

    # Output the best parameters
    best_params = random_search.best_params_
    print("Evaluate Decision Tree Regressor Model:")
    print("Best Parameters:", best_params)

    # Output the performance of the best model
    best_model = random_search.best_estimator_
    print("Best Model Performance (accuracy):", random_search.best_score_)
    
    # Make predictions
    predictions = best_model.predict(test_x)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_y, predictions)

    # For multiclass classification, use average='weighted' for precision, recall, and f1
    precision = precision_score(test_y, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_y, predictions, average='weighted', zero_division=0)
    # Calculate AUC
    if best_model.classes_.shape[0] > 2:  # For multiclass classification
        auc = roc_auc_score(pd.get_dummies(test_y), best_model.predict_proba(test_x), multi_class='ovr')
    else:  # For binary classification
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
        plt.title('Decision Tree Hyperparameter Tuning')
        plt.xlabel('Max Depth')
        plt.ylabel('Accuracy')
        plt.show()

        # Visualize the relationship between 'min_samples_split' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_min_samples_split'], results_df['mean_test_score'])
        plt.title('Decision Tree Hyperparameter Tuning')
        plt.xlabel('Min Samples Split')
        plt.ylabel('Accuracy')
        plt.show()

        # Visualize the relationship between 'min_samples_leaf' and performance
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['param_min_samples_leaf'], results_df['mean_test_score'])
        plt.title('Decision Tree Hyperparameter Tuning')
        plt.xlabel('Min Samples Leaf')
        plt.ylabel('Accuracy')
        plt.show()

    return best_model
