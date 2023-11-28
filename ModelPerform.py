from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def evaluate_model(model, test_x, test_y):
    # Make predictions
    predictions = model.predict(test_x)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_y, predictions)
    precision = precision_score(test_y, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_y, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_y, predictions, average='weighted', zero_division=0)

    # Print other metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Calculate AUC
    if model.classes_.shape[0] > 2:  # For multiclass classification
        auc = roc_auc_score(pd.get_dummies(test_y), model.predict_proba(test_x), multi_class='ovr')
    else:  # For binary classification
        auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])

    print(f"AUC: {auc:.4f}")
    print("")
