from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_model(Model_1, Model_2, Model_3, test_x, test_y, model_names):

    # Evaluate models
    models = [Model_1, Model_2, Model_3]

    for idx, (model, name) in enumerate(zip(models, model_names), 1):
        print(f"\nEvaluate Model {idx} ({name}):")

        # Make predictions
        predictions = model.predict(test_x)

        # Calculate evaluation metrics
        accuracy = accuracy_score(test_y, predictions)

        # For multiclass classification, use average='weighted' for precision, recall, and f1
        precision = precision_score(test_y, predictions, average='weighted', zero_division=0)
        recall = recall_score(test_y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(test_y, predictions, average='weighted', zero_division=0)

        # Print other metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
