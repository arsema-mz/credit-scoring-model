import joblib

def load_model(model_path):
    """Load the trained model from a given file path."""
    return joblib.load(model_path)

def print_model_feature_names(model_path):
    """Print the feature names expected by the model."""
    model = load_model(model_path)

    # If it's a linear model (like logistic regression), you can access feature names through the model's attributes.
    # Adjust this part if your model is not a linear model or if it doesn't have feature names.
    if hasattr(model, 'feature_names_in_'):
        print("Feature names used during training:")
        print(model.feature_names_in_)
    else:
        print("The model does not have an attribute for feature names.")

if __name__ == "__main__":
    # Specify the path to your model file
    model_path = "models/logreg_best.pkl"  # Change this to your model path
    print_model_feature_names(model_path)