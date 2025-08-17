import shap
import matplotlib.pyplot as plt
import joblib
import pandas as pd


def load_model(model_path: str):
    """
    Load a trained model from disk.
    """
    return joblib.load(model_path)


def explain_model(model, X, output_dir="reports/figures", sample_size=200):
    """
    Generate SHAP explanations for a given model and dataset.
    
    Args:
        model: Trained ML model (sklearn, xgboost, etc.)
        X (pd.DataFrame): Input features used for predictions.
        output_dir (str): Directory where SHAP plots will be saved.
        sample_size (int): Subset size for faster computation.
    """
    # Subsample for speed
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42)
    else:
        X_sample = X

    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    # === Global Importance ===
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(f"{output_dir}/shap_summary.png", bbox_inches="tight")
    plt.close()

    # === Feature Dependence for Top Feature ===
    top_feature = X_sample.columns[abs(shap_values.values).mean(0).argmax()]
    plt.figure()
    shap.dependence_plot(top_feature, shap_values.values, X_sample, show=False)
    plt.savefig(f"{output_dir}/shap_dependence.png", bbox_inches="tight")
    plt.close()

    print(f"[✓] SHAP plots saved to {output_dir}")


def explain_single_prediction(model, X, index=0):
    """
    Generate SHAP force plot for a single prediction.
    
    Args:
        model: Trained ML model.
        X (pd.DataFrame): Input features.
        index (int): Row index for explanation.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    return shap.force_plot(explainer.expected_value, shap_values[index], X.iloc[index, :])
