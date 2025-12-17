from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import os

def evaluate_model(name, y_true, y_pred):
    """
    Computes standardized evaluation metrics for a given model.
    Focuses on Critical threats (Class 3).
    """
    report = classification_report(y_true, y_pred, output_dict=True)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 2),
        "F1_Score_Class_3": round(report["3"]["f1-score"], 2),
        "Recall_Class_3": round(report["3"]["recall"], 2)
    }


def export_evaluation_results(results, output_dir):
    """
    Exports evaluation metrics to CSV and generates a README-ready table.
    """
    df = pd.DataFrame(results)

    csv_path = os.path.join(output_dir, "model_evaluation_summary.csv")
    df.to_csv(csv_path, index=False)

    readme_table = df.to_markdown(index=False)

    return csv_path, readme_table
