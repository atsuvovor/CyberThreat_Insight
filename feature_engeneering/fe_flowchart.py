#!/usr/bin/env python3
# --------------------------------------------------------------
# Cyber Threat Insight - Feature Engineering Workflow Flowchart
# Executable Python version
# Author: Atsu Vovor
# --------------------------------------------------------------

from graphviz import Digraph
import os

def generate_feature_engineering_flowchart(output_name="features_engineering_flowchart"):
    """
    Generates the Cyber Threat Insight Feature Engineering flowchart as a PNG file.
    Returns the full path to the generated file.
    """

    dot = Digraph(
        "Cyber Threat Insight - Feature Engineering Workflow",
        format="png"
    )

    # ---------------------------
    # Feature Engineering Phases
    # ---------------------------
    dot.node('Start', 'Start')

    dot.node('DataInj',
             'Data Injection\n(Cholesky-Based Perturbation)',
             shape='box', style='filled', fillcolor='lightblue')

    dot.node('Scaling',
             'Feature Normalization & Scaling\n(Min-Max, Z-score)',
             shape='box', style='filled', fillcolor='lightgray')

    dot.node('CorrHeat',
             'Correlation Heatmap Analysis\n(Pearson/Spearman)',
             shape='box', style='filled', fillcolor='orange')

    dot.node('FeatImp',
             'Feature Importance\n(Random Forest)',
             shape='box', style='filled', fillcolor='gold')

    dot.node('SHAP',
             'Model Explainability\n(SHAP Values)',
             shape='box', style='filled', fillcolor='lightgreen')

    dot.node('PCA',
             'PCA & Variance Explained\n(Scree Plot)',
             shape='box', style='filled', fillcolor='plum')

    dot.node('Augment',
             'Data Augmentation\n(SMOTE, GAN)',
             shape='box', style='filled', fillcolor='lightpink')

    dot.node('End',
             'Feature Set Ready for Modeling',
             shape='ellipse', style='filled', fillcolor='lightyellow')

    # ---------------------------
    # Arrows (Workflow Steps)
    # ---------------------------
    dot.edge('Start', 'DataInj')
    dot.edge('DataInj', 'Scaling')
    dot.edge('Scaling', 'CorrHeat')
    dot.edge('CorrHeat', 'FeatImp')
    dot.edge('FeatImp', 'SHAP')
    dot.edge('SHAP', 'PCA')
    dot.edge('PCA', 'Augment')
    dot.edge('Augment', 'End')

    # ---------------------------
    # Render to PNG
    # ---------------------------
    filepath = dot.render(output_name, format="png", cleanup=False)
    display(Image(filename=filepath)
    return filepath


# --------------------------------------------------------------
# Execute as standalone script
# --------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Feature Engineering Flowchart...")

    output_file = generate_feature_engineering_flowchart()
    
    abs_path = os.path.abspath(output_file)

    display(Image(filename=abs_path)
            
    print("\nFlowchart generated successfully!")
    print(f"Saved to: {abs_path}")
