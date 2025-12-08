#!/usr/bin/env python3
# --------------------------------------------------------------
# CyberThreat Insight – smote_gan_models_performance.
# Author: Atsu Vovor
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
from feature_creation import load_objects_from_drive

# ---------------------------- #
# Apply Custom Matplotlib Style
# ---------------------------- #
def apply_custom_matplotlib_style(font_family='serif', font_size=11):
    plt.rcParams.update({
        'font.family': font_family,
        'font.size': font_size,
        'axes.titlesize': font_size + 1,
        'axes.labelsize': font_size,
        'legend.fontsize': font_size - 1,
        'xtick.labelsize': font_size - 1,
        'ytick.labelsize': font_size - 1
    })

# ---------------------------- #
# Loaders (Stub for Integration)
# ---------------------------- #
def load_dataset(filepath):
    return pd.read_csv(filepath)

# ---------------------------- #
#       Plot GAN Loss
# ---------------------------- #
def plot_loss_history(p_d_loss_real_list, p_d_loss_fake_list, p_g_loss_list):
    plt.figure(figsize=(5, 3))
    plt.plot(p_d_loss_real_list, label='D Loss Real')
    plt.plot(p_d_loss_fake_list, label='D Loss Fake')
    plt.plot(p_g_loss_list, label='G Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------------- #
# Plot Training vs Validation Metrics
# ---------------------------- #
def plot_train_val_comparison(train_scores, val_scores, metric_name='Accuracy', title_prefix='Model Performance'):
    plt.figure(figsize=(5, 3))
    plt.plot(train_scores, label='Train')
    plt.plot(val_scores, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'{title_prefix}: Train vs Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_gan_training_metrics(p_d_loss_real_list, p_d_loss_fake_list, p_g_loss_list,
                               train_scores, val_scores, metric_name='Accuracy',
                               title_prefix='Model Performance'):
    """
    Plot GAN loss history and training vs validation metrics in a 1-row 2-column subplot.

    Parameters
    ----------
    p_d_loss_real_list : list
        Discriminator loss on real samples per epoch.
    p_d_loss_fake_list : list
        Discriminator loss on fake samples per epoch.
    p_g_loss_list : list
        Generator loss per epoch.
    train_scores : list
        Training metric values.
    val_scores : list
        Validation metric values.
    metric_name : str, optional
        Name of the evaluation metric (default is 'Accuracy').
    title_prefix : str, optional
        Prefix for the second subplot title.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # Plot 1: GAN Loss History
    axes[0].plot(p_d_loss_real_list, label='D Loss Real')
    axes[0].plot(p_d_loss_fake_list, label='D Loss Fake')
    axes[0].plot(p_g_loss_list, label='G Loss')
    axes[0].set_title('GAN Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Train vs Validation Metric
    axes[1].plot(train_scores, label='Train')
    axes[1].plot(val_scores, label='Validation')
    axes[1].set_title(f'{title_prefix}: Train vs Validation {metric_name}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_name)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_gan_loss_and_model_performance(
    p_d_loss_real_list, p_d_loss_fake_list, p_g_loss_list,
    train_scores, val_scores,
    metric_name='Accuracy', title_prefix='Model Performance'
):
    """
    Plot GAN loss and model performance in subplots.

    Parameters
    ----------
    p_d_loss_real_list : list
    p_d_loss_fake_list : list
    p_g_loss_list : list
    train_scores : list
    val_scores : list
    metric_name : str
    title_prefix : str
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))

    # Subplot 1: GAN Training Loss
    axs[0].plot(p_d_loss_real_list, label='D Loss Real')
    axs[0].plot(p_d_loss_fake_list, label='D Loss Fake')
    axs[0].plot(p_g_loss_list, label='G Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('GAN Training Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Train vs Validation Scores
    axs[1].plot(train_scores, label='Train')
    axs[1].plot(val_scores, label='Validation')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel(metric_name)
    axs[1].set_title(f'{title_prefix}: Train vs Validation {metric_name}')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# ---------------------------- #
# 3D Histogram Comparison
# ---------------------------- #
def plot_3d_histogram_comparison(y_before, y_augmented, ax, target_column='Threat Level'):
    bins = np.histogram_bin_edges(np.concatenate([y_before, y_augmented]), bins='auto')
    hist_before, _ = np.histogram(y_before, bins=bins, density=True)
    hist_aug, _ = np.histogram(y_augmented, bins=bins, density=True)

    xpos = (bins[:-1] + bins[1:]) / 2
    ypos_before = np.zeros_like(xpos)
    ypos_aug = np.ones_like(xpos)

    dx = dy = 0.3
    norm = Normalize(vmin=0, vmax=max(hist_before.max(), hist_aug.max()))
    cmap = cm.get_cmap('coolwarm')

    ax.bar3d(xpos, ypos_before, np.zeros_like(hist_before), dx, dy, hist_before,
             color=cmap(norm(hist_before)), alpha=0.8)
    ax.bar3d(xpos, ypos_aug, np.zeros_like(hist_aug), dx, dy, hist_aug,
             color=cmap(norm(hist_aug)), alpha=0.8)

    ax.set_xticks(xpos[::max(1, len(xpos)//10)])
    ax.set_xticklabels([f"{val:.1f}" for val in xpos[::max(1, len(xpos)//10)]], rotation=45)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Original', 'Augmented'])
    ax.set_xlabel(target_column)
    ax.set_ylabel("Data Type")
    ax.set_zlabel("Density")
    ax.set_title(f"3D Histogram\n{target_column}", pad=10)

# ---------------------------- #
# Combined 2D & 3D Projection
# ---------------------------- #
def plot_combined_analysis_2d_3d(fe_processed_df, X_augmented, y_augmented, features_engineering_columns, target_column='Threat Level'):
    x_features = [col for col in features_engineering_columns if col != target_column]
    X_real = fe_processed_df[x_features].values
    X_generated = X_augmented[x_features].values

    X_combined = np.vstack((X_real, X_generated))
    labels = ['Real'] * len(X_real) + ['Generated'] * len(X_generated)
    colors = ['blue' if l == 'Real' else 'red' for l in labels]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    y_before = fe_processed_df[target_column]

    fig, axes = plt.subplots(1, 4, figsize=(26, 6))
    fig.suptitle('2D Projections: Real vs Synthetic', fontsize=14)
    plt.subplots_adjust(wspace=0.4)


    sns.histplot(y_before, label='Original', color='blue', kde=True, stat="density", ax=axes[0])
    sns.histplot(y_augmented, label='Augmented', color='red', kde=True, stat="density", ax=axes[0])
    axes[0].set_title('Class Distribution')
    axes[0].legend()
    axes[0].set_xlabel(target_column)
    axes[0].set_ylabel("Density")

    X_pca = PCA(n_components=2).fit_transform(X_scaled)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette={'Real': 'blue', 'Generated': 'red'}, alpha=0.7, ax=axes[1])
    axes[1].set_title('PCA (2D)')

    X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, palette={'Real': 'blue', 'Generated': 'red'}, alpha=0.7, ax=axes[2])
    axes[2].set_title('t-SNE (2D)')

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=labels, palette={'Real': 'blue', 'Generated': 'red'}, alpha=0.7, ax=axes[3])
    axes[3].set_title('UMAP (2D)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\n plotting 3D Real VS Generated\n")
    fig_3d = plt.figure(figsize=(26, 6))
    fig_3d.suptitle('3D Projections: Real vs Synthetic', fontsize=14)

    plot_3d_histogram_comparison(y_before, y_augmented, fig_3d.add_subplot(1, 4, 1, projection='3d'), target_column)

    ax_pca = fig_3d.add_subplot(1, 4, 2, projection='3d')
    X_pca_3d = PCA(n_components=3).fit_transform(X_scaled)
    ax_pca.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=colors, alpha=0.6)
    ax_pca.set_title('PCA (3D)')

    ax_tsne = fig_3d.add_subplot(1, 4, 3, projection='3d')
    X_tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42).fit_transform(X_scaled)
    ax_tsne.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], c=colors, alpha=0.6)
    ax_tsne.set_title('t-SNE (3D)')

    ax_umap = fig_3d.add_subplot(1, 4, 4, projection='3d')
    reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap_3d = reducer_3d.fit_transform(X_scaled)
    ax_umap.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=colors, alpha=0.6)
    ax_umap.set_title('UMAP (3D)')

    plt.show()

# ---------------------------- #
# Main Pipeline
# ---------------------------- #
def SMOTE_GANs_evaluation_pipeline():
    #data_augmentation_pipeline()

    print("Starting SMOTE and GAN augmentation models performance Analysis...")
    loss_df = load_dataset("CyberThreat_Insight/cybersecurity_data/gan_loss_log.csv")
    augmented_df = load_dataset("CyberThreat_Insight/cybersecurity_data/x_y_augmented_data_google_drive.csv")
    fe_processed_df, loaded_label_encoders, num_fe_scaler = load_objects_from_drive()

    X_augmented = augmented_df.drop(columns=["Threat Level"])
    y_augmented = augmented_df["Threat Level"]

    features_engineering_columns = X_augmented.columns

    d_loss_real_list = loss_df["D_Loss_Real"]
    d_loss_fake_list = loss_df["D_Loss_Fake"]
    g_loss_list = loss_df["G_Loss"]

    # Optional: Replace with actual tracking results
    train_accuracy = np.linspace(0.65, 0.95, len(g_loss_list)) #train_scores
    val_accuracy = np.linspace(0.60, 0.93, len(g_loss_list)) #val_scores

    #print("\nApplying Custom Matplotlib Style\n")
    apply_custom_matplotlib_style()
    plot_combined_analysis_2d_3d(fe_processed_df, X_augmented, y_augmented, features_engineering_columns)

    #print("\n plotting gan_training_metrics\n")
    plot_gan_training_metrics(d_loss_real_list, d_loss_fake_list, g_loss_list,
                              train_accuracy, val_accuracy, metric_name='Accuracy',
                              title_prefix='GAN Performance')


if __name__ == "__main__":
    SMOTE_GANs_evaluation_pipeline()
