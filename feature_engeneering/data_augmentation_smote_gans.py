#!/usr/bin/env python3
# --------------------------------------------------------------
# CyberThreat Insight – Data Augmentation (SMOTE + GAN)
# 
# Author: Atsu Vovor
# --------------------------------------------------------------

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from IPython.display import display
from feature_creation import load_objects_from_drive
from smote_gan_models_performance import plot_combined_analysis_2d_3d


# ------------------------- SMOTE: Handle class imbalance -------------------------
def balance_data_with_smote(df, target_column="Threat Level"):
    """
    Apply SMOTE to balance minority classes in the dataset.
    Returns resampled feature set and target labels.
    """
    print("Balancing data with SMOTE...")
    X = df.drop(columns=[target_column])
    y = df[target_column]
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# ------------------- Build Generator and Discriminator for GAN -------------------
def build_gan(latent_dim, n_outputs):
    """
    Build and compile a basic GAN architecture with:
    - A generator that outputs synthetic samples
    - A discriminator that classifies real vs synthetic samples
    Returns both models.
    """
    def build_generator():
        model = tf.keras.Sequential([
            layers.Dense(128, activation="relu", input_dim=latent_dim),
            layers.Dense(256, activation="relu"),
            layers.Dense(n_outputs, activation="tanh")
        ])
        return model

    def build_discriminator():
        model = tf.keras.Sequential([
            layers.Dense(256, activation="relu", input_shape=(n_outputs,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])
        return model

    generator = build_generator()
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy')
    return generator, discriminator

# -------------------------- Train GAN with Logging --------------------------
def train_gan(generator, discriminator, X_real, latent_dim, epochs=1000, batch_size=64,
              plot_loss=False, early_stop_patience=50, output_dir="/content/drive/My Drive/Cybersecurity Data/"):
    """
    Train GAN using real synthetic data with optional logging, early stopping, and visualization.
    Tracks generator and discriminator losses and saves logs and plots to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    d_loss_real_list = []
    d_loss_fake_list = []
    g_loss_list = []

    best_g_loss = np.inf
    patience_counter = 0

    for epoch in tqdm(range(epochs), desc="Training GAN"):
        # Generate fake samples
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_data = generator.predict(noise, verbose=0)

        # Sample real data
        idx = np.random.randint(0, X_real.shape[0], batch_size)
        real_data = X_real.iloc[idx].values

        # Labels for real and fake data
        real_labels = np.ones((batch_size, 1))
        fake_labels = np.zeros((batch_size, 1))

        # Train discriminator on real and fake data
        d_loss_real = discriminator.train_on_batch(real_data, real_labels)
        d_loss_fake = discriminator.train_on_batch(gen_data, fake_labels)

        # Train generator to fool the discriminator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = discriminator.train_on_batch(generator.predict(noise, verbose=0), real_labels)

        # Log losses
        d_loss_real_list.append(d_loss_real)
        d_loss_fake_list.append(d_loss_fake)
        g_loss_list.append(g_loss)

        # Early stopping logic for generator loss
        if g_loss < best_g_loss:
            best_g_loss = g_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch} - No improvement in G loss for {early_stop_patience} epochs.")
                break

    # Save loss plot and CSV log
    plt.savefig(os.path.join(output_dir, "gan_loss_plot.png"))
    plt.close()

    loss_df = pd.DataFrame({
        "D_Loss_Real": d_loss_real_list,
        "D_Loss_Fake": d_loss_fake_list,
        "G_Loss": g_loss_list
    })
    loss_df.to_csv(os.path.join(output_dir, "gan_loss_log.csv"), index=False)

    return generator, d_loss_real_list, d_loss_fake_list, g_loss_list

# -------------------------- Generate synthetic samples --------------------------
def generate_synthetic_data(generator, n_samples, latent_dim, columns):
    """
    Generate synthetic samples using a trained GAN generator.
    Returns a DataFrame with the same feature columns.
    """
    noise = np.random.normal(0, 1, (n_samples, latent_dim))
    synthetic_data = generator.predict(noise, verbose=0)
    return pd.DataFrame(synthetic_data, columns=columns)

# -------------------------- Combine real + synthetic --------------------------
def augment_data(X_resampled, y_resampled, synthetic_data):
    """
    Combine real (SMOTE) and synthetic (GAN) data.
    Returns the concatenated feature set and target labels.
    """
    X_augmented = pd.concat([X_resampled, synthetic_data], axis=0)
    y_augmented = pd.concat([y_resampled, pd.Series(np.repeat(y_resampled.mode()[0], synthetic_data.shape[0]))])
    return X_augmented, y_augmented

# -------------------------- Concatenate into a final dataframe --------------------------
def concatenate_data_along_columns(X_augmented, y_augmented):
    """
    Merge features and labels into a single DataFrame.
    Returns the augmented DataFrame with a labeled target column.
    """
    augmented_df = pd.concat([X_augmented.copy(), y_augmented.copy()], axis=1)
    return augmented_df.rename(columns={0: "Threat Level"})

# -------------------------- Load/save utilities (assumed implemented) --------------------------
def save_dataframe_to_google_drive(df, path):
    """
    Utility function to save DataFrame to Google Drive path as CSV.
    """
    df.to_csv(path, index=False)

# -------------------------- Main pipeline function --------------------------
def data_augmentation_pipeline(file_path="", lead_save_true_false = True):
    """
    Main function that executes the entire data augmentation pipeline:
    1. Load data
    2. Apply SMOTE
    3. Build and train GAN
    4. Generate synthetic samples
    5. Combine with real samples
    6. Save final augmented dataset and loss logs
    """

    print("Applying SMOTE ... Building and training GAN")
    x_y_augmented_data_google_drive = "CyberThreat_Insight/cybersecurity_data/x_y_augmented_data_google_drive.csv"
    loss_data_google_drive = "CyberThreat_Insight/cybersecurity_data/loss_data_google_drive.csv"

    # Load preprocessed data from Google Drive
    if lead_save_true_false:
        print("Loading objects from Google Drive...")
        fe_processed_df, cat_cols_label_encoders, num_fe_scaler = load_objects_from_drive()
    else:
        fe_processed_df, cat_cols_label_encoders, num_fe_scaler = features_engineering_pipeline(file_path, analysis_true_false = False)

    if fe_processed_df is not None and cat_cols_label_encoders is not None:
        print("Data loaded from Google Drive.")
        processed_num_df = fe_processed_df.copy()
    else:
        print("Failed to load objects from Google Drive.")
        return None, None

    # Step 1: Balance data using SMOTE
    X_resampled, y_resampled = balance_data_with_smote(processed_num_df)

    # Step 2: Build GAN architecture
    latent_dim = 100
    n_outputs = X_resampled.shape[1]
    generator, discriminator = build_gan(latent_dim, n_outputs)

    # Step 3: Train GAN with logging and early stopping
    generator, d_loss_real_list, d_loss_fake_list, g_loss_list = train_gan(
        generator, discriminator, X_resampled, latent_dim, epochs=1000, batch_size=64
    )

    # Step 4: Generate synthetic data samples
    synthetic_data = generate_synthetic_data(generator, n_samples=1000, latent_dim=latent_dim, columns=X_resampled.columns)

    # Step 5: Combine real and synthetic data
    X_augmented, y_augmented = augment_data(X_resampled, y_resampled, synthetic_data)

    # Step 6: Concatenate into a single DataFrame
    augmented_df = concatenate_data_along_columns(X_augmented, y_augmented)

    # Step 7: Save the final augmented dataset to Google Drive
    if lead_save_true_false:
        print("Saving data to Google Drive...")
        save_dataframe_to_google_drive(augmented_df, x_y_augmented_data_google_drive)

    print("plotting combined analysis 2D and 3D")
    features_engineering_columns = X_augmented.columns
    plot_combined_analysis_2d_3d(fe_processed_df, X_augmented, y_augmented, features_engineering_columns)
    
    print("Data augmentation process complete.")

    return augmented_df, d_loss_real_list, d_loss_fake_list, g_loss_list

# -------------------------- Run the pipeline --------------------------
if __name__ == "__main__":
    # Execute the full augmentation pipeline if the script is run directly
    augmented_df, d_loss_real_list, d_loss_fake_list, g_loss_list = data_augmentation_pipeline()
