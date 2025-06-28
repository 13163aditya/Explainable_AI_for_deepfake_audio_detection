import os
import librosa
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from scipy import stats
import lime
import lime.lime_tabular
import shap
from tqdm import tqdm
import json

# Configuration
DATA_DIR = "c:/Users/adity/Downloads/X_AI_for_fake_real_audio_detection/Data/"
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
FEATURE_COUNT = 76
os.makedirs("research_results", exist_ok=True)

# Feature names matching your extraction pipeline
FEATURE_NAMES = [
    *[f"MFCC_{i + 1}_mean" for i in range(20)],
    *[f"MFCC_{i + 1}_std" for i in range(20)],
    "Chroma_mean", "Chroma_std",
    "SpectralCentroid_mean", "SpectralCentroid_std",
    "SpectralBandwidth_mean", "SpectralBandwidth_std",
    "SpectralRolloff_mean", "SpectralRolloff_std",
    "ZCR_mean", "ZCR_std",
    "RMS_mean", "RMS_std",
    *[f"Contrast_band{i + 1}_mean" for i in range(6)],
    *[f"Contrast_band{i + 1}_std" for i in range(6)],
    *[f"Tonnetz_{i + 1}_mean" for i in range(6)],
    *[f"Tonnetz_{i + 1}_std" for i in range(6)]
]


def extract_features(file_path):
    """Feature extraction matching your trained model"""
    try:
        audio, _ = librosa.load(file_path, sr=SR)
        features = []

        # 1. MFCCs (40 features: 20 means + 20 std)
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # 2. Chroma (2 features)
        chroma = librosa.feature.chroma_stft(y=audio, sr=SR,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.extend([np.mean(chroma), np.std(chroma)])

        # 3. Spectral Features (6 features)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff)
        ])

        # 4. Zero Crossing Rate (2 features)
        zcr = librosa.feature.zero_crossing_rate(audio,
                                                 frame_length=N_FFT, hop_length=HOP_LENGTH)
        features.extend([np.mean(zcr), np.std(zcr)])

        # 5. RMS Energy (2 features)
        rms = librosa.feature.rms(y=audio,
                                  frame_length=N_FFT, hop_length=HOP_LENGTH)
        features.extend([np.mean(rms), np.std(rms)])

        # 6. Spectral Contrast (12 features: 6 means + 6 std)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=SR,
                                                     n_bands=6,
                                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        contrast_mean = np.mean(contrast[:6], axis=1)
        contrast_std = np.std(contrast[:6], axis=1)
        features.extend(contrast_mean)
        features.extend(contrast_std)

        # 7. Tonnetz (12 features: 6 means + 6 std)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=SR)
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))

        features = np.array(features)
        if len(features) != FEATURE_COUNT:
            raise ValueError(f"Feature mismatch: {len(features)} != {FEATURE_COUNT}")

        return features.reshape(1, -1)

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def load_audio_dataset():
    """Load dataset with labels for research analysis"""
    X, y, filenames = [], [], []

    for label, folder in enumerate(["real", "fake"]):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Directory not found: {folder_path}")
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        print(f"Found {len(files)} {folder} samples")

        for file in tqdm(files, desc=f"Processing {folder}"):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features[0])
                y.append(label)
                filenames.append(file)

    return np.array(X), np.array(y), filenames


def perform_statistical_analysis(X, y):
    """Compare features between real and fake audio"""
    print("\nPerforming statistical analysis...")

    # Create DataFrames
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['label'] = y
    real_df = df[df['label'] == 0].drop('label', axis=1)
    fake_df = df[df['label'] == 1].drop('label', axis=1)

    # Calculate statistical significance
    results = []
    for feature in FEATURE_NAMES:
        t_stat, p_val = stats.ttest_ind(real_df[feature], fake_df[feature])
        cohen_d = (real_df[feature].mean() - fake_df[feature].mean()) / np.sqrt(
            (real_df[feature].std() ** 2 + fake_df[feature].std() ** 2) / 2)
        results.append({
            'feature': feature,
            'real_mean': float(real_df[feature].mean()),
            'fake_mean': float(fake_df[feature].mean()),
            'mean_diff': float(real_df[feature].mean() - fake_df[feature].mean()),
            'p_value': float(p_val),
            'effect_size': float(cohen_d),
            'significant': p_val < 0.05
        })

    # Save results
    stats_df = pd.DataFrame(results)
    stats_df.to_csv("research_results/feature_statistics.csv", index=False)

    # Plot top significant features
    top_features = stats_df[stats_df['significant']].sort_values(
        'effect_size', key=abs, ascending=False).head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='effect_size', y='feature', data=top_features, palette='viridis')
    plt.title("Top 20 Most Discriminative Features (by Effect Size)")
    plt.xlabel("Cohen's d Effect Size")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("research_results/top_discriminative_features.png")
    plt.close()

    print("Statistical analysis complete. Results saved to research_results/")


def analyze_with_lime(model, scaler, background_data, X, y, filenames):
    """Analyze predictions using LIME explanations"""
    print("\nAnalyzing with LIME...")

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        background_data,
        feature_names=FEATURE_NAMES,
        class_names=["Real", "Fake"],
        mode="classification",
        verbose=True,
        discretize_continuous=False
    )

    # Analyze samples from each class
    explanations = []
    for label in [0, 1]:
        label_samples = X[y == label]
        sample_filenames = [f for f, l in zip(filenames, y) if l == label]

        for i in range(min(5, len(label_samples))):  # Analyze 5 samples per class
            sample = label_samples[i]
            scaled_sample = scaler.transform(sample.reshape(1, -1)) if scaler else sample.reshape(1, -1)

            try:
                # Get LIME explanation
                try:
                    exp = explainer.explain_instance(
                        scaled_sample[0],
                        model.predict,
                        top_labels=1,
                        num_features=20,
                        num_samples=1000
                    )
                    # Try to get explanation for the actual label first
                    try:
                        exp_list = exp.as_list(label=label)
                    except:
                        # If that fails, try the predicted label
                        pred_label = model.predict(scaled_sample).argmax()
                        exp_list = exp.as_list(label=pred_label)
                except Exception as e:
                    print(f"LIME explanation failed for sample {i} (label {label}): {str(e)}")
                    continue

                # Convert numpy types to native Python types for JSON serialization
                exp_list_py = [(str(feat), float(impact)) for feat, impact in exp_list]

                # Store explanation
                explanations.append({
                    'filename': str(sample_filenames[i]),
                    'label': int(label),
                    'prediction': int(model.predict(scaled_sample).argmax()),
                    'features': exp_list_py
                })

            except Exception as e:
                print(f"Failed to process sample {i} (label {label}): {str(e)}")
                continue

    if not explanations:
        print("Warning: No LIME explanations were successfully generated")
        return

    # Save LIME results
    with open("research_results/lime_explanations.json", "w") as f:
        json.dump(explanations, f, indent=2)

    # Generate summary of important features
    lime_summary = {}
    for exp in explanations:
        for feature, impact in exp['features']:
            if feature not in lime_summary:
                lime_summary[feature] = []
            lime_summary[feature].append(impact)

    # Calculate average impacts
    lime_importance = []
    for feature, impacts in lime_summary.items():
        lime_importance.append({
            'feature': str(feature),
            'mean_impact': float(np.mean(np.abs(impacts))),
            'std_impact': float(np.std(impacts)),
            'count': int(len(impacts))
        })

    lime_df = pd.DataFrame(lime_importance).sort_values('mean_impact', ascending=False)
    lime_df.to_csv("research_results/lime_feature_importance.csv", index=False)

    # Plot top LIME features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mean_impact', y='feature', data=lime_df.head(20), palette='viridis')
    plt.title("Top 20 Features by LIME Importance")
    plt.xlabel("Mean Absolute Impact")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("research_results/lime_feature_importance.png")
    plt.close()

    print("LIME analysis complete. Results saved to research_results/")


def analyze_with_shap(model, X, background_samples=100):
    """Analyze model with SHAP values"""
    print("\nAnalyzing with SHAP...")

    # Create background data
    background = X[np.random.choice(X.shape[0], min(background_samples, X.shape[0]), replace=False)]

    # Create appropriate explainer based on model type
    if len(model.layers) > 1:  # If it's a neural network
        explainer = shap.DeepExplainer(model, background)
    else:
        explainer = shap.Explainer(model.predict, background)

    # Calculate SHAP values
    try:
        shap_values = explainer.shap_values(X)
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        return

    # Plot summary
    plt.figure()
    if isinstance(shap_values, list):
        # For classification models
        shap.summary_plot(shap_values[0], X, feature_names=FEATURE_NAMES, plot_type="bar", show=False)
    else:
        # For regression models
        shap.summary_plot(shap_values, X, feature_names=FEATURE_NAMES, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("research_results/shap_feature_importance.png")
    plt.close()

    # Save SHAP values
    try:
        np.save("research_results/shap_values.npy", shap_values, allow_pickle=True)
    except Exception as e:
        print(f"Error saving SHAP values: {str(e)}")

    print("SHAP analysis complete. Results saved to research_results/")


def main():
    # Load your pre-trained model and artifacts
    print("Loading pre-trained model and artifacts...")
    try:
        model = load_model("audio_model_nn_32.h5")
        scaler = joblib.load("scaler_nn_32.joblib")
        background_data = joblib.load("background_nn_32.joblib")
    except Exception as e:
        print(f"Error loading model or artifacts: {str(e)}")
        return

    # Load and process audio dataset
    print("\nLoading audio dataset...")
    X, y, filenames = load_audio_dataset()

    if len(X) == 0:
        print("No valid samples found. Exiting.")
        return

    # Scale features
    X_scaled = scaler.transform(X)

    # 1. Perform statistical analysis
    perform_statistical_analysis(X, y)

    # 2. Analyze with LIME
    analyze_with_lime(model, scaler, background_data, X_scaled, y, filenames)

    # 3. Analyze with SHAP (may take longer)
    analyze_with_shap(model, X_scaled)

    print("\nResearch analysis complete!")


if __name__ == "__main__":
    main()