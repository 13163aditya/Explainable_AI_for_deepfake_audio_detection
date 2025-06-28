import streamlit as st
import librosa
import numpy as np
import joblib
import tempfile
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import load_model
import os
import warnings
import seaborn as sns
import librosa.display
from tqdm import tqdm
from glob import glob
import sounddevice as sd
from scipy.io.wavfile import write

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configuration - MUST match training
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
FEATURE_COUNT = 76
DATA_DIR = "c:/Users/adity/Downloads/X_AI_for_fake_real_audio_detection/processed_audio/"


def record_audio(duration=5, sample_rate=22050):
    """Record audio from microphone"""
    st.write(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1,
                       dtype='float32')
    sd.wait()  # Wait until recording is finished
    return recording.flatten(), sample_rate


def plot_audio_features(audio):
    """Plot various audio visualizations with error handling"""
    try:
        # Create figure with subplots
        fig, ax = plt.subplots(3, 2, figsize=(15, 12))

        # Check if audio is valid
        if len(audio) == 0:
            raise ValueError("Empty audio data")

        # 1. Waveform
        try:
            librosa.display.waveshow(audio, sr=SR, ax=ax[0, 0])
            ax[0, 0].set(title='Waveform', xlabel='Time', ylabel='Amplitude')
        except Exception as e:
            ax[0, 0].set(title='Waveform (failed)', visible=False)
            st.warning(f"Could not generate waveform: {str(e)}")

        # 2. Spectrogram
        try:
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=SR, hop_length=HOP_LENGTH,
                                           x_axis='time', y_axis='log', ax=ax[0, 1])
            fig.colorbar(img, ax=ax[0, 1], format="%+2.0f dB")
            ax[0, 1].set(title='Spectrogram')
        except Exception as e:
            ax[0, 1].set(title='Spectrogram (failed)', visible=False)
            st.warning(f"Could not generate spectrogram: {str(e)}")

        # 3. MFCCs
        try:
            mfccs = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20)
            img = librosa.display.specshow(mfccs, sr=SR, hop_length=HOP_LENGTH,
                                           x_axis='time', ax=ax[1, 0])
            fig.colorbar(img, ax=ax[1, 0])
            ax[1, 0].set(title='MFCCs')
        except Exception as e:
            ax[1, 0].set(title='MFCCs (failed)', visible=False)
            st.warning(f"Could not generate MFCCs: {str(e)}")

        # 4. Chromagram
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=SR)
            img = librosa.display.specshow(chroma, sr=SR, hop_length=HOP_LENGTH,
                                           x_axis='time', y_axis='chroma', ax=ax[1, 1])
            fig.colorbar(img, ax=ax[1, 1])
            ax[1, 1].set(title='Chromagram')
        except Exception as e:
            ax[1, 1].set(title='Chromagram (failed)', visible=False)
            st.warning(f"Could not generate chromagram: {str(e)}")

        # 5. Spectral Centroid and Rolloff
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)[0]
            frames = range(len(spectral_centroid))
            t = librosa.frames_to_time(frames, sr=SR)
            ax[2, 0].plot(t, spectral_centroid, label='Spectral Centroid')
            ax[2, 0].plot(t, spectral_rolloff, color='r', label='Rolloff Frequency')
            ax[2, 0].legend(loc='upper right')
            ax[2, 0].set(title='Spectral Features', xlabel='Time')
        except Exception as e:
            ax[2, 0].set(title='Spectral Features (failed)', visible=False)
            st.warning(f"Could not generate spectral features: {str(e)}")

        # 6. Zero Crossing Rate
        try:
            zcr = librosa.feature.zero_crossing_rate(audio)
            if len(zcr) > 0:
                t = librosa.frames_to_time(range(len(zcr[0])), sr=SR)
                ax[2, 1].plot(t, zcr[0])
                ax[2, 1].set(title='Zero Crossing Rate', xlabel='Time')
            else:
                raise ValueError("Empty ZCR data")
        except Exception as e:
            ax[2, 1].set(title='Zero Crossing Rate (failed)', visible=False)
            st.warning(f"Could not generate ZCR: {str(e)}")

        plt.tight_layout()
        return fig

    except Exception as e:
        st.error(f"Audio visualization error: {str(e)}")
        return None


def extract_features(file_path):
    """Feature extraction matching train.py"""
    try:
        audio, _ = librosa.load(file_path, sr=SR)
        features = []

        # 1. MFCCs (40 features)
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # 2. Chroma (2 features)
        chroma = librosa.feature.chroma_stft(y=audio, sr=SR,
                                             n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.extend([np.mean(chroma), np.std(chroma)])

        # 3. Spectral Features (6)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=SR)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=SR)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SR)
        features.extend([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
            np.mean(spectral_rolloff), np.std(spectral_rolloff)
        ])

        # 4. Zero Crossing Rate (2)
        zcr = librosa.feature.zero_crossing_rate(audio,
                                                 frame_length=N_FFT, hop_length=HOP_LENGTH)
        features.extend([np.mean(zcr), np.std(zcr)])

        # 5. RMS Energy (2)
        rms = librosa.feature.rms(y=audio,
                                  frame_length=N_FFT, hop_length=HOP_LENGTH)
        features.extend([np.mean(rms), np.std(rms)])

        # 6. Spectral Contrast (12)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=SR,
                                                     n_bands=6,
                                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
        features.extend(np.mean(contrast[:6], axis=1))
        features.extend(np.std(contrast[:6], axis=1))

        # 7. Tonnetz (12)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=SR)
        features.extend(np.mean(tonnetz, axis=1))
        features.extend(np.std(tonnetz, axis=1))

        features = np.array(features)
        if len(features) != FEATURE_COUNT:
            raise ValueError(f"Feature mismatch: {len(features)} != {FEATURE_COUNT}")

        return features.reshape(1, -1)

    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None


def plot_top_features(model, feature_names, n_features=20):
    """Plot top features based on neural network weights"""
    try:
        # Get weights from first dense layer
        weights = model.layers[0].get_weights()[0]

        # Calculate mean absolute weights across neurons
        importance = np.mean(np.abs(weights), axis=1)

        # Handle output layer if single neuron
        if len(importance) != len(feature_names):
            importance = np.abs(weights.flatten())

        # Create DataFrame
        feat_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False).head(n_features)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
        plt.title(f'Top {n_features} Important Features (Neural Network Weights)')
        plt.tight_layout()
        return plt.gcf()

    except Exception as e:
        st.error(f"Feature importance error: {str(e)}")
        return None


def analyze_full_dataset(model, scaler, background_data, feature_names, sample_size=50):
    """Analyze all audio files in the dataset"""
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        background_data,
        feature_names=feature_names,
        class_names=["Real", "Fake"],
        mode="classification"
    )

    results = []

    for label, folder in enumerate(["real", "fake"]):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.exists(folder_path):
            continue

        files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        files = files[:min(sample_size, len(files))]  # Limit sample size

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(files):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)

            if features is not None:
                scaled_features = scaler.transform(features)
                proba = model.predict(scaled_features, verbose=0)[0]
                pred_class = np.argmax(proba)

                # Get LIME explanation
                exp = lime_explainer.explain_instance(
                    scaled_features[0],
                    lambda x: model.predict(x, verbose=0),
                    top_labels=1,
                    num_features=20
                )

                # Process explanation
                lime_values = exp.as_list(label=pred_class)
                for feature, value in lime_values:
                    results.append({
                        'file': file,
                        'label': folder,
                        'feature': feature.split('=')[0].strip() if '=' in feature else feature,
                        'value': value,
                        'abs_value': abs(value),
                        'prediction': "Real" if pred_class == 0 else "Fake",
                        'confidence': proba[pred_class]
                    })

            # Update progress
            progress = (i + 1) / len(files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {folder} files: {i + 1}/{len(files)}")

    return pd.DataFrame(results)


def visualize_dataset_results(results_df):
    """Visualize the full dataset analysis results with detailed feature statistics"""
    st.subheader("📊 Full Dataset Analysis Results")

    # Overall statistics
    st.write(f"Total files analyzed: {len(results_df['file'].unique())}")
    st.write(f"Total feature contributions recorded: {len(results_df)}")

    # Feature importance by class
    st.subheader("Feature Importance by Class")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Real Audio Samples**")
        real_features = results_df[results_df['label'] == 'real']
        real_top = real_features.groupby('feature')['abs_value'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(real_top)

    with col2:
        st.write("**Fake Audio Samples**")
        fake_features = results_df[results_df['label'] == 'fake']
        fake_top = fake_features.groupby('feature')['abs_value'].mean().sort_values(ascending=False).head(10)
        st.bar_chart(fake_top)

    # Detailed Feature Statistics Section
    st.subheader("📈 Comprehensive Feature Statistics")
    st.write("""
    This section provides detailed statistics for each audio feature and explains how they typically differ between real and fake audio samples.
    """)

    # Group statistics by feature and label
    feature_stats = results_df.groupby(['feature', 'label']).agg({
        'value': ['mean', 'std', 'count'],
        'abs_value': ['mean']
    })

    # Sort by absolute value mean (descending)
    feature_stats = feature_stats.sort_values(('abs_value', 'mean'), ascending=False)

    # Display interactive feature selector
    selected_feature = st.selectbox(
        "Select a feature to view detailed analysis:",
        sorted(results_df['feature'].unique()),
        index=0
    )

    # Get data for selected feature
    feat_data = results_df[results_df['feature'] == selected_feature]
    real_data = feat_data[feat_data['label'] == 'real']
    fake_data = feat_data[feat_data['label'] == 'fake']

    # Display feature statistics
    st.markdown(f"### 🎯 {selected_feature} Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Impact (Real)", f"{real_data['value'].mean():.4f}")
        st.metric("Standard Deviation (Real)", f"{real_data['value'].std():.4f}")
        st.metric("Sample Count (Real)", len(real_data))

    with col2:
        st.metric("Mean Impact (Fake)", f"{fake_data['value'].mean():.4f}")
        st.metric("Standard Deviation (Fake)", f"{fake_data['value'].std():.4f}")
        st.metric("Sample Count (Fake)", len(fake_data))

    # Feature impact direction visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x='label', y='value',
        data=feat_data,
        palette=['blue', 'red'],
        ax=ax
    )
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Impact Distribution for {selected_feature}")
    plt.ylabel("LIME Value (Positive supports prediction)")
    st.pyplot(fig)

    # Feature interpretation guide
    st.markdown("""
    ### 📖 Feature Interpretation Guide

    Below is a detailed explanation of what different types of features typically represent and how they might differ between real and fake audio:
    """)

    feature_categories = {
        "MFCC Features": """
        **Mel-Frequency Cepstral Coefficients (MFCCs)**:
        - Represent the short-term power spectrum of sound
        - Real audio typically shows more natural variation in MFCCs
        - Fake audio often has more uniform MFCC distributions
        - Higher MFCC values in fake audio may indicate artificial smoothing
        - Look for differences in mean values (overall spectral shape) and std (variability)
        """,
        "Chroma Features": """
        **Chroma Features**:
        - Represent the 12 distinct pitch classes in music
        - Real audio has more natural harmonic relationships
        - Fake audio may show artificial harmonic patterns
        - Higher chroma std in fake audio may indicate synthesis artifacts
        """,
        "Spectral Features": """
        **Spectral Characteristics**:
        - Centroid: "Brightness" of the sound (higher = more high frequencies)
        - Bandwidth: Range of frequencies present
        - Rolloff: Frequency below which 85% of energy is contained
        - Real audio typically has more natural spectral transitions
        - Fake audio often has artificial spectral contours
        """,
        "ZCR": """
        **Zero Crossing Rate**:
        - Measures how often the signal changes sign
        - Higher values indicate more high-frequency content
        - Real speech has natural ZCR patterns
        - Fake audio may show artificially regular ZCR
        """,
        "RMS": """
        **Root Mean Square Energy**:
        - Represents the overall loudness/energy
        - Real audio has natural dynamic range
        - Fake audio may show compressed or uniform energy
        """,
        "Spectral Contrast": """
        **Spectral Contrast**:
        - Measures the difference between peaks and valleys in the spectrum
        - Real audio has natural contrast between harmonics and noise
        - Fake audio may show artificial contrast patterns
        """,
        "Tonnetz": """
        **Tonnetz Features**:
        - Represent tonal space and harmonic relationships
        - Real audio shows natural harmonic progressions
        - Fake audio may show artificial tonal patterns
        """
    }

    # Determine which category the selected feature belongs to
    feature_category = None
    if "MFCC" in selected_feature:
        feature_category = "MFCC Features"
    elif "Chroma" in selected_feature:
        feature_category = "Chroma Features"
    elif "Spectral" in selected_feature:
        feature_category = "Spectral Features"
    elif "ZCR" in selected_feature:
        feature_category = "ZCR"
    elif "RMS" in selected_feature:
        feature_category = "RMS"
    elif "Contrast" in selected_feature:
        feature_category = "Spectral Contrast"
    elif "Tonnetz" in selected_feature:
        feature_category = "Tonnetz"

    if feature_category:
        st.markdown(f"#### {feature_category}")
        st.markdown(feature_categories[feature_category])

    # Interpretation of current feature's statistics
    st.markdown("""
    ### 🔍 Interpretation of This Feature's Statistics

    Based on the current analysis:
    """)

    real_mean = real_data['value'].mean()
    fake_mean = fake_data['value'].mean()

    if abs(real_mean) > abs(fake_mean):
        st.write("- This feature has a **stronger impact** on real audio classifications")
    else:
        st.write("- This feature has a **stronger impact** on fake audio classifications")

    if real_mean * fake_mean > 0:  # Same direction
        if abs(real_mean) > abs(fake_mean):
            st.write("- Both real and fake audio are affected in the same direction, but real audio more strongly")
        else:
            st.write("- Both real and fake audio are affected in the same direction, but fake audio more strongly")
    else:
        st.write("- This feature affects real and fake audio in **opposite directions**")

    if real_data['value'].std() > fake_data['value'].std():
        st.write("- Real audio shows **more variability** in how this feature affects classification")
    else:
        st.write("- Fake audio shows **more variability** in how this feature affects classification")

    # Full statistics table
    st.markdown("### 📋 Complete Feature Statistics Table")

    # Display the multi-index DataFrame properly
    st.dataframe(
        feature_stats.style.background_gradient(
            cmap='Blues',
            subset=pd.IndexSlice[:, [('value', 'mean'), ('abs_value', 'mean')]]
        )
    )

    # Correlation analysis
    st.markdown("### 🔗 Feature Correlation Analysis")
    try:
        # Pivot data for correlation analysis
        pivot_df = results_df.pivot_table(
            index='file',
            columns='feature',
            values='value',
            aggfunc='mean'
        ).corr()

        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            pivot_df,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            annot=False
        )
        plt.title("Feature Value Correlation Matrix")
        st.pyplot(plt.gcf())
        plt.close()

        st.write("""
        **How to interpret the correlation matrix:**
        - Positive values (red): Features that tend to vary together
        - Negative values (blue): Features that tend to vary inversely
        - Near zero: Little relationship between features
        """)
    except Exception as e:
        st.warning(f"Could not generate correlation matrix: {str(e)}")


def main():
    st.title("🎙️ Audio Authenticity Detector")
    st.markdown("Upload a WAV file or record audio to check if it's real or AI-generated")

    # Initialize session state
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'dataset_results' not in st.session_state:
        st.session_state.dataset_results = None

    try:
        # Load model and artifacts
        with st.spinner("Loading model resources..."):
            model = load_model("model_results_large_with_shap/best_model.h5")
            scaler = joblib.load("model_results_large_with_shap/scaler.joblib")
            background_data = joblib.load("model_results_large_with_shap/background_data.joblib")

            # Generate feature names
            feature_names = [
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

        st.success("✅ Model and artifacts loaded successfully")
    except Exception as e:
        st.error(f"❌ Loading error: {str(e)}")
        return

    # Analysis mode selection
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single File Analysis", "Full Dataset Analysis"]
    )

    if analysis_mode == "Single File Analysis":
        # Input method selection
        input_method = st.radio(
            "Input Method",
            ["Upload WAV File", "Record Live Audio"],
            horizontal=True
        )

        if input_method == "Record Live Audio":
            duration = st.slider("Recording duration (seconds)", 1, 10, 5)
            if st.button("Start Recording"):
                with st.spinner("Recording..."):
                    try:
                        # Record audio
                        audio, sr = record_audio(duration)

                        # Save to temp file (to reuse existing pipeline)
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            write(tmp_file.name, sr, audio)
                            tmp_path = tmp_file.name

                        # Visualization
                        st.subheader("🎵 Audio Analysis")
                        audio_fig = plot_audio_features(audio)
                        if audio_fig:
                            st.pyplot(audio_fig)
                            plt.close()

                        # Feature extraction and prediction
                        features = extract_features(tmp_path)
                        os.unlink(tmp_path)

                        if features is None or features.shape[1] != FEATURE_COUNT:
                            st.error(f"❌ Feature extraction failed! Expected {FEATURE_COUNT} features")
                            return

                        # Feature scaling and prediction
                        features_scaled = scaler.transform(features)
                        proba = model.predict(features_scaled, verbose=0)[0]
                        prediction = np.argmax(proba)
                        confidence = proba[prediction]

                        # Display results
                        st.subheader("🔍 Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction",
                                      "Real" if prediction == 0 else "Fake",
                                      delta=f"{confidence * 100:.1f}% confidence")
                        with col2:
                            st.metric("Probability Distribution",
                                      f"Real: {proba[0]:.3f}\nFake: {proba[1]:.3f}")

                        # Feature Analysis
                        st.subheader("📊 Explanation")

                        # Top 20 Features
                        with st.expander("🔝 Top 20 Important Features"):
                            with st.spinner("Calculating feature importance..."):
                                top_features_plot = plot_top_features(model, feature_names)
                                if top_features_plot:
                                    st.pyplot(top_features_plot)
                                    plt.close()
                                else:
                                    st.warning("Could not generate feature importance plot")

                        # LIME Explanation
                        with st.expander("🍋 LIME Explanation"):
                            try:
                                with st.spinner("Generating LIME explanation..."):
                                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                                        background_data,
                                        feature_names=feature_names,
                                        class_names=["Real", "Fake"],
                                        mode="classification",
                                        verbose=True
                                    )

                                    # Get predicted class index
                                    pred_class = np.argmax(proba)

                                    exp = lime_explainer.explain_instance(
                                        features_scaled[0],
                                        lambda x: model.predict(x, verbose=0),
                                        top_labels=1,
                                        num_features=20,
                                        labels=[pred_class]  # Explicitly explain predicted class
                                    )

                                    # Plot for the predicted class
                                    fig = exp.as_pyplot_figure(label=pred_class)
                                    plt.title("LIME Explanation (Blue supports prediction, Red opposes)")
                                    st.pyplot(fig)
                                    plt.close()

                                    # Show explanation as list
                                    st.markdown("""
                                    **Detailed Feature Contributions:**
                                    - Blue bars: Features supporting the prediction
                                    - Red bars: Features opposing the prediction
                                    """)

                                    lime_list = exp.as_list(label=pred_class)
                                    lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Impact'])

                                    # Color the dataframe based on impact
                                    def color_impact(val):
                                        color = 'blue' if val > 0 else 'red'
                                        return f'color: {color}'

                                    styled_df = lime_df.style.applymap(color_impact, subset=['Impact'])
                                    st.dataframe(styled_df)

                                    # Additional explanation
                                    st.markdown("""
                                    **How to interpret this explanation:**
                                    - Positive values (blue) indicate features that support the model's prediction
                                    - Negative values (red) indicate features that contradict the model's prediction
                                    - The magnitude shows how strongly each feature influenced the decision
                                    """)

                            except Exception as e:
                                st.error(f"⚠️ LIME explanation failed: {str(e)}")

                    except Exception as e:
                        st.error(f"Recording error: {str(e)}")

        else:  # Original file uploader code
            uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])
            if uploaded_file:
                with st.spinner("Analyzing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_path = tmp_file.name

                    try:
                        # Load audio for visualization
                        audio, _ = librosa.load(tmp_path, sr=SR)

                        # Audio Visualization
                        st.subheader("🎵 Audio Analysis")
                        with st.spinner("Generating audio visualizations..."):
                            audio_fig = plot_audio_features(audio)
                            if audio_fig:
                                st.pyplot(audio_fig)
                                plt.close()

                        # Feature extraction
                        features = extract_features(tmp_path)
                        os.unlink(tmp_path)

                        if features is None or features.shape[1] != FEATURE_COUNT:
                            st.error(f"❌ Feature extraction failed! Expected {FEATURE_COUNT} features")
                            return

                        # Feature scaling and prediction
                        features_scaled = scaler.transform(features)
                        proba = model.predict(features_scaled, verbose=0)[0]
                        prediction = np.argmax(proba)
                        confidence = proba[prediction]

                        # Display results
                        st.subheader("🔍 Results")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Prediction",
                                      "Real" if prediction == 0 else "Fake",
                                      delta=f"{confidence * 100:.1f}% confidence")
                        with col2:
                            st.metric("Probability Distribution",
                                      f"Real: {proba[0]:.3f}\nFake: {proba[1]:.3f}")

                        # Feature Analysis
                        st.subheader("📊 Explanation")

                        # Top 20 Features
                        with st.expander("🔝 Top 20 Important Features"):
                            with st.spinner("Calculating feature importance..."):
                                top_features_plot = plot_top_features(model, feature_names)
                                if top_features_plot:
                                    st.pyplot(top_features_plot)
                                    plt.close()
                                else:
                                    st.warning("Could not generate feature importance plot")

                        # LIME Explanation
                        with st.expander("🍋 LIME Explanation"):
                            try:
                                with st.spinner("Generating LIME explanation..."):
                                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                                        background_data,
                                        feature_names=feature_names,
                                        class_names=["Real", "Fake"],
                                        mode="classification",
                                        verbose=True
                                    )

                                    # Get predicted class index
                                    pred_class = np.argmax(proba)

                                    exp = lime_explainer.explain_instance(
                                        features_scaled[0],
                                        lambda x: model.predict(x, verbose=0),
                                        top_labels=1,
                                        num_features=20,
                                        labels=[pred_class]  # Explicitly explain predicted class
                                    )

                                    # Plot for the predicted class
                                    fig = exp.as_pyplot_figure(label=pred_class)
                                    plt.title("LIME Explanation (Blue supports prediction, Red opposes)")
                                    st.pyplot(fig)
                                    plt.close()

                                    # Show explanation as list
                                    st.markdown("""
                                    **Detailed Feature Contributions:**
                                    - Blue bars: Features supporting the prediction
                                    - Red bars: Features opposing the prediction
                                    """)

                                    lime_list = exp.as_list(label=pred_class)
                                    lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Impact'])

                                    # Color the dataframe based on impact
                                    def color_impact(val):
                                        color = 'blue' if val > 0 else 'red'
                                        return f'color: {color}'

                                    styled_df = lime_df.style.applymap(color_impact, subset=['Impact'])
                                    st.dataframe(styled_df)

                                    # Additional explanation
                                    st.markdown("""
                                    **How to interpret this explanation:**
                                    - Positive values (blue) indicate features that support the model's prediction
                                    - Negative values (red) indicate features that contradict the model's prediction
                                    - The magnitude shows how strongly each feature influenced the decision
                                    """)

                            except Exception as e:
                                st.error(f"⚠️ LIME explanation failed: {str(e)}")

                    except Exception as e:
                        st.error(f"❌ Processing error: {str(e)}")

    else:  # Full Dataset Analysis
        st.subheader("Full Dataset Analysis")
        st.write(
            "This will analyze all audio files in your dataset to identify consistent patterns in feature importance.")

        sample_size = st.slider("Maximum samples per class", 10, 200, 50)

        if st.button("Run Full Dataset Analysis"):
            if st.session_state.dataset_results is not None:
                if st.button("Use cached results"):
                    visualize_dataset_results(st.session_state.dataset_results)
                    return

            with st.spinner("Analyzing dataset (this may take several minutes)..."):
                results_df = analyze_full_dataset(
                    model, scaler, background_data, feature_names, sample_size
                )
                st.session_state.dataset_results = results_df
                visualize_dataset_results(results_df)

        elif st.session_state.dataset_results is not None:
            if st.button("Show Previous Results"):
                visualize_dataset_results(st.session_state.dataset_results)


if __name__ == "__main__":
    main()
