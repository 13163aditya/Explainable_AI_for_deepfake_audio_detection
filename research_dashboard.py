import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib
from tensorflow.keras.models import load_model
import json
import lime
import lime.lime_tabular
import shap
from PIL import Image

# Configuration
st.set_page_config(layout="wide", page_title="Audio Authenticity Research")
FEATURE_COUNT = 76

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


@st.cache_resource
def load_artifacts():
    """Load pre-trained model and artifacts"""
    artifacts = {}
    try:
        artifacts['model'] = load_model("audio_model_nn_32.h5")
        artifacts['scaler'] = joblib.load("scaler_nn_32.joblib")
        artifacts['background'] = joblib.load("background_nn_32.joblib")

        # Load research results if available
        try:
            artifacts['stats'] = pd.read_csv("research_results/feature_statistics.csv")
            artifacts['lime_importance'] = pd.read_csv("research_results/lime_feature_importance.csv")
            with open("research_results/lime_explanations.json", "r") as f:
                artifacts['lime_explanations'] = json.load(f)
        except:
            st.warning("Some research results not found. Run audio_research.py first.")

        return artifacts
    except Exception as e:
        st.error(f"Error loading artifacts: {str(e)}")
        return None


def plot_audio_analysis(audio):
    """Plot comprehensive audio visualizations"""
    fig, ax = plt.subplots(3, 2, figsize=(15, 12))

    # Waveform
    librosa.display.waveshow(audio, sr=22050, ax=ax[0, 0])
    ax[0, 0].set(title='Waveform', xlabel='Time', ylabel='Amplitude')

    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    img = librosa.display.specshow(D, sr=22050, hop_length=512,
                                   x_axis='time', y_axis='log', ax=ax[0, 1])
    fig.colorbar(img, ax=ax[0, 1], format="%+2.0f dB")
    ax[0, 1].set(title='Spectrogram')

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=20)
    img = librosa.display.specshow(mfccs, sr=22050, hop_length=512,
                                   x_axis='time', ax=ax[1, 0])
    fig.colorbar(img, ax=ax[1, 0])
    ax[1, 0].set(title='MFCCs')

    # Chromagram
    chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
    img = librosa.display.specshow(chroma, sr=22050, hop_length=512,
                                   x_axis='time', y_axis='chroma', ax=ax[1, 1])
    fig.colorbar(img, ax=ax[1, 1])
    ax[1, 1].set(title='Chromagram')

    # Spectral Features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=22050)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=22050)[0]
    frames = range(len(spectral_centroid))
    t = librosa.frames_to_time(frames, sr=22050)
    ax[2, 0].plot(t, spectral_centroid, label='Spectral Centroid')
    ax[2, 0].plot(t, spectral_rolloff, color='r', label='Rolloff Frequency')
    ax[2, 0].legend(loc='upper right')
    ax[2, 0].set(title='Spectral Features', xlabel='Time')

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    t = librosa.frames_to_time(range(len(zcr[0])), sr=22050)
    ax[2, 1].plot(t, zcr[0])
    ax[2, 1].set(title='Zero Crossing Rate', xlabel='Time')

    plt.tight_layout()
    return fig


def show_feature_analysis(artifacts):
    """Show feature importance analysis"""
    st.header("🔍 Feature Importance Analysis")

    tab1, tab2, tab3 = st.tabs(["Statistical Analysis", "LIME Analysis", "SHAP Analysis"])

    with tab1:
        if 'stats' in artifacts:
            st.subheader("Top Discriminative Features")
            st.image("research_results/top_discriminative_features.png")

            st.subheader("Feature Statistics")
            st.dataframe(artifacts['stats'].sort_values('effect_size', key=abs, ascending=False))
        else:
            st.warning("Statistical analysis results not found. Run audio_research.py first.")

    with tab2:
        if 'lime_importance' in artifacts:
            st.subheader("LIME Feature Importance")
            st.image("research_results/lime_feature_importance.png")

            st.subheader("Example LIME Explanations")
            if 'lime_explanations' in artifacts:
                selected_exp = st.selectbox(
                    "Select explanation to view",
                    [f"{exp['filename']} (Label: {exp['label']})" for exp in artifacts['lime_explanations']]
                )
                exp = artifacts['lime_explanations'][int(selected_exp.split()[0])]

                st.write(f"**Filename:** {exp['filename']}")
                st.write(f"**True Label:** {'Real' if exp['label'] == 0 else 'Fake'}")
                st.write(f"**Model Prediction:** {'Real' if exp['prediction'] == 0 else 'Fake'}")

                # Show feature impacts
                impacts = pd.DataFrame(exp['features'], columns=['Feature', 'Impact'])
                st.dataframe(impacts.style.applymap(
                    lambda x: 'color: blue' if x > 0 else 'color: red',
                    subset=['Impact']
                ))
        else:
            st.warning("LIME analysis results not found. Run audio_research.py first.")

    with tab3:
        try:
            st.subheader("SHAP Feature Importance")
            st.image("research_results/shap_feature_importance.png")
        except:
            st.warning("SHAP analysis results not found. Run audio_research.py first.")


def analyze_audio_file(artifacts):
    """Analyze uploaded audio file"""
    st.header("🎤 Audio Analysis")

    uploaded_file = st.file_uploader("Upload WAV audio file", type=["wav"])

    if uploaded_file:
        with st.spinner("Analyzing audio..."):
            # Save to temp file
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                # Load and play audio
                audio, _ = librosa.load("temp_audio.wav", sr=22050)
                st.audio(uploaded_file, format="audio/wav")

                # Show audio visualizations
                st.subheader("Audio Visualizations")
                audio_fig = plot_audio_analysis(audio)
                st.pyplot(audio_fig)
                plt.close()

                # Extract features
                features = extract_features("temp_audio.wav")
                if features is None:
                    st.error("Feature extraction failed")
                    return

                # Scale features and predict
                features_scaled = artifacts['scaler'].transform(features)
                proba = artifacts['model'].predict(features_scaled, verbose=0)[0]
                prediction = np.argmax(proba)
                confidence = proba[prediction]

                # Show prediction
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Prediction",
                              "Real" if prediction == 0 else "Fake",
                              delta=f"{confidence * 100:.1f}% confidence")
                with col2:
                    st.metric("Probability Distribution",
                              f"Real: {proba[0]:.3f} | Fake: {proba[1]:.3f}")

                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    artifacts['background'],
                    feature_names=FEATURE_NAMES,
                    class_names=["Real", "Fake"],
                    mode="classification"
                )

                # Explain prediction
                exp = explainer.explain_instance(
                    features_scaled[0],
                    artifacts['model'].predict,
                    top_labels=1,
                    num_features=20
                )

                # Show explanation
                st.subheader("Model Explanation (LIME)")
                lime_list = exp.as_list(label=prediction)
                lime_df = pd.DataFrame(lime_list, columns=['Feature', 'Impact'])

                # Color by impact direction
                def color_impact(val):
                    color = 'blue' if val > 0 else 'red'
                    return f'color: {color}'

                st.dataframe(lime_df.style.applymap(color_impact, subset=['Impact']))

                # Interpretation guide
                with st.expander("How to interpret this explanation"):
                    st.markdown("""
                    - **Blue features**: Support the prediction (higher values make the prediction more likely)
                    - **Red features**: Contradict the prediction (higher values make the prediction less likely)
                    - **Magnitude**: Shows how strongly each feature influenced the decision
                    """)

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")


def main():
    st.title("🔬 Audio Authenticity Research Dashboard")

    # Load artifacts
    artifacts = load_artifacts()
    if artifacts is None:
        st.error("Could not load model artifacts. Ensure files are in the correct location.")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["Feature Analysis", "Test Audio"])

    with tab1:
        show_feature_analysis(artifacts)

    with tab2:
        analyze_audio_file(artifacts)


if __name__ == "__main__":
    main()