import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.models import load_model
import tempfile
import os
import cv2

# Configure the app
st.set_page_config(page_title="Audio Authenticity Detector", layout="wide")
st.title("🔍 Audio Authenticity Detector")
st.markdown("""
This app detects whether an audio recording is **genuine human speech** or **AI-generated fake audio**  
using deep learning and explains its predictions with LIME (Local Interpretable Model-agnostic Explanations).
""")


# Load model once
@st.cache_resource
def load_audio_model():
    return load_model('best_model.h5')  # Replace with your actual model


# Audio processing to mel-spectrogram
def create_spectrogram(audio_path, n_mels=128, target_length=128):
    y, sr = librosa.load(audio_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Fix length
    if mel_spec_db.shape[1] < target_length:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_length - mel_spec_db.shape[1])))
    else:
        mel_spec_db = mel_spec_db[:, :target_length]

    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    return mel_spec_db


# LIME expects RGB, so fake it by stacking grayscale
def to_rgb_image(spectrogram):
    spectrogram_rgb = np.stack([spectrogram]*3, axis=-1)  # Shape: (H, W, 3)
    spectrogram_rgb = (spectrogram_rgb * 255).astype(np.uint8)
    return spectrogram_rgb


# Generate LIME explanation
def generate_lime_explanation(model, spectrogram, num_samples=2000):
    explainer = lime_image.LimeImageExplainer()

    # Convert spectrogram to RGB
    image_rgb = to_rgb_image(spectrogram)

    def model_predict(images):
        # Convert back to grayscale and reshape
        images_gray = np.mean(images, axis=-1)
        images_gray = images_gray[..., np.newaxis]  # Add channel
        return model.predict(images_gray)

    explanation = explainer.explain_instance(
        image_rgb,
        model_predict,
        top_labels=1,
        num_samples=num_samples,
        batch_size=32,
        hide_color=0
    )
    return explanation


# Main Streamlit app
def main():
    model = load_audio_model()

    with st.sidebar:
        st.header("Settings")
        num_samples = st.slider("LIME Explanation Detail", 1000, 5000, 2000, 500)
        st.markdown("---")
        st.info("Note: Explanations may take 30–60 seconds to compute.")

    uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=['wav'])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name

        try:
            with st.spinner("Processing audio..."):
                # Generate spectrogram
                spec = create_spectrogram(audio_path)

                # Predict
                pred = model.predict(spec[np.newaxis, ..., np.newaxis])[0][0]
                label = "REAL" if pred >= 0.5 else "FAKE"
                confidence = abs(pred - 0.5) * 200  # Convert to %

                # Generate explanation
                explanation = generate_lime_explanation(model, spec, num_samples)
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0],
                    positive_only=False,
                    num_features=10,
                    hide_rest=False
                )

                # Show results
                st.success(f"Prediction: **{label}** (Confidence: {confidence:.1f}%)")

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Audio Player")
                    st.audio(uploaded_file)

                    st.subheader("Mel-Spectrogram")
                    fig1 = plt.figure(figsize=(8, 4))
                    librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='mel', cmap='magma')
                    plt.colorbar()
                    st.pyplot(fig1)

                with col2:
                    st.subheader("Explanation (LIME)")
                    fig2 = plt.figure(figsize=(10, 8))

                    plt.subplot(2, 1, 1)
                    plt.imshow(mask, cmap='coolwarm', aspect='auto')
                    plt.title("Heatmap of Important Regions")
                    plt.colorbar()

                    plt.subplot(2, 1, 2)
                    lime_overlay = mark_boundaries(to_rgb_image(spec), mask > 0)
                    plt.imshow(lime_overlay, aspect='auto')
                    plt.title("LIME: Highlighted Regions")
                    st.pyplot(fig2)

                    # Show top features (optional)
                    st.subheader("Top Influential Regions")
                    features = sorted(explanation.local_exp[explanation.top_labels[0]],
                                      key=lambda x: abs(x[1]), reverse=True)[:10]

                    rows = []
                    for i, (idx, weight) in enumerate(features, start=1):
                        freq_band = idx // spec.shape[1]
                        time_step = idx % spec.shape[1]
                        rows.append([
                            i,
                            f"{freq_band}-{freq_band+1}",
                            f"{time_step/spec.shape[1]*3:.1f}s",
                            f"{weight:.3f}",
                            "Supports REAL" if weight > 0 else "Supports FAKE"
                        ])

                    st.table({
                        "Rank": [r[0] for r in rows],
                        "Freq Band": [r[1] for r in rows],
                        "Time Pos": [r[2] for r in rows],
                        "Weight": [r[3] for r in rows],
                        "Effect": [r[4] for r in rows],
                    })
        finally:
            os.unlink(audio_path)


if __name__ == "__main__":
    main()
