import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
from tqdm import tqdm
import os
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from textwrap import wrap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Configuration
SR = 22050  # Sample rate
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between frames
DATA_DIR = "./fake-real-audio/"  # Update this path to your dataset
MAX_FILES_PER_CLASS = 200  # Maximum files to process per class
STANDARD_DURATION = 3  # Standard audio duration in seconds
OUTPUT_DIR = "audio_analysis_reports"  # Output directory for reports
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist

# LIME Configuration
NUM_LIME_FEATURES = 10  # Number of top features to show in LIME explanation
NUM_LIME_SAMPLES = 5  # Number of samples to analyze with LIME per class


def extract_features_for_lime(audio):
    """Extract features for LIME analysis"""
    features = []

    # Waveform statistics
    features.extend([np.mean(audio), np.std(audio), np.max(audio), np.min(audio)])

    # Spectrogram features
    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    features.extend([np.mean(S_db), np.std(S_db), np.max(S_db), np.min(S_db)])

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH)
    features.extend([np.mean(chroma), np.std(chroma)])

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=N_FFT, hop_length=HOP_LENGTH)
    features.extend([np.mean(zcr), np.std(zcr)])

    # Spectral features
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SR)))
    features.append(np.mean(librosa.feature.spectral_flatness(y=audio)))

    return np.array(features)


def create_lime_explanations(X_train, X_test, y_train, y_test, feature_names):
    """Create and visualize LIME explanations"""
    # Train a simple classifier
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Classifier accuracy: {accuracy:.2f}")

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_scaled,
        feature_names=feature_names,
        class_names=['Fake', 'Real'],
        mode='classification'
    )

    # Select samples to explain
    real_samples = X_test_scaled[y_test == 1][:NUM_LIME_SAMPLES]
    fake_samples = X_test_scaled[y_test == 0][:NUM_LIME_SAMPLES]

    # Generate explanations
    lime_results = {'real': [], 'fake': []}

    for sample in real_samples:
        exp = explainer.explain_instance(sample, clf.predict_proba, num_features=NUM_LIME_FEATURES)
        lime_results['real'].append(exp.as_list())

    for sample in fake_samples:
        exp = explainer.explain_instance(sample, clf.predict_proba, num_features=NUM_LIME_FEATURES)
        lime_results['fake'].append(exp.as_list())

    # Process and average explanations
    def process_explanations(explanations):
        feature_impacts = {}
        for exp in explanations:
            for feature, impact in exp:
                if feature not in feature_impacts:
                    feature_impacts[feature] = []
                feature_impacts[feature].append(impact)

        avg_impacts = {k: np.mean(v) for k, v in feature_impacts.items()}
        return sorted(avg_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    avg_real = process_explanations(lime_results['real'])
    avg_fake = process_explanations(lime_results['fake'])

    # Visualize results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Real audio explanations
    features_real = [x[0] for x in avg_real[:NUM_LIME_FEATURES]]
    impacts_real = [x[1] for x in avg_real[:NUM_LIME_FEATURES]]
    colors_real = ['green' if x > 0 else 'red' for x in impacts_real]
    ax1.barh(features_real, impacts_real, color=colors_real)
    ax1.set_title('Top Features for Real Audio Classification')
    ax1.set_xlabel('Impact on Prediction')
    ax1.axvline(0, color='black', linestyle='--')

    # Fake audio explanations
    features_fake = [x[0] for x in avg_fake[:NUM_LIME_FEATURES]]
    impacts_fake = [x[1] for x in avg_fake[:NUM_LIME_FEATURES]]
    colors_fake = ['green' if x > 0 else 'red' for x in impacts_fake]
    ax2.barh(features_fake, impacts_fake, color=colors_fake)
    ax2.set_title('Top Features for Fake Audio Classification')
    ax2.set_xlabel('Impact on Prediction')
    ax2.axvline(0, color='black', linestyle='--')

    plt.tight_layout()
    st.pyplot(fig)

    return fig, avg_real, avg_fake


def create_feature_explanation_page(pdf, feature_name, lime_real=None, lime_fake=None):
    """Creates a PDF page with technical explanations for a specific audio feature"""

    # Technical explanations for each feature
    explanations = {
        'waveform': {
            'title': "Waveform Differences: Technical Analysis",
            'sections': [
                ("Biological Basis of Real Speech",
                 ["- Vocal fold vibrations create natural amplitude modulation",
                  "- Respiratory patterns affect speech rhythm and dynamics",
                  "- Articulator movements (lips, tongue) create smooth transitions"]),

                ("Synthesis Artifacts in Fake Audio",
                 ["- Vocoders often produce overly regular amplitude envelopes",
                  "- Neural networks may generate artificial periodic patterns",
                  "- Lack of natural micro-variations in synthesized speech"]),

                ("Recording Characteristics",
                 ["- Real recordings capture room acoustics and ambient noise",
                  "- Microphone frequency response affects waveform shape",
                  "- Natural mouth-to-mic distance variations create dynamics"]),

                ("Key Differentiators (LIME Analysis)" if lime_real else "",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_real[:3]] if lime_real else [])
            ]
        },
        'spectrogram': {
            'title': "Spectrogram Differences: Technical Analysis",
            'sections': [
                ("Vocal Tract Physics",
                 ["- Formant frequencies (F1-F4) reflect vocal tract shape",
                  "- Natural spectral tilt from voice source characteristics",
                  "- Proper harmonic-to-noise ratio in voiced sounds"]),

                ("Spectral Artifacts in Synthesis",
                 ["- Phase discontinuities in neural vocoders",
                  "- Overly regular harmonic spacing in synthetic speech",
                  "- Missing high-frequency components above 8kHz"]),

                ("Phonetic Accuracy",
                 ["- Real speech maintains proper formant transitions",
                  "- Accurate representation of fricatives and plosives",
                  "- Natural spectral evolution during coarticulation"]),

                ("Key Differentiators (LIME Analysis)" if lime_real else "",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_real[:3]] if lime_real else [])
            ]
        },
        'mfcc': {
            'title': "MFCC Differences: Technical Analysis",
            'sections': [
                ("Natural Speech Characteristics",
                 ["- Complex spectral envelope from vocal tract filtering",
                  "- Time-varying cepstral features from articulation",
                  "- Proper representation of voice quality parameters"]),

                ("Synthesis Limitations",
                 ["- Mel-spectrogram inversion loses spectral detail",
                  "- Reduced dimensionality in feature extraction",
                  "- Over-smoothing of spectral features"]),

                ("Dynamic Speech Features",
                 ["- Natural delta and delta-delta coefficients",
                  "- Authentic temporal evolution of spectral features",
                  "- Proper representation of phonetic transitions"]),

                ("Key Differentiators (LIME Analysis)" if lime_real else "",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_real[:3]] if lime_real else [])
            ]
        },
        'chroma': {
            'title': "Chromagram Differences: Technical Analysis",
            'sections': [
                ("Prosodic Features of Natural Speech",
                 ["- Micro-intonation from neural control of pitch",
                  "- Emotion-driven pitch variations",
                  "- Natural vibrato in sustained vowels"]),

                ("Pitch Synthesis Artifacts",
                 ["- Overly stable F0 from pitch predictors",
                  "- Quantization effects in pitch contours",
                  "- Missing natural pitch fluctuations"]),

                ("Harmonic Relationships",
                 ["- Proper harmonic spacing in real voice",
                  "- Natural spectral tilt in harmonic structure",
                  "- Authentic harmonic-to-noise ratios"]),

                ("Key Differentiators (LIME Analysis)" if lime_real else "",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_real[:3]] if lime_real else [])
            ]
        },
        'zcr': {
            'title': "Zero Crossing Rate Differences: Technical Analysis",
            'sections': [
                ("Phonetic Patterns in Real Speech",
                 ["- Natural voiced/unvoiced transitions",
                  "- Proper ZCR values for different phoneme classes",
                  "- Gradual onsets and offsets in speech sounds"]),

                ("Synthetic Artifacts",
                 ["- Abrupt voicing state changes",
                  "- Incorrect ZCR for fricatives and plosives",
                  "- Artificially regular temporal patterns"]),

                ("Recording Artifacts",
                 ["- Natural noise floor affects ZCR values",
                  "- Microphone characteristics influence ZCR",
                  "- Ambient noise contributes to ZCR variations"]),

                ("Key Differentiators (LIME Analysis)" if lime_real else "",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_real[:3]] if lime_real else [])
            ]
        },
        'lime': {
            'title': "LIME Feature Importance Analysis",
            'sections': [
                ("Most Important Features for Real Audio",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_real[:5]] if lime_real else []),

                ("Most Important Features for Fake Audio",
                 [f"- {feat}: {impact:.2f}" for feat, impact in lime_fake[:5]] if lime_fake else []),

                ("Interpretation Guidelines",
                 ["- Positive values: Feature supports REAL classification",
                  "- Negative values: Feature supports FAKE classification",
                  "- Magnitude indicates relative importance"])
            ]
        }
    }

    # Create explanation page
    plt.figure(figsize=(8.27, 11.69))  # A4 size
    plt.axis('off')

    # Title
    plt.text(0.5, 0.9, explanations[feature_name]['title'],
             ha='center', va='center', fontsize=14, fontweight='bold')

    # Content sections
    y_pos = 0.8
    for section_title, section_items in explanations[feature_name]['sections']:
        if not section_title:  # Skip empty sections
            continue

        # Section header
        plt.text(0.1, y_pos, section_title + ":",
                 ha='left', va='top', fontsize=12, fontweight='bold')
        y_pos -= 0.05

        # Section items
        for item in section_items:
            wrapped_lines = wrap(item, width=90)
            for line in wrapped_lines:
                plt.text(0.15, y_pos, line,
                         ha='left', va='top', fontsize=11)
                y_pos -= 0.04
            y_pos -= 0.01  # Extra space between items

        y_pos -= 0.03  # Space between sections

    # Footer
    plt.text(0.5, 0.05, "Audio Authenticity Analysis Report | Generated: " +
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             ha='center', va='center', fontsize=9, alpha=0.7)

    pdf.savefig(bbox_inches='tight')
    plt.close()


def generate_pdf_report(figures, feature_order, stats_df, lime_real=None, lime_fake=None):
    """Generates a comprehensive PDF report with visualizations and explanations"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pdf_path = os.path.join(OUTPUT_DIR, f"audio_analysis_report_{timestamp}.pdf")

    with PdfPages(pdf_path) as pdf:
        # Title page
        plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.7, "Audio Authenticity Analysis Report",
                 ha='center', va='center', fontsize=18, fontweight='bold')
        plt.text(0.5, 0.6, "Comparative Analysis of Real vs Synthetic Speech",
                 ha='center', va='center', fontsize=14)
        plt.text(0.5, 0.5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ha='center', va='center', fontsize=12)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Summary statistics page
        plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.9, "Feature Statistics Summary",
                 ha='center', va='center', fontsize=14, fontweight='bold')

        # Create table
        cell_text = []
        for row in stats_df.values:
            cell_text.append([str(x) for x in row])

        table = plt.table(cellText=cell_text,
                          colLabels=stats_df.columns,
                          rowLabels=stats_df.index,
                          loc='center',
                          cellLoc='center',
                          colWidths=[0.2] * len(stats_df.columns))

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        plt.text(0.5, 0.1, "Key: 1 = Real Audio | 0 = Fake Audio",
                 ha='center', va='center', fontsize=10)
        pdf.savefig(bbox_inches='tight')
        plt.close()

        # Add LIME analysis page if available
        if lime_real and lime_fake:
            create_feature_explanation_page(pdf, 'lime', lime_real, lime_fake)

        # Add visualizations and explanations
        for fig, feature in zip(figures, feature_order):
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            create_feature_explanation_page(pdf, feature, lime_real, lime_fake)

    return pdf_path


def analyze_audio_features(real_files, fake_files):
    """Main analysis function that processes audio files and generates comparisons"""
    st.title("🔍 Audio Authenticity Analyzer")
    st.subheader("Comparative Analysis of Real vs Synthetic Speech")

    # Initialize accumulators
    features = {
        'waveform': {'real': [], 'fake': []},
        'spectrogram': {'real': [], 'fake': []},
        'mfcc': {'real': [], 'fake': []},
        'chroma': {'real': [], 'fake': []},
        'zcr': {'real': [], 'fake': []},
        'statistics': {'real': [], 'fake': []}
    }

    # Prepare data for LIME analysis
    lime_features = []
    lime_labels = []

    # Process files
    def process_files(files, label):
        for file in tqdm(files[:MAX_FILES_PER_CLASS], desc=f"Processing {label} files"):
            try:
                audio, _ = librosa.load(file, sr=SR)
                target_samples = SR * STANDARD_DURATION

                # Standardize length
                if len(audio) > target_samples:
                    audio = audio[:target_samples]
                else:
                    audio = np.pad(audio, (0, target_samples - len(audio)), 'constant')

                # Waveform
                features['waveform'][label].append(audio)

                # Spectrogram
                S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
                features['spectrogram'][label].append(librosa.amplitude_to_db(np.abs(S), ref=np.max))

                # MFCCs
                mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=20,
                                            n_fft=N_FFT, hop_length=HOP_LENGTH)
                features['mfcc'][label].append(mfcc)

                # Chroma
                chroma = librosa.feature.chroma_stft(y=audio, sr=SR,
                                                     n_fft=N_FFT, hop_length=HOP_LENGTH)
                features['chroma'][label].append(chroma)

                # ZCR
                zcr = librosa.feature.zero_crossing_rate(audio,
                                                         frame_length=N_FFT,
                                                         hop_length=HOP_LENGTH)
                features['zcr'][label].append(zcr[0])

                # Statistical features
                features['statistics'][label].append({
                    'rms': np.sqrt(np.mean(audio ** 2)),
                    'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=SR)),
                    'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SR)),
                    'spectral_flatness': np.mean(librosa.feature.spectral_flatness(y=audio)),
                    'zcr_mean': np.mean(zcr)
                })

                # Extract features for LIME
                lime_features.append(extract_features_for_lime(audio))
                lime_labels.append(1 if label == 'real' else 0)

            except Exception as e:
                st.warning(f"Skipped {file}: {str(e)}")

    # Process real and fake files
    with st.spinner("Analyzing audio files (this may take a few minutes)..."):
        process_files(real_files, 'real')
        process_files(fake_files, 'fake')

    # Calculate averages
    def get_average(feature, label):
        arrs = features[feature][label]
        if not arrs:
            return None
        if feature == 'waveform':
            return np.mean(arrs, axis=0)

        min_frames = min(a.shape[1] if len(a.shape) > 1 else len(a) for a in arrs)
        if len(arrs[0].shape) > 1:  # For 2D features
            return np.mean([a[:, :min_frames] for a in arrs], axis=0)
        else:  # For 1D features
            return np.mean([a[:min_frames] for a in arrs], axis=0)

    # Generate visualizations
    all_figures = []
    feature_order = []

    # ====================== WAVEFORM COMPARISON ======================
    st.markdown("---")
    st.subheader("1. Waveform Comparison")

    fig_wave = plt.figure(figsize=(12, 4))
    real_wave = get_average('waveform', 'real')
    fake_wave = get_average('waveform', 'fake')
    time = np.arange(len(real_wave)) / SR
    plt.plot(time, real_wave, label='Real Audio', color='blue', alpha=0.8)
    plt.plot(time, fake_wave, label='Synthetic Audio', color='red', alpha=0.6)
    plt.title(f'Average Waveform (First {STANDARD_DURATION} seconds)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    st.pyplot(fig_wave)
    all_figures.append(fig_wave)
    feature_order.append('waveform')

    with st.expander("🔍 Waveform Interpretation"):
        st.markdown("""
        **Key Differences:**
        - **Natural Variations**: Real speech shows organic amplitude fluctuations from vocal fold vibrations
        - **Synthetic Patterns**: Fake audio often has unnaturally regular amplitude envelopes
        - **Transients**: Real speech has smoother attack/decay characteristics

        **Technical Indicators:**
        - Look for micro-variations in real audio (biological imperfection)
        - Watch for repetitive patterns in synthetic audio (algorithmic generation)
        - Check for abrupt transitions in fake audio (vocoder artifacts)
        """)

    # ====================== SPECTROGRAM COMPARISON ======================
    st.markdown("---")
    st.subheader("2. Spectrogram Comparison")

    fig_spec, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    real_spec = get_average('spectrogram', 'real')
    fake_spec = get_average('spectrogram', 'fake')

    if real_spec is not None:
        img = librosa.display.specshow(real_spec, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='log', ax=ax1,
                                       cmap='viridis')
        ax1.set_title('Real Audio - Average Spectrogram')
        fig_spec.colorbar(img, ax=ax1, format="%+2.0f dB")

    if fake_spec is not None:
        img = librosa.display.specshow(fake_spec, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='log', ax=ax2,
                                       cmap='viridis')
        ax2.set_title('Synthetic Audio - Average Spectrogram')
        fig_spec.colorbar(img, ax=ax2, format="%+2.0f dB")

    plt.tight_layout()
    st.pyplot(fig_spec)
    all_figures.append(fig_spec)
    feature_order.append('spectrogram')

    with st.expander("🔍 Spectrogram Interpretation"):
        st.markdown("""
        **Key Differences:**
        - **Formant Structure**: Real vowels show clear, biologically-determined formant bands
        - **Harmonic Patterns**: Fake audio often has unnaturally regular harmonics
        - **High Frequencies**: Synthetic speech frequently lacks high-frequency detail

        **Technical Indicators:**
        - Check for blurred harmonics in fake audio (vocoder artifacts)
        - Look for missing high-frequency components (>8kHz) in synthetic speech
        - Notice unnatural transitions between phonemes
        - Observe spectral tilt differences (real speech has natural roll-off)
        """)

    # ====================== MFCC COMPARISON ======================
    st.markdown("---")
    st.subheader("3. MFCC Comparison")

    fig_mfcc, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    real_mfcc = get_average('mfcc', 'real')
    fake_mfcc = get_average('mfcc', 'fake')

    if real_mfcc is not None:
        img = librosa.display.specshow(real_mfcc, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', ax=ax1, cmap='magma')
        ax1.set_title('Real Audio - Average MFCCs')
        fig_mfcc.colorbar(img, ax=ax1)

    if fake_mfcc is not None:
        img = librosa.display.specshow(fake_mfcc, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', ax=ax2, cmap='magma')
        ax2.set_title('Synthetic Audio - Average MFCCs')
        fig_mfcc.colorbar(img, ax=ax2)

    plt.tight_layout()
    st.pyplot(fig_mfcc)
    all_figures.append(fig_mfcc)
    feature_order.append('mfcc')

    with st.expander("🔍 MFCC Interpretation"):
        st.markdown("""
        **Key Differences:**
        - **Spectral Envelope**: Real speech shows complex, time-varying spectral shapes
        - **Smoothness**: Fake audio often has unnaturally smooth coefficient trajectories
        - **Dynamic Range**: Real speech typically has wider dynamic range in coefficients

        **Technical Indicators:**
        - Look for overly smooth coefficient patterns (mel-spectrogram inversion artifacts)
        - Check for reduced dimensionality in synthetic MFCCs
        - Notice differences in temporal evolution of features
        - Observe delta and delta-delta coefficient variations
        """)

    # ====================== CHROMA COMPARISON ======================
    st.markdown("---")
    st.subheader("4. Chromagram Comparison")

    fig_chroma, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    real_chroma = get_average('chroma', 'real')
    fake_chroma = get_average('chroma', 'fake')

    if real_chroma is not None:
        img = librosa.display.specshow(real_chroma, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='chroma', ax=ax1,
                                       cmap='coolwarm')
        ax1.set_title('Real Audio - Average Chromagram')
        fig_chroma.colorbar(img, ax=ax1)

    if fake_chroma is not None:
        img = librosa.display.specshow(fake_chroma, sr=SR, hop_length=HOP_LENGTH,
                                       x_axis='time', y_axis='chroma', ax=ax2,
                                       cmap='coolwarm')
        ax2.set_title('Synthetic Audio - Average Chromagram')
        fig_chroma.colorbar(img, ax=ax2)

    plt.tight_layout()
    st.pyplot(fig_chroma)
    all_figures.append(fig_chroma)
    feature_order.append('chroma')

    with st.expander("🔍 Chromagram Interpretation"):
        st.markdown("""
        **Key Differences:**
        - **Pitch Variation**: Real speech has natural micro-intonation
        - **Stability**: Fake audio often shows unnaturally stable pitch
        - **Harmonics**: Real voice maintains proper harmonic relationships

        **Technical Indicators:**
        - Look for mechanical pitch patterns (pitch predictor artifacts)
        - Check for missing natural vibrato in sustained vowels
        - Notice unusual harmonic spacing in synthetic audio
        - Observe pitch contour smoothness and naturalness
        """)

    # ====================== ZCR COMPARISON ======================
    st.markdown("---")
    st.subheader("5. Zero Crossing Rate Comparison")

    fig_zcr = plt.figure(figsize=(12, 4))
    real_zcr = get_average('zcr', 'real')
    fake_zcr = get_average('zcr', 'fake')

    if real_zcr is not None and fake_zcr is not None:
        t = librosa.frames_to_time(range(len(real_zcr)),
                                   hop_length=HOP_LENGTH, sr=SR)
        plt.plot(t, real_zcr, label='Real Audio', color='blue', alpha=0.8)
        plt.plot(t, fake_zcr, label='Synthetic Audio', color='red', alpha=0.6)
        plt.title('Average Zero Crossing Rate')
        plt.xlabel('Time (s)')
        plt.ylabel('ZCR')
        plt.legend()
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_zcr)
    all_figures.append(fig_zcr)
    feature_order.append('zcr')

    with st.expander("🔍 ZCR Interpretation"):
        st.markdown("""
        **Key Differences:**
        - **Voicing Patterns**: Real speech shows natural transitions between voiced/unvoiced sounds
        - **Consonants**: Fake audio often misrepresents fricative ZCR values
        - **Temporal Patterns**: Synthetic speech may show artificially regular ZCR

        **Technical Indicators:**
        - Look for abrupt voicing state changes in fake audio
        - Check for incorrect ZCR values during fricatives (s, sh sounds)
        - Notice unnatural temporal distribution
        - Observe ZCR consistency during unvoiced segments
        """)

    # ====================== LIME FEATURE IMPORTANCE ======================
    st.markdown("---")
    st.subheader("6. LIME Feature Importance Analysis")

    # Prepare feature names for LIME
    lime_feature_names = [
        'wave_mean', 'wave_std', 'wave_max', 'wave_min',
        'spec_mean', 'spec_std', 'spec_max', 'spec_min',
        *[f'mfcc_mean_{i}' for i in range(20)],
        *[f'mfcc_std_{i}' for i in range(20)],
        'chroma_mean', 'chroma_std',
        'zcr_mean', 'zcr_std',
        'spectral_centroid', 'spectral_bandwidth', 'spectral_flatness'
    ]

    # Run LIME analysis
    X = np.array(lime_features)
    y = np.array(lime_labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lime_fig, lime_real, lime_fake = create_lime_explanations(
        X_train, X_test, y_train, y_test, lime_feature_names
    )
    all_figures.append(lime_fig)
    feature_order.append('lime')

    with st.expander("🔍 LIME Interpretation"):
        st.markdown("""
        **How to Interpret:**
        - **Positive Values**: Features that support REAL classification when present
        - **Negative Values**: Features that support FAKE classification when present
        - **Magnitude**: Relative importance of each feature

        **Common Findings:**
        - Real audio is often characterized by:
          - Natural spectral variations (MFCC dynamics)
          - Proper harmonic structure (chroma features)
          - Authentic temporal patterns (ZCR variations)

        - Fake audio often shows:
          - Overly smooth spectral features
          - Artificial periodicity
          - Inconsistent temporal patterns
        """)

    # ====================== STATISTICAL ANALYSIS ======================
    st.markdown("---")
    st.subheader("7. Statistical Feature Analysis")

    # Prepare statistical data
    real_stats = pd.DataFrame(features['statistics']['real'])
    fake_stats = pd.DataFrame(features['statistics']['fake'])

    stats_df = pd.concat([
        real_stats.mean().rename('Real Audio'),
        fake_stats.mean().rename('Synthetic Audio'),
        (real_stats.mean() - fake_stats.mean()).rename('Difference')
    ], axis=1).T

    st.dataframe(stats_df.style.format("{:.4f}").background_gradient(cmap='Blues'))

    with st.expander("🔍 Statistical Interpretation"):
        st.markdown("""
        **Key Metrics:**
        - **RMS Energy**: Real speech typically has more natural energy distribution
        - **Spectral Centroid**: Indicates brightness of sound (real speech often brighter)
        - **Spectral Bandwidth**: Real speech shows more natural bandwidth variations
        - **Spectral Flatness**: Measures noisiness (real unvoiced sounds are noisier)
        - **ZCR Mean**: Real speech has proper voiced/unvoiced transitions

        **Expected Patterns:**
        - Real audio usually has higher spectral centroid and bandwidth
        - Fake audio often shows lower RMS energy variation
        - Spectral flatness differences reveal voicing artifacts
        """)

    # Generate PDF report
    pdf_path = generate_pdf_report(all_figures, feature_order, stats_df, lime_real, lime_fake)

    st.success("Analysis complete!")
    st.markdown(f"**Report generated:** `{pdf_path}`")

    # Add download button
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="📥 Download Full Report (PDF)",
            data=f,
            file_name="audio_authenticity_report.pdf",
            mime="application/pdf"
        )


def main():
    st.set_page_config(
        layout="wide",
        page_title="Audio Authenticity Analyzer",
        page_icon="🔍"
    )

    # Load audio files
    real_files = glob(os.path.join(DATA_DIR, "real", "*.wav"))
    fake_files = glob(os.path.join(DATA_DIR, "fake", "*.wav"))

    if not real_files or not fake_files:
        st.error("Error: Could not find audio files. Please check your DATA_DIR path.")
        st.stop()

    analyze_audio_features(real_files, fake_files)


if __name__ == "__main__":
    main()