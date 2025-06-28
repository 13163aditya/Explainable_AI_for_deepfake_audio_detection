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
from lime import lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuration
SR = 22050  # Sample rate
N_FFT = 2048  # FFT window size
HOP_LENGTH = 512  # Number of samples between frames
DATA_DIR = "./fake-real-audio/"  # Update this path to your dataset
MAX_FILES_PER_CLASS = 200  # Maximum files to process per class
STANDARD_DURATION = 3  # Standard audio duration in seconds
SAMPLES_PER_CLIP = SR * STANDARD_DURATION
OUTPUT_DIR = "audio_analysis_reports"  # Output directory for reports
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Create output directory if it doesn't exist


def extract_features(audio, sr=SR):
    """Extract all features for a single audio file"""
    features = []

    # Waveform stats
    features.extend([np.mean(audio), np.std(audio), np.max(audio)])

    # Spectrogram stats
    S = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    features.extend([np.mean(S_db), np.std(S_db)])

    # MFCC stats
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Chroma stats
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend([np.mean(chroma), np.std(chroma)])

    # ZCR stats
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.extend([np.mean(zcr), np.std(zcr)])

    return np.array(features)


def get_feature_names():
    """Generate feature names for LIME explanation"""
    feature_names = []

    # Waveform features
    feature_names.extend(['wave_mean', 'wave_std', 'wave_max'])

    # Spectrogram features
    feature_names.extend(['spec_mean', 'spec_std'])

    # MFCC features (mean and std for each coefficient)
    for i in range(20):
        feature_names.append(f'mfcc{i}_mean')
    for i in range(20):
        feature_names.append(f'mfcc{i}_std')

    # Chroma features
    feature_names.extend(['chroma_mean', 'chroma_std'])

    # ZCR features
    feature_names.extend(['zcr_mean', 'zcr_std'])

    return feature_names


def get_lime_explanation(all_features, labels):
    """Generate and return LIME explanation with proper error handling"""
    try:
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(labels)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        # Create explainer
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=get_feature_names(),
            class_names=['Fake', 'Real'],
            mode='classification',
            discretize_continuous=False
        )

        # Find suitable instance for explanation
        explained_instance = None
        for instance in X_test:
            if len(np.unique(clf.predict_proba([instance]))) > 1:
                explained_instance = instance
                break

        if explained_instance is None:
            explained_instance = X_test[0]

        # Generate explanation
        exp = explainer.explain_instance(
            explained_instance,
            clf.predict_proba,
            num_features=10,
            top_labels=1
        )

        return exp

    except Exception as e:
        st.error(f"LIME explanation failed: {str(e)}")
        return None


def show_lime_explanation(exp):
    """Display LIME explanation in Streamlit"""
    if exp is None:
        st.warning("No LIME explanation available")
        return

    # Create figure
    fig = plt.figure(figsize=(10, 6))
    exp.as_pyplot_figure()
    plt.title("Top Influential Features for Detection")
    plt.tight_layout()
    st.pyplot(fig)

    # Show textual explanation
    with st.expander("Detailed Feature Explanations"):
        for label in exp.available_labels():
            st.subheader(f"For {'Real' if label == 1 else 'Fake'} Audio:")
            st.write(pd.DataFrame(
                exp.as_list(label=label),
                columns=["Feature", "Importance"]
            ).style.background_gradient(cmap="coolwarm"))


def create_feature_explanation_page(pdf, feature_name):
    """Creates a PDF page with technical explanations for a specific audio feature"""
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
                  "- Natural mouth-to-mic distance variations create dynamics"])
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
                  "- Natural spectral evolution during coarticulation"])
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
                  "- Proper representation of phonetic transitions"])
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
                  "- Authentic harmonic-to-noise ratios"])
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
                  "- Ambient noise contributes to ZCR variations"])
            ]
        },
        'lime': {
            'title': "LIME Feature Importance Analysis",
            'sections': [
                ("Understanding LIME Explanations",
                 ["- Explains individual predictions by perturbing inputs",
                  "- Shows which features most influenced classification",
                  "- Positive weights favor REAL classification",
                  "- Negative weights favor FAKE classification"]),

                ("Key Findings in Audio Authenticity",
                 ["- MFCC variations are often strongest indicators",
                  "- Spectral statistics reveal vocoder artifacts",
                  "- Waveform statistics capture natural speech patterns"]),

                ("Using These Insights",
                 ["- Focus on features with highest absolute weights",
                  "- Compare explanations across multiple samples",
                  "- Look for consistent patterns in important features"])
            ]
        }
    }

    # Create explanation page
    plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')

    # Title
    plt.text(0.5, 0.9, explanations[feature_name]['title'],
             ha='center', va='center', fontsize=14, fontweight='bold')

    # Content sections
    y_pos = 0.8
    for section_title, section_items in explanations[feature_name]['sections']:
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
            y_pos -= 0.01

        y_pos -= 0.03

    # Footer
    plt.text(0.5, 0.05, "Audio Authenticity Analysis Report | Generated: " +
             datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
             ha='center', va='center', fontsize=9, alpha=0.7)

    pdf.savefig(bbox_inches='tight')
    plt.close()


def generate_pdf_report(figures, feature_order, stats_df, lime_exp=None):
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

        # Add visualizations and explanations
        for fig, feature in zip(figures, feature_order):
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            create_feature_explanation_page(pdf, feature)

        # Add LIME explanation if available
        if lime_exp is not None:
            plt.figure(figsize=(8.27, 11.69))
            plt.axis('off')

            # Title
            plt.text(0.5, 0.9, "LIME Feature Importance Analysis",
                     ha='center', va='center', fontsize=14, fontweight='bold')

            # Explanation
            y_pos = 0.85
            for label in lime_exp.available_labels():
                plt.text(0.1, y_pos, f"Top Features for {'Real' if label == 1 else 'Fake'} Audio:",
                         ha='left', va='top', fontsize=12, fontweight='bold')
                y_pos -= 0.04

                for feature, weight in lime_exp.as_list(label=label):
                    plt.text(0.15, y_pos, f"{feature}: {weight:.3f}",
                             ha='left', va='top', fontsize=11)
                    y_pos -= 0.04
                y_pos -= 0.02

            # Interpretation guide
            y_pos -= 0.05
            plt.text(0.1, y_pos, "Interpretation Guide:",
                     ha='left', va='top', fontsize=12, fontweight='bold')
            y_pos -= 0.04
            plt.text(0.15, y_pos, "Positive weights indicate features more common in REAL audio",
                     ha='left', va='top', fontsize=11)
            y_pos -= 0.04
            plt.text(0.15, y_pos, "Negative weights indicate features more common in FAKE audio",
                     ha='left', va='top', fontsize=11)

            pdf.savefig(bbox_inches='tight')
            plt.close()

            create_feature_explanation_page(pdf, 'lime')

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

    # For LIME analysis
    all_features = []
    labels = []

    # Process files
    def process_files(files, label):
        for file in tqdm(files[:MAX_FILES_PER_CLASS], desc=f"Processing {label} files"):
            try:
                audio, _ = librosa.load(file, sr=SR)
                audio = librosa.util.fix_length(audio, size=SAMPLES_PER_CLIP)

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

                # For LIME
                features_vector = extract_features(audio)
                all_features.append(features_vector)
                labels.append(1 if label == 'real' else 0)

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
        """)

    # ====================== STATISTICAL ANALYSIS ======================
    st.markdown("---")
    st.subheader("6. Statistical Feature Analysis")

    # Prepare statistical data
    real_stats = pd.DataFrame(features['statistics']['real'])
    fake_stats = pd.DataFrame(features['statistics']['fake'])

    stats_df = pd.concat([
        real_stats.mean().rename('Real Audio'),
        fake_stats.mean().rename('Synthetic Audio')
    ], axis=1).T

    st.dataframe(stats_df.style.format("{:.4f}").background_gradient(cmap='Blues'))

    # ====================== LIME EXPLANATION ======================
    st.markdown("---")
    st.subheader("7. LIME Feature Importance Explanation")

    lime_exp = None
    if len(all_features) > 10:  # Minimum samples needed
        with st.spinner("Generating LIME explanation..."):
            lime_exp = get_lime_explanation(all_features, labels)
            show_lime_explanation(lime_exp)

            with st.expander("🔍 How to interpret LIME results"):
                st.markdown("""
                - **Positive values**: Features more common in REAL audio
                - **Negative values**: Features more common in FAKE audio
                - **Absolute magnitude**: Strength of the feature's influence

                **Key features to examine:**
                1. MFCC coefficients (especially 1-5)
                2. Spectral statistics (mean, std)
                3. Zero-crossing rate
                4. Chroma features
                """)
    else:
        st.warning("Insufficient data for LIME explanation (need at least 10 samples)")

    # Generate PDF report
    pdf_path = generate_pdf_report(all_figures, feature_order, stats_df, lime_exp)

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