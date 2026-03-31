"""
app.py - Streamlit Web App for Voice Language Prediction
Records voice input, extracts features, and predicts the language
using the trained CNN+LSTM deep learning model.
"""

import streamlit as st
import numpy as np
import librosa
import joblib
import json
import os
import tempfile
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="🎙️ Voice Language Predictor",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    .prediction-card h2 {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .prediction-card .confidence {
        font-size: 3rem;
        font-weight: bold;
    }
    .prediction-card .language {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: black;
    }
    .lang-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .pipeline-step {
        background: #ffffff;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: black;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 1.5rem;
        color: #667eea;
        margin: 0.2rem 0;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #888;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================
@st.cache_resource
def load_model_artifacts():
    """Load the trained model, label encoder, scaler, and config."""
    artifacts = {}
    
    try:
        # Load config
        with open('model_config.json', 'r') as f:
            artifacts['config'] = json.load(f)
        
        # Load model
        artifacts['model'] = tf.keras.models.load_model('model/language_model.keras')
        
        # Load label encoder
        artifacts['label_encoder'] = joblib.load('model/label_encoder.pkl')
        
        # Load scaler
        artifacts['scaler'] = joblib.load('model/scaler.pkl')
        
        artifacts['loaded'] = True
        
    except FileNotFoundError as e:
        artifacts['loaded'] = False
        artifacts['error'] = str(e)
    
    return artifacts


def extract_features(audio, sr, config):
    """
    Extract features from audio signal (same as training).
    """
    try:
        duration = config['duration']
        n_mfcc = config['n_mfcc']
        n_mels = config['n_mels']
        n_chroma = config['n_chroma']
        hop_length = config['hop_length']
        n_fft = config['n_fft']
        max_pad_len = config['max_pad_len']
        
        # Ensure minimum length
        if len(audio) < sr * 0.5:
            return None
        
        # Pad if shorter than duration
        max_len = sr * duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]
        
        # Pre-emphasis
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 1. MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc,
            n_fft=n_fft, hop_length=hop_length
        )
        
        # 2. MFCC Delta
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 3. MFCC Delta2
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 4. Chroma
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_chroma=n_chroma,
            n_fft=n_fft, hop_length=hop_length
        )
        
        # 5. Mel spectrogram (reduced)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels,
            n_fft=n_fft, hop_length=hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_reduced = mel_db[::6, :]
        
        # 6. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        
        # 7. Tonnetz
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio), sr=sr
        )
        
        # 8. ZCR
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
        
        # 9. RMS
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)
        
        # 10. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )
        
        # 11. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=hop_length
        )
        
        # 12. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=hop_length
        )
        
        # Concatenate
        features = np.concatenate([
            mfcc, mfcc_delta, mfcc_delta2, chroma, mel_reduced,
            spectral_contrast, tonnetz, zcr, rms,
            spectral_centroid, spectral_bandwidth, spectral_rolloff
        ], axis=0)
        
        features = features.T
        
        # Pad/truncate
        if features.shape[0] < max_pad_len:
            pad_width = max_pad_len - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:max_pad_len, :]
        
        return features
        
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None


def predict_language(audio, sr, model, scaler, label_encoder, config):
    """
    Full prediction pipeline:
    Voice -> Feature Extraction -> Scaling -> Model Prediction -> Language
    """
    # Step 1: Resample to target sample rate
    target_sr = config['sample_rate']
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # Step 2: Extract features
    features = extract_features(audio, sr, config)
    if features is None:
        return None, None
    
    # Step 3: Scale features
    n_features = features.shape[1]
    features_flat = features.reshape(-1, n_features)
    features_scaled = scaler.transform(features_flat)
    features_scaled = features_scaled.reshape(1, features.shape[0], n_features)
    
    # Replace NaN/Inf
    features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Step 4: Predict
    prediction = model.predict(features_scaled, verbose=0)
    
    # Step 5: Get results
    predicted_class = np.argmax(prediction[0])
    confidence = prediction[0][predicted_class]
    predicted_language = label_encoder.inverse_transform([predicted_class])[0]
    
    # All probabilities
    all_probs = {}
    for i, lang in enumerate(label_encoder.classes_):
        all_probs[lang] = float(prediction[0][i])
    
    return predicted_language, all_probs


def get_audio_info(audio, sr):
    """Get basic audio information."""
    duration = len(audio) / sr
    rms = np.sqrt(np.mean(audio**2))
    return {
        'duration': duration,
        'sample_rate': sr,
        'samples': len(audio),
        'rms_energy': rms,
        'max_amplitude': np.max(np.abs(audio)),
        'is_silent': rms < 0.001
    }


def get_language_flag(lang):
    """Get emoji/flag for each language."""
    flags = {
        'Bengali': '🇮🇳 বাংলা',
        'Gujarati': '🇮🇳 ગુજરાતી',
        'Hindi': '🇮🇳 हिन्दी',
        'Kannada': '🇮🇳 ಕನ್ನಡ',
        'Malayalam': '🇮🇳 മലയാളം',
        'Marathi': '🇮🇳 मराठी',
        'Punjabi': '🇮🇳 ਪੰਜਾਬੀ',
        'Tamil': '🇮🇳 தமிழ்',
        'Telugu': '🇮🇳 తెలుగు',
        'Urdu': '🇮🇳 اردو'
    }
    return flags.get(lang, lang)


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ Voice Language Predictor</h1>
        <p>Upload your voice and our AI will predict the Indian language!</p>
        <p>Pipeline: 📁 File → 📊 Feature Extraction → 🧠 Deep Learning → 🌐 Language Prediction</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    artifacts = load_model_artifacts()
    
    if not artifacts.get('loaded', False):
        st.error("⚠️ Model files not found! Please run `train.py` first to train the model.")
        st.info("""
        **Steps to get started:**
        1. Download the Indian Language Speech Dataset from Kaggle
        2. Extract it into a `dataset/` folder with language subdirectories
        3. Run `python train.py` to train the model
        4. Then run `streamlit run app.py`
        """)
        
        st.code("""
# Terminal commands:
pip install -r requirements.txt
python train.py
streamlit run app.py
        """, language="bash")
        return
    
    model = artifacts['model']
    label_encoder = artifacts['label_encoder']
    scaler = artifacts['scaler']
    config = artifacts['config']
    
    # ============================================================
    # SIDEBAR
    # ============================================================
    with st.sidebar:
        st.markdown("## 📋 Model Information")
        
        st.markdown(f"""
        <div class="info-card">
            <b>🎯 Model Accuracy:</b> {config['test_accuracy']*100:.2f}%<br>
            <b>🏗️ Architecture:</b> CNN + BiLSTM<br>
            <b>📊 Features:</b> {config['n_features']} per frame<br>
            <b>⏱️ Audio Duration:</b> {config['duration']}s<br>
            <b>🔊 Sample Rate:</b> {config['sample_rate']} Hz
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🌐 Supported Languages")
        langs_html = ""
        for lang in config['languages']:
            langs_html += f'<span class="lang-badge">{get_language_flag(lang)}</span> '
        st.markdown(langs_html, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🔄 Processing Pipeline")
        
        steps = [
            ("📁", "Voice Input", "Upload audio file"),
            ("📊", "Feature Extraction", "MFCC, Chroma, Mel, etc."),
            ("📏", "Normalization", "StandardScaler transform"),
            ("🧠", "CNN + BiLSTM", "Deep Learning prediction"),
            ("🌐", "Language Output", "Top prediction + confidence"),
        ]
        
        for emoji, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
                {emoji} <b>{title}</b><br>
                <small style="color: #666;">{desc}</small>
            </div>
            <div class="pipeline-arrow">⬇️</div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Feature Details")
        st.markdown("""
        - **MFCC** (40 coefficients)
        - **MFCC Delta** (velocity)
        - **MFCC Delta²** (acceleration)
        - **Chroma** (12 pitch classes)
        - **Mel Spectrogram** (compressed)
        - **Spectral Contrast** (7 bands)
        - **Tonnetz** (6 dimensions)
        - **ZCR** (Zero Crossing Rate)
        - **RMS Energy**
        - **Spectral Centroid**
        - **Spectral Bandwidth**
        - **Spectral Rolloff**
        """)
        
        # Display training plots if available
        if os.path.exists('training_history.png'):
            st.markdown("---")
            st.markdown("### 📈 Training History")
            st.image('training_history.png', use_container_width=True)
        
        if os.path.exists('confusion_matrix.png'):
            st.markdown("### 🔢 Confusion Matrix")
            st.image('confusion_matrix.png', use_container_width=True)
    
    # ============================================================
    # MAIN CONTENT
    # ============================================================
    st.markdown("## 📁 Upload Your Voice")
    
    audio_data = None
    audio_sr = None
    
    st.markdown("""
    <div class="info-card">
        📁 Upload an audio file in <b>WAV, MP3, FLAC, OGG, or M4A</b> format.
        The audio should contain speech in one of the supported Indian languages.
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
        help="Upload an audio file containing speech in an Indian language"
    )
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Save to temp file and load with librosa
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        try:
            audio_data, audio_sr = librosa.load(
                tmp_path, sr=config['sample_rate']
            )
        except Exception as e:
            st.error(f"Error loading audio file: {e}")
        finally:
            os.unlink(tmp_path)
    
    # ============================================================
    # PREDICTION
    # ============================================================
    if audio_data is not None and len(audio_data) > 0:
        st.markdown("---")
        st.markdown("## 🔮 Prediction Results")
        
        # Audio info
        audio_info = get_audio_info(audio_data, audio_sr)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Duration", f"{audio_info['duration']:.2f}s")
        with col2:
            st.metric("🔊 Sample Rate", f"{audio_info['sample_rate']} Hz")
        with col3:
            st.metric("📊 Samples", f"{audio_info['samples']:,}")
        with col4:
            st.metric("🔈 RMS Energy", f"{audio_info['rms_energy']:.4f}")
        
        # Check if audio is too quiet
        if audio_info['is_silent']:
            st.warning("⚠️ The audio appears to be very quiet or silent. Results may not be accurate.")
        
        # Run prediction
        with st.spinner("🧠 Analyzing voice patterns and predicting language..."):
            
            # Progress simulation for UX
            progress_bar = st.progress(0)
            
            progress_bar.progress(20, text="📊 Extracting audio features...")
            predicted_language, all_probs = predict_language(
                audio_data, audio_sr, model, scaler, label_encoder, config
            )
            progress_bar.progress(60, text="🧠 Running deep learning model...")
            
            import time
            time.sleep(0.3)
            progress_bar.progress(90, text="📝 Generating results...")
            time.sleep(0.2)
            progress_bar.progress(100, text="✅ Complete!")
            time.sleep(0.3)
            progress_bar.empty()
        
        if predicted_language is not None and all_probs is not None:
            confidence = all_probs[predicted_language]
            
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-card">
                <h2>🎯 Predicted Language</h2>
                <div class="language">{get_language_flag(predicted_language)}</div>
                <div class="language">{predicted_language}</div>
                <div class="confidence">{confidence*100:.1f}%</div>
                <p>Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed results
            st.markdown("### 📊 Detailed Probability Distribution")
            
            # Sort probabilities
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            
            # Display as horizontal bars
            col1, col2 = st.columns([2, 1])
            
            with col1:
                for lang, prob in sorted_probs:
                    is_predicted = lang == predicted_language
                    
                    # Color based on prediction
                    if is_predicted:
                        bar_color = "🟢"
                    elif prob > 0.1:
                        bar_color = "🟡"
                    else:
                        bar_color = "🔴"
                    
                    col_a, col_b, col_c = st.columns([3, 6, 2])
                    with col_a:
                        prefix = "**→ " if is_predicted else "   "
                        suffix = " ✓**" if is_predicted else ""
                        st.markdown(f"{prefix}{bar_color} {lang}{suffix}")
                    with col_b:
                        st.progress(prob)
                    with col_c:
                        st.markdown(f"**{prob*100:.1f}%**")
            
            with col2:
                # Top 3 predictions
                st.markdown("#### 🏆 Top 3 Predictions")
                for i, (lang, prob) in enumerate(sorted_probs[:3]):
                    medals = ["🥇", "🥈", "🥉"]
                    st.markdown(f"""
                    <div style="background: #f8f9fa; padding: 0.8rem; 
                         border-radius: 8px; margin: 0.5rem 0; 
                         border-left: 3px solid {'#11998e' if i==0 else '#667eea' if i==1 else '#888'};">
                        {medals[i]} <b>{lang}</b><br>
                        <span style="font-size: 1.3rem; font-weight: bold;">{prob*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Confidence analysis
            st.markdown("---")
            st.markdown("### 📈 Confidence Analysis")
            
            top2 = sorted_probs[:2]
            margin = top2[0][1] - top2[1][1] if len(top2) > 1 else top2[0][1]
            
            if confidence > 0.8:
                st.success(f"✅ **High Confidence Prediction** - The model is {confidence*100:.1f}% "
                          f"confident this is **{predicted_language}** speech.")
            elif confidence > 0.5:
                st.warning(f"⚠️ **Moderate Confidence** - The model is {confidence*100:.1f}% "
                          f"confident. The speech likely is **{predicted_language}**, "
                          f"but consider the second prediction: **{top2[1][0]}** ({top2[1][1]*100:.1f}%).")
            else:
                st.error(f"❌ **Low Confidence** - The model is only {confidence*100:.1f}% "
                        f"confident. The audio quality may be poor or the language might "
                        f"not be well represented in training data.")
            
            st.markdown(f"**Decision Margin:** {margin*100:.1f}% "
                       f"(difference between top 2 predictions)")
            
        else:
            st.error("❌ Could not process the audio. Please try a different audio file.")
            st.info("Tips: Ensure the audio contains clear speech and is at least 1 second long.")
    
    # ============================================================
    # HOW IT WORKS SECTION
    # ============================================================
    st.markdown("---")
    st.markdown("## 🔍 How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="pipeline-step" style="text-align: center;">
            <h3>🎤</h3>
            <h4>Step 1: Voice Input</h4>
            <p style="font-size: 0.85rem;">Upload or record audio containing 
            speech in any supported Indian language</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-step" style="text-align: center;">
            <h3>📊</h3>
            <h4>Step 2: Feature Extraction</h4>
            <p style="font-size: 0.85rem;">Extract MFCC, Chroma, Mel Spectrogram, 
            and 9+ audio features using Librosa</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-step" style="text-align: center;">
            <h3>🧠</h3>
            <h4>Step 3: Deep Learning</h4>
            <p style="font-size: 0.85rem;">CNN layers extract local patterns, 
            BiLSTM captures sequential information</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="pipeline-step" style="text-align: center;">
            <h3>🌐</h3>
            <h4>Step 4: Prediction</h4>
            <p style="font-size: 0.85rem;">Softmax output gives probability 
            for each of the 10 Indian languages</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Model architecture info
    with st.expander("🏗️ Model Architecture Details"):
        st.markdown("""
        ### CNN + Bidirectional LSTM Architecture
        
        ```
        Input (216 timesteps × N features)
            │
            ├── Conv1D (64 filters, kernel=5) + BatchNorm + MaxPool + Dropout(0.3)
            ├── Conv1D (128 filters, kernel=5) + BatchNorm + MaxPool + Dropout(0.3)
            ├── Conv1D (256 filters, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
            ├── Conv1D (256 filters, kernel=3) + BatchNorm + MaxPool + Dropout(0.3)
            │
            ├── Bidirectional LSTM (128 units, return_sequences=True) + Dropout(0.4)
            ├── Bidirectional LSTM (64 units) + Dropout(0.4)
            │
            ├── Dense (256) + BatchNorm + Dropout(0.5)
            ├── Dense (128) + BatchNorm + Dropout(0.5)
            ├── Dense (64) + Dropout(0.3)
            │
            └── Dense (10, softmax) → Language Prediction
        ```
        
        **Why this architecture?**
        - **CNN layers**: Capture local spectral patterns (phonemes, formants)
        - **BiLSTM layers**: Model temporal dependencies in speech (prosody, rhythm)
        - **Batch Normalization**: Stabilize training, faster convergence
        - **Dropout**: Prevent overfitting
        - **Pre-emphasis**: Boost high-frequency speech components
        """)
    
    with st.expander("📊 Audio Features Explained"):
        st.markdown("""
        | Feature | Dimensions | Description |
        |---------|-----------|-------------|
        | **MFCC** | 40 | Mel-Frequency Cepstral Coefficients - vocal tract shape |
        | **MFCC Delta** | 40 | First derivative - speech dynamics |
        | **MFCC Delta²** | 40 | Second derivative - acceleration |
        | **Chroma** | 12 | Pitch class distribution |
        | **Mel Spectrogram** | ~21 | Compressed mel-scale frequency representation |
        | **Spectral Contrast** | 7 | Peak-valley differences in spectrum |
        | **Tonnetz** | 6 | Tonal centroid features |
        | **ZCR** | 1 | Zero Crossing Rate - noisiness |
        | **RMS** | 1 | Root Mean Square Energy - loudness |
        | **Spectral Centroid** | 1 | Brightness of sound |
        | **Spectral Bandwidth** | 1 | Spread of spectrum |
        | **Spectral Rolloff** | 1 | Frequency below which 85% energy exists |
        
        Each language has unique patterns in these features due to different:
        - **Phoneme inventories** (different sounds)
        - **Prosodic patterns** (intonation, stress, rhythm)
        - **Phonotactic rules** (allowed sound combinations)
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🎙️ Voice Language Predictor | Built with TensorFlow, Librosa & Streamlit</p>
        <p>Dataset: Indian Language Speech Dataset (Kaggle) | 
        Languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
