# 🎙️ Indian Language Speech Prediction

An end-to-end Machine Learning project that classifies Indian languages from voice input. Using advanced audio feature extraction and a hybrid **CNN + Bidirectional LSTM** deep learning architecture, the model can identify languages directly from speech patterns without relying on external APIs.

DEMO LINK :  https://learnedge-voice-language-detection.streamlit.app/
---

## 🚀 Overview

This project provides a complete pipeline from raw audio data to a real-time web application:
1.  **Data Processing**: Converts audio files into rich high-dimensional feature maps.
2.  **Model Training**: Trains a sophisticated neural network on the "Indian Language Speech Dataset".
3.  **Real-time Interface**: A Streamlit web dashboard for users to upload audio files and get instant predictions with confidence scores.

### 🌐 Supported Languages
The current model is trained to recognize:
*   🇮🇳 **Hindi** (हिन्दी)
*   🇮🇳 **Bengali** (বাংলা)
*   🇮🇳 **Marathi** (मराठी)
*   🇮🇳 **Punjabi** (ਪੰਜਾਬੀ)
*   🇮🇳 **Gujarati** (ગુજરાતી)

---

## 📂 Project Structure

```bash
├── dataset/                # Input audio files (organized by language folders)
├── model/                  # Exported model artifacts
│   ├── language_model.keras # Trained CNN+LSTM model
│   ├── label_encoder.pkl   # Language label mappings
│   └── scaler.pkl          # Feature normalization parameters
├── app.py                  # Streamlit Web Application
├── train.py                # Model training and evaluation script
├── model_config.json       # Hyperparameters and model metadata
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

---

## 🛠️ Technical Pipeline

### 1. Data Engineering (Feature Extraction)
We extract **12 distinct categories** of audio features to capture the unique phonetics, rhythm, and tonality of each language:
*   **MFCCs (40 bands)**: Captures the shape of the vocal tract.
*   **Chroma features**: Identifies tonal/harmonic content.
*   **Mel Spectrogram**: Mimics human auditory frequency response.
*   **Spectral Contrast**: Distinguishes between peaks and valleys in the spectrum.
*   **Tonnetz**: Captures harmonic relationships between pitches.
*   **Dynamics**: Zero Crossing Rate, RMS Energy, and Spectral Centroids.

### 2. Deep Learning Architecture
The model uses a **Spatiotemporal Hybrid Architecture**:
*   **Convolutional (Conv1D) Layers**: Act as local feature detectors to find patterns in frequency and time.
*   **Bidirectional LSTM Layers**: Process the sequence in both forward and backward directions to understand the temporal context of speech.
*   **Global Pooling & Dense Layers**: Aggregate the learned features into a final language probability distribution.

---

## ⚙️ Installation & Usage

### 1. Prerequisite Setup
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
1.  Download the **Indian Language Speech Dataset** from Kaggle.
2.  Place the audio folders inside the `dataset/` directory.
3.  Structure: `dataset/Hindi/*.wav`, `dataset/Bengali/*.wav`, etc.

### 3. Training the Model
Run the training script to process data and generate the model artifacts:
```bash
python train.py
```
*This will generate `language_model.keras`, `label_encoder.pkl`, and `scaler.pkl`.*

### 4. Running the Web App
Launch the Streamlit interface:
```bash
streamlit run app.py
```

---

## 📊 Technologies Used

*   **Deep Learning**: TensorFlow & Keras
*   **Audio Processing**: Librosa & SoundFile
*   **Data Science**: NumPy, Pandas, Scikit-learn
*   **Visualization**: Matplotlib & Seaborn
*   **Frontend**: Streamlit

---

## 📈 Model Performance
Based on the latest training session:
*   **Test Accuracy**: ~79.4%
*   **Strengths**: Extremely high accuracy for Bengali, Hindi, and Marathi (>97%).
*   **Optimization**: Includes Early Stopping and Learning Rate Reduction on Plateau to ensure optimal convergence.

---

*Built with ❤️ for Indian Language Computing.*
