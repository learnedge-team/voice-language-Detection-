"""
train.py - Voice Language Prediction Model Training
Uses MFCC, Chroma, Mel Spectrogram features from audio files
Trains a Deep Learning model (CNN + LSTM) to predict language
Dataset: Indian Language Speech Dataset from Kaggle
"""

import os
import sys
import numpy as np
import librosa
import joblib
import warnings
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv1D, MaxPooling1D, LSTM, BatchNormalization,
    Input, Flatten, Bidirectional, GlobalAveragePooling1D, Reshape,
    Attention, concatenate, GRU
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "dataset"  # Path to the dataset folder
SAMPLE_RATE = 22050
DURATION = 5  # seconds - each audio clip will be padded/truncated to this
N_MFCC = 40
N_MELS = 128
N_CHROMA = 12
HOP_LENGTH = 512
N_FFT = 2048
MAX_PAD_LEN = 216  # Max time steps after feature extraction

LANGUAGES = [
    'Bengali', 'Gujarati', 'Hindi','Marathi', 'Punjabi'
]

BATCH_SIZE = 32
EPOCHS = 100
TEST_SIZE = 0.2
RANDOM_STATE = 42


def extract_features(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Extract multiple audio features from a single audio file:
    - MFCC (40 coefficients)
    - MFCC Delta
    - MFCC Delta2
    - Chroma
    - Mel Spectrogram (compressed)
    - Spectral Contrast
    - Tonnetz
    - Zero Crossing Rate
    - RMS Energy
    
    All features are concatenated along the feature axis and
    padded/truncated to fixed time length.
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sr, duration=duration)
        
        # Ensure minimum length
        if len(audio) < sr * 0.5:  # Less than 0.5 seconds
            return None
        
        # Pad if shorter than duration
        max_len = sr * duration
        if len(audio) < max_len:
            audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
        else:
            audio = audio[:max_len]
        
        # Pre-emphasis filter to boost high frequencies
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        # 1. MFCC features (40 coefficients)
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=N_MFCC,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # 2. MFCC Delta (velocity)
        mfcc_delta = librosa.feature.delta(mfcc)
        
        # 3. MFCC Delta2 (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 4. Chroma features
        chroma = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_chroma=N_CHROMA,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # 5. Mel spectrogram (take first 20 bands for dimensionality)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Reduce mel dimensions by taking every 6th band
        mel_reduced = mel_db[::6, :]  # ~21 features
        
        # 6. Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # 7. Tonnetz
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio), sr=sr
        )
        
        # 8. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            audio, hop_length=HOP_LENGTH
        )
        
        # 9. RMS Energy
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)
        
        # 10. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=HOP_LENGTH
        )
        
        # 11. Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=sr, hop_length=HOP_LENGTH
        )
        
        # 12. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=sr, hop_length=HOP_LENGTH
        )
        
        # Concatenate all features along feature axis
        features = np.concatenate([
            mfcc,               # 40 x T
            mfcc_delta,         # 40 x T
            mfcc_delta2,        # 40 x T
            chroma,             # 12 x T
            mel_reduced,        # ~21 x T
            spectral_contrast,  # 7 x T
            tonnetz,            # 6 x T
            zcr,                # 1 x T
            rms,                # 1 x T
            spectral_centroid,  # 1 x T
            spectral_bandwidth, # 1 x T
            spectral_rolloff    # 1 x T
        ], axis=0)
        
        # Transpose to (time_steps, features)
        features = features.T
        
        # Pad or truncate to fixed length
        if features.shape[0] < MAX_PAD_LEN:
            pad_width = MAX_PAD_LEN - features.shape[0]
            features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
        else:
            features = features[:MAX_PAD_LEN, :]
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_dataset(dataset_path):
    """
    Load all audio files from the dataset directory.
    Each subdirectory name is the language label.
    """
    features_list = []
    labels_list = []
    file_count = {}
    
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path '{dataset_path}' not found!")
        print("Please download the Indian Language Speech Dataset from Kaggle")
        print("and extract it into the 'dataset' folder.")
        sys.exit(1)
    
    # Get available languages
    available_langs = []
    for lang in LANGUAGES:
        lang_path = os.path.join(dataset_path, lang)
        if os.path.exists(lang_path):
            available_langs.append(lang)
            count = len([f for f in os.listdir(lang_path) 
                        if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))])
            file_count[lang] = count
            print(f"  {lang}: {count} audio files found")
        else:
            print(f"  {lang}: DIRECTORY NOT FOUND - SKIPPING")
    
    if len(available_langs) < 2:
        print("ERROR: Need at least 2 language directories!")
        sys.exit(1)
    
    print(f"\nTotal languages found: {len(available_langs)}")
    print(f"Total audio files: {sum(file_count.values())}")
    print("\nExtracting features...")
    
    total_processed = 0
    total_failed = 0
    
    for lang in available_langs:
        lang_path = os.path.join(dataset_path, lang)
        audio_files = [f for f in os.listdir(lang_path) 
                      if f.endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a'))]
        
        lang_processed = 0
        for i, filename in enumerate(audio_files):
            file_path = os.path.join(lang_path, filename)
            
            # Progress indicator
            if (i + 1) % 50 == 0 or (i + 1) == len(audio_files):
                print(f"  {lang}: Processing {i+1}/{len(audio_files)}...")
            
            features = extract_features(file_path)
            
            if features is not None:
                features_list.append(features)
                labels_list.append(lang)
                lang_processed += 1
                total_processed += 1
            else:
                total_failed += 1
        
        print(f"  {lang}: Successfully processed {lang_processed}/{len(audio_files)}")
    
    print(f"\nTotal successfully processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    return X, y, available_langs


def build_model(input_shape, num_classes):
    """
    Build a CNN + Bidirectional LSTM model for language classification.
    
    Architecture:
    - Conv1D layers for local pattern extraction
    - Bidirectional LSTM for sequential pattern learning
    - Dense layers for classification
    """
    model = Sequential([
        # Input
        Input(shape=input_shape),
        
        # Conv Block 1
        Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Conv Block 2
        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Conv Block 3
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Conv Block 4
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Bidirectional LSTM layers
        Bidirectional(LSTM(128, return_sequences=True)),
        Dropout(0.4),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.4),
        
        # Dense classification head
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=classes, yticklabels=classes
    )
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('Actual', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    print("=" * 60)
    print("  VOICE LANGUAGE PREDICTION - MODEL TRAINING")
    print("  Indian Language Speech Dataset")
    print("=" * 60)
    print()
    
    # ============================================================
    # Step 1: Load Dataset
    # ============================================================
    X, y, available_langs = load_dataset(DATASET_PATH)
    
    print(f"\nDataset shape: X={X.shape}, y={y.shape}")
    print(f"Feature dimensions: {X.shape[1]} time steps x {X.shape[2]} features")
    
    # ============================================================
    # Step 2: Encode Labels
    # ============================================================
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    
    print(f"\nLanguage classes ({num_classes}):")
    for i, lang in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        print(f"  {i}: {lang} ({count} samples)")
    
    # One-hot encode
    y_onehot = to_categorical(y_encoded, num_classes=num_classes)
    
    # ============================================================
    # Step 3: Normalize Features
    # ============================================================
    print("\nNormalizing features...")
    
    # Reshape for scaler: (samples * time_steps, features)
    n_samples, n_timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, n_timesteps, n_features)
    
    # Replace any NaN/Inf values
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ============================================================
    # Step 4: Train/Test Split
    # ============================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_onehot, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # ============================================================
    # Step 5: Build Model
    # ============================================================
    print("\nBuilding model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_model(input_shape, num_classes)
    
    # Compile
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # ============================================================
    # Step 6: Callbacks
    # ============================================================
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # ============================================================
    # Step 7: Train Model
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # ============================================================
    # Step 8: Evaluate Model
    # ============================================================
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    # Load best model
    model = tf.keras.models.load_model('best_model.keras')
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification Report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    print("\nClassification Report:")
    print("-" * 60)
    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=label_encoder.classes_
    )
    print(report)
    
    # ============================================================
    # Step 9: Save Plots
    # ============================================================
    plot_training_history(history)
    plot_confusion_matrix(
        y_true_classes, y_pred_classes,
        label_encoder.classes_
    )
    
    # ============================================================
    # Step 10: Save Model and Artifacts
    # ============================================================
    print("\nSaving model and artifacts...")
    
    # Save the final model
    model.save('language_model.keras')
    print("  Model saved: language_model.keras")
    
    # Save label encoder
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("  Label encoder saved: label_encoder.pkl")
    
    # Save scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("  Scaler saved: scaler.pkl")
    
    # Save config
    config = {
        'sample_rate': SAMPLE_RATE,
        'duration': DURATION,
        'n_mfcc': N_MFCC,
        'n_mels': N_MELS,
        'n_chroma': N_CHROMA,
        'hop_length': HOP_LENGTH,
        'n_fft': N_FFT,
        'max_pad_len': MAX_PAD_LEN,
        'input_shape': list(input_shape),
        'num_classes': num_classes,
        'languages': list(label_encoder.classes_),
        'test_accuracy': float(test_accuracy),
        'n_features': int(n_features)
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("  Config saved: model_config.json")
    
    print("\n" + "=" * 60)
    print(f"  TRAINING COMPLETE!")
    print(f"  Best Test Accuracy: {test_accuracy*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()