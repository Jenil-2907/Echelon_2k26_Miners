import joblib
import numpy as np
import pandas as pd
import librosa
import os
import time

# Hack for old libraries using time.clock
if not hasattr(time, 'clock'):
    time.clock = time.time

# Suppress TF logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.models import load_model

class LSTMPredictor:
    def __init__(self, scaler_path='scaler.joblib', model_path='best_lstm.h5'):
        self.scaler_path = scaler_path
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.load_resources()

    def load_resources(self):
        try:
            print(f"[LSTM] Loading scaler from {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            
            print(f"[LSTM] Loading model from {self.model_path}...")
            self.model = load_model(self.model_path)
            
            # Get model input shape for sequence length
            # Shape: (None, seq_len, features)
            self.sequence_length = self.model.input_shape[1]
            self.n_features = self.model.input_shape[2]
            
            print(f"[LSTM] Model loaded. SeqLen: {self.sequence_length}, Feats: {self.n_features}")
            
        except Exception as e:
            print(f"[LSTM] Error loading resources: {e}")
            self.model = None

    def extract_features(self, audio_path):
        """
        Extract MFCC features from audio file.
        Returns: (n_samples, n_features) array
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None) # Use native SR or fixed?
            # Standard is often 16k or 22k. RawNet uses 16k usually.
            
            # Extract MFCCs
            # n_mfcc must match self.n_features (checked as 40)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_features)
            
            # Transpose to (Time, Features)
            mfccs = mfccs.T 
            return mfccs
            
        except Exception as e:
            print(f"[LSTM] Feature extraction error: {e}")
            return None

    def preprocess_input(self, data_array):
        # Scale
        scaled_data = self.scaler.transform(data_array)
        
        # Create sequences
        X = []
        if len(scaled_data) < self.sequence_length:
            # Pad if too short?
            # padding with zeros
            pad_len = self.sequence_length - len(scaled_data)
            padding = np.zeros((pad_len, self.n_features))
            scaled_data = np.vstack([scaled_data, padding])
            X.append(scaled_data)
        else:
            # Sliding window or non-overlapping?
            # Let's do non-overlapping for speed? Or stride=1?
            # Stride = sequence_length // 2
            stride = max(1, self.sequence_length // 2)
            
            for i in range(0, len(scaled_data) - self.sequence_length + 1, stride):
                X.append(scaled_data[i:i + self.sequence_length])
                
        return np.array(X)

    def predict(self, audio_path):
        if self.model is None:
            return None

        # 1. Extract
        features = self.extract_features(audio_path)
        if features is None:
            return None
            
        # 2. Preprocess (Scale & Sequence)
        try:
            X = self.preprocess_input(features)
            if len(X) == 0:
                print("[LSTM] No sequences created.")
                return None
                
            # 3. Predict
            preds = self.model.predict(X, verbose=0)
            
            # 4. Aggregate
            # Identify output format. Probability?
            # Usually sigmoid output [0,1].
            # 0=Real, 1=Fake or vice versa?
            # RawNet logic was 1=Real.
            # Assuming standard binary classification: >0.5 is class 1.
            # Need to verify which class is which.
            # Usually: 0=Fake, 1=Real ?? Or 0=Real, 1=Spoof?
            # ASVspoof: Logic often varies.
            # We'll assume typical Deepfake detector (1=Fake) unless proven otherwise.
            # User wants to detect Fakes.
            
            avg_score = np.mean(preds)
            
            return avg_score
            
        except Exception as e:
            print(f"[LSTM] Prediction error: {e}")
            return None
