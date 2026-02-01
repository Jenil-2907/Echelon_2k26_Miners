import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\PROJECTTO\best_lstm.h5"

try:
    model = load_model(model_path)
    print(f"Model Input Shape: {model.input_shape}")
    print(f"Model Output Shape: {model.output_shape}")
except Exception as e:
    print(f"Error: {e}")
