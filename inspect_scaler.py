import joblib
import sys

path = r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\PROJECTTO\scaler.joblib"

print(f"Loading {path}...")
try:
    scaler = joblib.load(path)
    print(f"Scaler loaded: {type(scaler)}")
    
    if hasattr(scaler, 'n_features_in_'):
        print(f"n_features_in_: {scaler.n_features_in_}")
    else:
        print("Scaler does not have 'n_features_in_'. Inspecting mean_ or scale_...")
        if hasattr(scaler, 'mean_'):
            print(f"mean_ shape: {scaler.mean_.shape}")
        if hasattr(scaler, 'scale_'):
             print(f"scale_ shape: {scaler.scale_.shape}")
             
except Exception as e:
    print(f"Error: {e}")
