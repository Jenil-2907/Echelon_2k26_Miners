import os
import sys
import torch
import soundfile as sf
import numpy as np
from collections import OrderedDict

# 1. Setup paths so we can import the model structure
# Adjust this path if 'RawNet2_95_acc.pth' folder is located elsewhere
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CODE_DIR = os.path.join(PROJECT_DIR, "RawNet2_95_acc.pth", "RawNet2_95_acc", "data")
sys.path.append(MODEL_CODE_DIR)

try:
    from model import RawNet2
except ImportError:
    print(f"Error: Could not import RawNet2 from {MODEL_CODE_DIR}")
    print("Please check if the folder structure is correct.")
    sys.exit(1)

# 2. Configuration for RawNet2
MODEL_ARGS = {
    'nb_samp': 64600,
    'first_conv': 1024,
    'in_channels': 1,
    'filts': [20, [20, 20], [20, 128]], 
    'gru_node': 1024,
    'nb_gru_layer': 3,
    'nb_fc_node': 1024,
    'nb_classes': 2
}

# 3. Path to the trained weights
# Try to find the model file in common locations
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(PROJECT_DIR), "RawNet2_VoIP_Tuned_Ep2.pth")
if not os.path.exists(MODEL_WEIGHTS_PATH):
    # Fallback to current directory
    MODEL_WEIGHTS_PATH = os.path.join(PROJECT_DIR, "RawNet2_VoIP_Tuned_Ep2.pth")

def check_audio_real_or_fake(audio_path):
    """
    Checks if the input audio file is Real or Fake.
    Returns: string 'Real' or 'Fake'
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load Model
    model = RawNet2(MODEL_ARGS).to(device)
    
    try:
        if device == 'cpu':
            checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location='cpu')
        else:
            checkpoint = torch.load(MODEL_WEIGHTS_PATH)
            
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict()

        # Remove 'module.' prefix if present
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return "Error"

    model.eval()

    # Load and process Audio
    try:
        audio, sr = sf.read(audio_path, dtype='float32')
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return "Error"

    # Use first channel if multi-channel
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Pad or truncate to correct length
    target_len = MODEL_ARGS['nb_samp']
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), 'wrap')
    else:
        audio = audio[:target_len]

    # Prepare tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(audio_tensor, is_test=False)
        probabilities = torch.softmax(output, dim=1)
        _, prediction_idx = torch.max(output, 1)
        
    # Index 1 is Real, 0 is Fake (based on typical ASVspoof labels)
    # Adjust if your specific training swapped them, but 1=Real is standard
    result = "Real" if prediction_idx.item() == 1 else "Fake"
    
    return result

if __name__ == "__main__":
    # Test with audio1.wav in the current folder or project root
    test_audio = os.path.join(PROJECT_DIR, "audio1.wav")
    
    if os.path.exists(test_audio):
        print(f"Checking {test_audio}...")
        result = check_audio_real_or_fake(test_audio)
        print(f"Result: {result}")
    else:
        print(f"Please place an audio1.wav file in {PROJECT_DIR} to test.")
