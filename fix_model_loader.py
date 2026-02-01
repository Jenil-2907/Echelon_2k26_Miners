import torch
import os
import sys

# Paths to try
paths = [
    r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\RawNet2_VoIP_Tuned_Ep2.pth",
    r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\RawNet2_VoIP_Tuned_Ep2_model.pth",
    r"c:\Users\HARSH KUKADIYA\OneDrive\Desktop\RawNet2_VoIP_Tuned_Ep2\PROJECTTO\RawNet2_95_acc.pth" # The folder? No, torch.load can't load folder
]

def sanitize(state_dict):
    """Remove specific module prefixes and odd keys"""
    new_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_dict[k] = v
    return new_dict

print(f"PyTorch Version: {torch.__version__}")

for p in paths:
    if not os.path.exists(p):
        print(f"Skipping {p}: Not found")
        continue
        
    print(f"\nAttempting to load: {p}")
    try:
        # 1. Standard Load
        data = torch.load(p, map_location='cpu')
        print("  -> Loaded successfully with standard torch.load")
        
        if isinstance(data, dict):
            print("  -> It's a dictionary.")
            if 'state_dict' in data:
                data = data['state_dict']
            
            # Re-save cleanly
            clean_path = "cleaned_model.pth"
            torch.save(data, clean_path)
            print(f"  -> Re-saved state_dict to {clean_path}")
            sys.exit(0)
            
        else:
            print("  -> It's a full model object.")
            # Re-save state dict
            torch.save(data.state_dict(), "cleaned_model.pth")
            print("  -> Saved state_dict to cleaned_model.pth")
            sys.exit(0)
            
    except Exception as e:
        print(f"  -> Standard load failed: {e}")
        
    # 2. Try simple pickle
    import pickle
    try:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        print("  -> Loaded with pickle!")
        # Inspect
        print(f"Type: {type(data)}")
    except Exception as e:
        print(f"  -> Pickle failed: {e}")
