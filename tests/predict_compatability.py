import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# 1. SETUP PATHS
# Get absolute path to 'musicanalyser' (the project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

# Add project root to sys.path so we can find 'core' and 'utils'
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import our custom modules
from core.model import SiameseTwin
from utils.spectro_tools import wav_to_cqt_tensor

def get_compatibility(path_a, path_b, model_path):
    """
    Takes two audio paths and returns a compatibility score (0 to 100%).
    """
    # Verify file existence before processing
    for p in [path_a, path_b, model_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"❌ File not found: {p}")

    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load the Model
    model = SiameseTwin()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Process Audio into Tensors
    t1 = wav_to_cqt_tensor(path_a).unsqueeze(0).to(device)
    t2 = wav_to_cqt_tensor(path_b).unsqueeze(0).to(device)

    # Forward Pass
    with torch.no_grad():
        out1, out2 = model(t1, t2)
        similarity = F.cosine_similarity(out1, out2)
        
    # Scale -1...1 to 0...100%
    score = ((similarity.item() + 1) / 2) * 100
    return score

if __name__ == "__main__":
    # Define absolute paths for everything so it never fails
    MODEL_PATH = os.path.join(project_root, "harmony_model.pth")
    
    # --- TEST 1: Neighboring Stems (Seen Logic) ---
    file_1 = os.path.join(project_root, "ownmusic/pair2_1.wav")
    file_2 = os.path.join(project_root, "ownmusic/pair2_2.wav")
    
    print(f"\n🔍 TEST 1: {os.path.basename(file_1)} vs {os.path.basename(file_2)}")
    try:
        res1 = get_compatibility(file_1, file_2, MODEL_PATH)
        print(f"🎸 SCORE: {res1:.2f}%")
        if res1 > 85: print("✅ GREAT match!")
        elif res1 > 60: print("⚠️ OKAY match.")
        else: print("❌ CLASH.")
    except Exception as e:
        print(e)

    print("-" * 30)

    # --- TEST 2: Already Seen Data Test ---
    file_3 = os.path.join(project_root, "processed_loops/Track00001_S04/loop_003.wav")
    file_4 = os.path.join(project_root, "processed_loops/Track00001_S05/loop_003.wav")

    print(f"🔍 TEST 2: {os.path.basename(file_3)} vs {os.path.basename(file_4)}")
    try:
        res2 = get_compatibility(file_3, file_4, MODEL_PATH)
        print(f"🎸 SCORE: {res2:.2f}%")
    except Exception as e:
        print(e)

    print("\n🏁 Testing complete.")