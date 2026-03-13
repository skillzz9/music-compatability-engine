import torch
import librosa
import numpy as np
import torch.nn.functional as F
import os
from model import SiameseTwin

def load_and_preprocess_loop(wav_path, target_width=300):
    """Ensures every loop is exactly target_width for the SNN."""
    y, sr = librosa.load(wav_path, sr=16000)
    
    # 1. CQT Transformation
    cqt = librosa.cqt(y, sr=sr, n_bins=84, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # 2. Normalization
    cqt_norm = (cqt_db - np.min(cqt_db)) / (np.max(cqt_db) - np.min(cqt_db) + 1e-6)
    
    # 3. Convert to Tensor [1, 84, width]
    tensor = torch.from_numpy(cqt_norm).unsqueeze(0).float()
    
    # 4. Dimension Matching (Padding/Trimming)
    current_width = tensor.shape[2]
    if current_width < target_width:
        pad_amount = target_width - current_width
        tensor = F.pad(tensor, (0, pad_amount, 0, 0), value=0)
    else:
        tensor = tensor[:, :, :target_width]
    
    # 5. Add Batch Dimension [1, 1, 84, 300]
    return tensor.unsqueeze(0)

def test_cross_tracks(track_a, stem_a, loop_a, track_b, stem_b, loop_b):
    # 1. Build Custom Paths for two different tracks
    base_dir = "processed_loops"
    
    folder_a = f"Track{str(track_a).zfill(5)}_S{str(stem_a).zfill(2)}"
    file_a = f"loop_{str(loop_a).zfill(3)}.wav"
    
    folder_b = f"Track{str(track_b).zfill(5)}_S{str(stem_b).zfill(2)}"
    file_b = f"loop_{str(loop_b).zfill(3)}.wav"

    path_a = os.path.join(base_dir, folder_a, file_a)
    path_b = os.path.join(base_dir, folder_b, file_b)

    # 2. Safety Check
    if not os.path.exists(path_a) or not os.path.exists(path_b):
        print(f"Error: Files not found!\nCheck: {path_a}\nCheck: {path_b}")
        return

    # 3. Load Model
    model = SiameseTwin()
    model.load_state_dict(torch.load("mini_snn_model.pth", map_location="cpu"))
    model.eval()

    # 4. Run Inference
    t1 = load_and_preprocess_loop(path_a)
    t2 = load_and_preprocess_loop(path_b)

    with torch.no_grad():
        out1, out2 = model(t1, t2)
        # Calculate Euclidean Distance
        dist = torch.norm(out1 - out2).item()
        
        # Compatibility Score (1.0 margin)
        score = max(0, (1 - dist) * 100)
        
        print("\n" + "✖" * 40)
        print(f"CROSS-TRACK TEST")
        print(f"A: Track {track_a} | Stem {stem_a} | Loop {loop_a}")
        print(f"B: Track {track_b} | Stem {stem_b} | Loop {loop_b}")
        print("-" * 40)
        print(f"Calculated Distance: {dist:.4f}")
        print(f"COMPATIBILITY SCORE: {score:.2f}%")
        print("✖" * 40)

if __name__ == "__main__":
    # Example: Compare Track 1 (Stem 0, Loop 12) with Track 5 (Stem 5, Loop 12)
    test_cross_tracks(track_a=1, stem_a=0, loop_a=12, 
                      track_b=5, stem_b=5, loop_b=12)