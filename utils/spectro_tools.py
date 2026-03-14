import librosa
import numpy as np
import torch
import torch.nn.functional as F

def wav_to_cqt_tensor(audio_input, n_bins=84, target_width=300, sr=16000):
    """
    Converts audio (path or array) into a standardized 84x300 CQT Tensor.
    """
    # 1. Handle Input Type
    if isinstance(audio_input, str):
        # It's a file path
        y, _ = librosa.load(audio_input, sr=sr)
    else:
        # It's already a numpy array (e.g. from pitch_shift)
        y = audio_input
    
    # 2. Constant-Q Transform
    cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=12)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # 3. Min-Max Normalization to [0, 1]
    # Adding a small epsilon (1e-6) to avoid division by zero
    norm = (cqt_db - np.min(cqt_db)) / (np.max(cqt_db) - np.min(cqt_db) + 1e-6)
    
    # 4. Convert to Torch Tensor [1, 84, CurrentWidth]
    tensor = torch.from_numpy(norm).unsqueeze(0).float()
    
    # 5. Strict Dimension Matching (Pad or Trim)
    current_width = tensor.shape[2]
    if current_width < target_width:
        pad_amount = target_width - current_width
        tensor = F.pad(tensor, (0, pad_amount, 0, 0), value=0)
    else:
        tensor = tensor[:, :, :target_width]
        
    return tensor