import librosa
import numpy as np
import os
import glob
from tqdm import tqdm # Optional: pip install tqdm for a progress bar

def wav_to_cqt_tensor(wav_path):
    # Load audio at a fixed 16k to match the model's expectations
    y, sr = librosa.load(wav_path, sr=16000)
    
    # Generate Constant-Q Transform (CQT)
    # 84 bins = 7 octaves, 12 bins per octave = 1 pixel per semitone
    cqt = librosa.cqt(y, sr=sr, n_bins=84, bins_per_octave=12)
    
    # Convert to decibels (log scale)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    
    # Critical: Min-Max normalization to [0, 1] 
    # This helps the Siamese Network learn faster across different songs
    cqt_norm = (cqt_db - np.min(cqt_db)) / (np.max(cqt_db) - np.min(cqt_db) + 1e-6)
    
    return cqt_norm.astype(np.float32)

def preprocess_scaled_dataset(input_root, output_root):
    if not os.path.exists(input_root):
        print(f"Error: Folder '{input_root}' not found!")
        return

    os.makedirs(output_root, exist_ok=True)
    wav_files = glob.glob(f"{input_root}/**/*.wav", recursive=True)
    
    print(f"Found {len(wav_files)} files. Starting transformation...")
    
    # Using a simple counter since you might not have tqdm installed
    for i, wav_path in enumerate(wav_files):
        try:
            # Maintain folder structure (pos, neg_hard, neg_easy)
            rel_path = os.path.relpath(wav_path, input_root)
            out_path = os.path.join(output_root, rel_path).replace('.wav', '.npy')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            # Convert and save
            cqt_data = wav_to_cqt_tensor(wav_path)
            np.save(out_path, cqt_data)
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(wav_files)} files...")
                
        except Exception as e:
            print(f"Failed to transform {wav_path}: {e}")

    print(f"Success! Scaled tensors are ready in '{output_root}'")

if __name__ == "__main__":
    # Point this to your new multi-song folder
    preprocess_scaled_dataset('multi_song_dataset', 'multi_song_tensors')