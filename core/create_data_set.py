import os
import random
import numpy as np
import librosa
import torch
from utils.spectro_tools import wav_to_cqt_tensor

def create_training_dataset(base_dir, export_dir):
    """
    Final production pipeline: Generates .npy tensors for Siamese training.
    Optimized for speed by keeping augmentations in RAM.
    """
    categories = ['pos', 'neg_hard', 'neg_easy']
    for cat in categories:
        os.makedirs(os.path.join(export_dir, cat), exist_ok=True)

    all_folders = sorted([f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))])
    
    # Group folders by Track ID
    tracks = {}
    for folder in all_folders:
        track_id = folder.split('_')[0] 
        if track_id not in tracks:
            tracks[track_id] = []
        tracks[track_id].append(folder)

    print("🚀 Phase 2.1: Generating Positive and Hard Negative pairs...")
    for track_id, stem_folders in tracks.items():
        stem_folders.sort() 
        for i in range(len(stem_folders) - 1):
            folder_a, folder_b = stem_folders[i], stem_folders[i+1]
            s_a, s_b = folder_a.split('_')[1], folder_b.split('_')[1] 
            
            # Find loop indices common to both stems
            common_loops = sorted(list(set(os.listdir(os.path.join(base_dir, folder_a))) & 
                                       set(os.listdir(os.path.join(base_dir, folder_b)))))

            for loop_file in common_loops:
                loop_idx = loop_file.split('_')[1].split('.')[0] 
                path_a = os.path.join(base_dir, folder_a, loop_file)
                path_b = os.path.join(base_dir, folder_b, loop_file)

                # --- 1. Load Anchor (t1) ---
                t1 = wav_to_cqt_tensor(path_a)

                # --- 2. POSITIVE PAIR ---
                t2_pos = wav_to_cqt_tensor(path_b)
                np.save(os.path.join(export_dir, 'pos', f"{track_id}_{s_a}_{s_b}_{int(loop_idx)}_pos.npy"), 
                        torch.cat((t1, t2_pos), dim=0).numpy())

                # --- 3. NEG_HARD PAIR (Pitch Shift in RAM) ---
                y_b, sr = librosa.load(path_b, sr=16000)
                y_shifted = librosa.effects.pitch_shift(y_b, sr=sr, n_steps=random.uniform(1, 3))
                t2_hd = wav_to_cqt_tensor(y_shifted)
                
                np.save(os.path.join(export_dir, 'neg_hard', f"{track_id}_{s_a}_{s_b}_{int(loop_idx)}_hd.npy"), 
                        torch.cat((t1, t2_hd), dim=0).numpy())

    print("🚀 Phase 2.2: Generating Easy Negative pairs...")
    num_easy = len(os.listdir(os.path.join(export_dir, 'pos')))
    for i in range(num_easy):
        f1, f2 = random.sample(all_folders, 2)
        # Verify cross-track mismatch
        if f1.split('_')[0] != f2.split('_')[0]:
            p1 = os.path.join(base_dir, f1, random.choice(os.listdir(os.path.join(base_dir, f1))))
            p2 = os.path.join(base_dir, f2, random.choice(os.listdir(os.path.join(base_dir, f2))))
            
            t1, t2 = wav_to_cqt_tensor(p1), wav_to_cqt_tensor(p2)
            np.save(os.path.join(export_dir, 'neg_easy', f"mix_{i}_ez.npy"), 
                    torch.cat((t1, t2), dim=0).numpy())

    print(f"✅ Success. Training dataset ready at: {export_dir}")