import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import os
import glob

class MultiSongDataset(Dataset):
    def __init__(self, tensor_root):
        self.pairs = []
        
        # 1. Find all "Base" files in the positive folder
        pos_files = glob.glob(os.path.join(tensor_root, 'pos', '*.npy'))
        base_files = [f for f in pos_files if not f.endswith('_p.npy')]

        for b_path in base_files:
            # Partners and Negatives naming logic
            p_path = b_path.replace('.npy', '_p.npy')
            h_path = b_path.replace('/pos/', '/neg_hard/').replace('.npy', '_sh.npy')
            e_path = b_path.replace('/pos/', '/neg_easy/').replace('.npy', '_ez.npy')

            if os.path.exists(p_path):
                self.pairs.append((b_path, p_path, 0.0)) # Positive
            if os.path.exists(h_path):
                self.pairs.append((b_path, h_path, 1.0)) # Hard Negative
            if os.path.exists(e_path):
                self.pairs.append((b_path, e_path, 1.0)) # Easy Negative

        print(f"Dataset initialized with {len(self.pairs)} total pairs.")

    def __len__(self):
        return len(self.pairs)

    # --- THIS MUST BE DEFINED INSIDE THE CLASS ---
    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        
        t1 = torch.from_numpy(np.load(path1)).unsqueeze(0).float()
        t2 = torch.from_numpy(np.load(path2)).unsqueeze(0).float()

        # Target width to handle different BPMs
        target_width = 300 
        
        def pad_tensor(t, width):
            pad_amount = width - t.shape[2]
            if pad_amount > 0:
                # Pad the time dimension (width) with zeros
                return F.pad(t, (0, pad_amount, 0, 0), value=0)
            return t[:, :, :width]

        t1 = pad_tensor(t1, target_width)
        t2 = pad_tensor(t2, target_width)
        
        return t1, t2, torch.tensor(label, dtype=torch.float)