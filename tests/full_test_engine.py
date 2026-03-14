import os
import sys
import torch
import numpy as np
import torch.nn.functional as F

# Path Fixes
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from core.model import SiameseTwin

def run_full_test(data_dir, model_path, track_limit=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load Model
    model = SiameseTwin()
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    categories = {'pos': 1, 'neg_hard': -1, 'neg_easy': -1}
    results = {cat: {"correct": 0, "total": 0} for cat in categories}

    print(f"\n🧪 STARTING FULL EVALUATION (Limit: {track_limit} tracks)")
    print(f"{'CATEGORY':<10} | {'FILE NAME':<45} | {'SIMILARITY':<10} | {'RESULT'}")
    print("-" * 85)

    with torch.no_grad():
        for cat, target_label in categories.items():
            cat_path = os.path.join(data_dir, cat)
            if not os.path.exists(cat_path): continue

            files = sorted([f for f in os.listdir(cat_path) if f.endswith('.npy')])
            
            for f in files:
                # Track limit filter logic
                track_id = f.split('_')[0]
                try:
                    track_num = int(track_id.replace("Track", ""))
                    if track_num > track_limit: continue
                except: pass

                # Load Tensor
                data = np.load(os.path.join(cat_path, f))
                t1 = torch.from_numpy(data[0]).unsqueeze(0).unsqueeze(0).to(device).float()
                t2 = torch.from_numpy(data[1]).unsqueeze(0).unsqueeze(0).to(device).float()

                # Inference
                out1, out2 = model(t1, t2)
                sim_raw = F.cosine_similarity(out1, out2).item()
                
                # Convert -1...1 to 0...100%
                sim_pct = ((sim_raw + 1) / 2) * 100

                # Prediction (Similarity > 65% is treated as a match)
                prediction = 1 if sim_pct > 65 else -1
                
                is_correct = (prediction == target_label)
                status = "✅ PASS" if is_correct else "❌ FAIL"
                
                results[cat]["total"] += 1
                if is_correct:
                    results[cat]["correct"] += 1

                # Detailed Per-File Print
                print(f"{cat.upper():<10} | {f:<45} | {sim_pct:>8.2f}% | {status}")

    # --- FINAL SUMMARY REPORT ---
    print("\n" + "="*40)
    print("📊 SIAMESE NETWORK EVALUATION SUMMARY")
    print("="*40)
    
    overall_correct = 0
    overall_total = 0

    for cat, stats in results.items():
        if stats["total"] > 0:
            acc = (stats["correct"] / stats["total"]) * 100
            print(f"{cat.upper():<10} | Accuracy: {acc:>6.2f}% ({stats['correct']}/{stats['total']})")
            overall_correct += stats["correct"]
            overall_total += stats["total"]

    if overall_total > 0:
        total_acc = (overall_correct / overall_total) * 100
        print("-" * 40)
        print(f"🔥 FINAL TOTAL ACCURACY: {total_acc:.2f}%")
        print("="*40)

if __name__ == "__main__":
    DATA_DIR = os.path.join(project_root, "training_dataset")
    MODEL_PATH = os.path.join(project_root, "harmony_model.pth")
    
    run_full_test(DATA_DIR, MODEL_PATH, track_limit=20)