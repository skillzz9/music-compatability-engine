import sys
from core.create_processed_loops import run_slicer_pipeline
from core.create_data_set import create_training_dataset
from core.trainer import run_training_pipeline

def str_to_bool(val):
    return val.lower() in ("true", "t", "1", "yes")

def main():
    RAW_DIR = "music/babyslakh_16k"
    LOOPS_DIR = "processed_loops"
    TRAIN_DATA_DIR = "training_dataset"

    # Get flags from CLI
    do_p1 = str_to_bool(sys.argv[1]) if len(sys.argv) > 1 else False
    do_p2 = str_to_bool(sys.argv[2]) if len(sys.argv) > 2 else False
    do_p3 = str_to_bool(sys.argv[3]) if len(sys.argv) > 3 else False

    if do_p1:
        print("\n--- PHASE 1: SLICING ---")
        run_slicer_pipeline(RAW_DIR, LOOPS_DIR, track_limit=20)

    if do_p2:
        print("\n--- PHASE 2: DATASET CREATION ---")
        create_training_dataset(LOOPS_DIR, TRAIN_DATA_DIR)

    if do_p3:
        print("\n--- PHASE 3: TRAINING ---")
        run_training_pipeline(TRAIN_DATA_DIR, epochs=10, batch_size=16)

if __name__ == "__main__":
    main()