import os
import librosa
import soundfile as sf

def slice_stems(wav_path, bpm, output_dir, bar_count=4, silence_threshold=0.01):
    """
    Slices 1 WAV into 4-bar chunks and saves audible ones to output_dir.

    Args:
        wav_path (str): The filesystem path to the source .wav file.

        bpm (float): The tempo of the track, used to calculate bar duration.

        output_dir (str): The directory where the sliced .wav loops will be saved.

        bar_count (int, optional): The number of musical bars per slice. Defaults to 4.

        silence_threshold (float, optional): The RMS energy limit. Segments quieter than this are discarded. Defaults to 0.01.

    Returns:
        list: A list of strings containing the file paths to all successfully 
            saved loop segments.

    saves in the form of
    └── processed_loops/          
    └── Track00001_S00/        
        ├── loop_000.wav       
        ├── loop_001.wav
        └── loop_002.wav
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load at 16k to match model requirements
    y, sr = librosa.load(wav_path, sr=16000)
    
    # Math: (60 / BPM) = seconds per beat. 
    # Beats per loop = beats_per_bar (4) * bar_count (4)
    seconds_per_loop = (60.0 / bpm) * 4 * bar_count
    samples_per_loop = int(seconds_per_loop * sr)
    
    saved_paths = []
    
    for i, start in enumerate(range(0, len(y) - samples_per_loop, samples_per_loop)):
        chunk = y[start : start + samples_per_loop]
        
        # Calculate energy to detect silence
        rms = librosa.feature.rms(y=chunk)[0]
        if max(rms) > silence_threshold:
            file_name = f"loop_{str(i).zfill(3)}.wav"
            save_path = os.path.join(output_dir, file_name)
            sf.write(save_path, chunk, sr)
            saved_paths.append(save_path)
            
    return saved_paths