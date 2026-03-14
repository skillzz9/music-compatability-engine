import os
from utils.audio_slicer import slice_stems
from utils.bpm_engine import get_bpm_from_midi

def run_slicer_pipeline(raw_music_dir, output_base_dir, track_limit=5):
    """
    Handles the batch processing of raw tracks into loops.
    """
    all_tracks = sorted([d for d in os.listdir(raw_music_dir) 
                         if os.path.isdir(os.path.join(raw_music_dir, d)) and d.startswith("Track")])
    target_tracks = all_tracks[:track_limit]

    print(f"🎵 Slicing the first {track_limit} tracks...")

    for track_name in target_tracks:
        track_path = os.path.join(raw_music_dir, track_name)
        midi_dir = os.path.join(track_path, "MIDI")
        stems_dir = os.path.join(track_path, "stems")

        # Get BPM from MIDI
        midi_files = [f for f in os.listdir(midi_dir) if f.endswith(".mid")]
        if not midi_files: continue
            
        bpm = get_bpm_from_midi(os.path.join(midi_dir, midi_files[0]))
        if bpm == 0: continue

        print(f"✅ Processing {track_name} ({bpm:.2f} BPM)")

        # Slice Stems
        wav_files = [f for f in os.listdir(stems_dir) if f.endswith(".wav")]
        for wav_file in wav_files:
            stem_id = wav_file.replace(".wav", "")
            stem_output_dir = os.path.join(output_base_dir, f"{track_name}_{stem_id}")
            
            slice_stems(
                wav_path=os.path.join(stems_dir, wav_file),
                bpm=bpm,
                output_dir=stem_output_dir
            )
    print("✨ Slicing Complete.")