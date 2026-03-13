import mido
import librosa
import soundfile as sf
import os
import numpy as np

def get_bpm_from_midi(midi_path):
    try:
        mid = mido.MidiFile(midi_path)
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    return mido.tempo2bpm(msg.tempo)
    except:
        return 0 
    return 0

def create_scaled_dataset(track_list, base_path, output_root):
    # Setup folders
    for f in ['pos', 'neg_hard', 'neg_easy']: 
        os.makedirs(os.path.join(output_root, f), exist_ok=True)

    for track_id in track_list:
        track_path = os.path.join(base_path, track_id)
        stem_dir = os.path.join(track_path, 'stems')
        
        # Get sorted list of stems (S00.wav, S01.wav, etc.)
        stems = sorted([s for s in os.listdir(stem_dir) if s.endswith('.wav')])
        
        # We need a MIDI for BPM - usually S00.mid or S01.mid works
        midi_path = os.path.join(track_path, 'MIDI', stems[0].replace('.wav', '.mid'))
        bpm = get_bpm_from_midi(midi_path)
        sr = 16000 # Standardize sample rate for the model
        
        seconds_per_bar = 60 / bpm
        loop_duration = seconds_per_bar * 16 

        # --- PAIRING LOOP ---
        # This does the "Neighbor Pairing": (S00, S01), (S01, S02), etc.
        for i in range(len(stems) - 1):
            stem_a_name = stems[i]
            stem_b_name = stems[i+1]
            
            print(f"Processing {track_id}: {stem_a_name} + {stem_b_name} @ {bpm} BPM")
            
            y_base, _ = librosa.load(os.path.join(stem_dir, stem_a_name), sr=sr)
            y_partner, _ = librosa.load(os.path.join(stem_dir, stem_b_name), sr=sr)

            count = 0
            idx = 0
            # We take 5 high-quality loops per pair to keep the dataset balanced
            while count < 5:
                start_sample = int(idx * loop_duration * sr)
                end_sample = int(start_sample + (loop_duration * sr))
                
                if end_sample > len(y_base): break
                
                chunk_base = y_base[start_sample:end_sample]
                chunk_partner = y_partner[start_sample:end_sample]

                if librosa.feature.rms(y=chunk_base).mean() > 0.005 and \
                   librosa.feature.rms(y=chunk_partner).mean() > 0.005:
                    
                    # File naming: Track_StemA_StemB_Index
                    prefix = f"{track_id}_{stem_a_name[:-4]}_{stem_b_name[:-4]}_{count}"
                    
                    # Save Positive
                    sf.write(f"{output_root}/pos/{prefix}.wav", chunk_base, sr)
                    sf.write(f"{output_root}/pos/{prefix}_p.wav", chunk_partner, sr)

                    # Save Hard Negative
                    partner_shifted = librosa.effects.pitch_shift(chunk_partner, sr=sr, n_steps=1)
                    sf.write(f"{output_root}/neg_hard/{prefix}_sh.wav", partner_shifted, sr)

                    # Save Easy Negative (from later in the track)
                    easy_start = start_sample + int(20 * sr)
                    if easy_start + int(loop_duration * sr) < len(y_partner):
                        chunk_easy = y_partner[easy_start:easy_start + int(loop_duration * sr)]
                        sf.write(f"{output_root}/neg_easy/{prefix}_ez.wav", chunk_easy, sr)

                    count += 1
                idx += 1

# --- RUN IT ---
my_tracks = ['Track00001', 'Track00002', 'Track00003', 'Track00004', 'Track00005']
create_scaled_dataset(my_tracks, 'music/babyslakh_16k', 'multi_song_dataset')