import mido
import librosa
import soundfile as sf
import os


def get_bpm_from_midi(midi_path):
   """Extracts BPM from a MIDI file."""
   mid = mido.MidiFile(midi_path)
   for track in mid.tracks:
       for msg in track:
           if msg.type == 'set_tempo':
               return mido.tempo2bpm(msg.tempo)
   return 0  # Default if not found


def slice_stem_to_4_bars(audio_path, midi_path, output_folder):
   # 1. Get the Tempo
   bpm = get_bpm_from_midi(midi_path)
   print(f"Detected BPM: {bpm}")


   # 2. Calculate loop duration in seconds
   # Formula: (60 / BPM) * 4 beats * 4 bars
   seconds_per_bar = 60 / bpm
   loop_duration = seconds_per_bar * 16 # 16 beats total
  
   # 3. Load the full stem
   y, sr = librosa.load(audio_path, sr=None)
   total_duration = librosa.get_duration(y=y, sr=sr)
  
   # 4. Create output directory
   os.makedirs(output_folder, exist_ok=True)
  
   # 5. Slice and Save
   num_loops = int(total_duration // loop_duration)
  
   for i in range(num_loops):
       start_sec = i * loop_duration
       end_sec = start_sec + loop_duration
      
       # Convert seconds to sample indices
       start_sample = int(start_sec * sr)
       end_sample = int(end_sec * sr)
      
       chunk = y[start_sample:end_sample]
      
       # Skip if the loop is mostly silent (RMS energy check)
       rms = librosa.feature.rms(y=chunk).mean()
       if rms < 0.005:
           continue
          
       output_filename = f"loop_{i:03d}.wav"
       sf.write(os.path.join(output_folder, output_filename), chunk, sr)
      
   print(f"Finished! Created {num_loops} potential loops in {output_folder}")


# Example usage based on your screenshot:
slice_stem_to_4_bars('music/babyslakh_16k/Track00005/stems/S05.wav', 'music/babyslakh_16k/Track00005/MIDI/S05.mid', 'processed_loops/Track00005_S05')