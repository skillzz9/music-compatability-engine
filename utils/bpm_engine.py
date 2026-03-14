import mido

def get_bpm_from_midi(midi_path):
    """
    Extracts the first BPM (tempo) marking found in a MIDI file.
    Defaults to 0 if no tempo is found or file is missing. 
    This function is used to provide the BPM to the audio slicer.

    Args:
        midi_path: path to the midi file of the track you want to find the BPM of 

    Returns:
        float: The tempo of the song in Beats Per Minute (BPM). 
               Returns 0 if the file is unreadable or no tempo data exists.
    """
    try:
        mid = mido.MidiFile(midi_path)
        for msg in mid:
            if msg.type == 'set_tempo':
                # Convert microseconds per beat to BPM
                return mido.tempo2bpm(msg.tempo)
    except Exception as e:
        print(f"Warning: Could not read MIDI {midi_path}. Error: {e}")
    
    return 0 # If BPM is 0, then there was an error. 