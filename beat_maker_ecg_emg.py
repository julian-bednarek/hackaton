import pickle
import numpy as np
import neurokit2 as nk
from pydub import AudioSegment
from pydub.generators import Sine, Square, Sawtooth, WhiteNoise
from scipy.signal import find_peaks
import warnings

# --- Configuration ---
INPUT_FILE = 'WESAD/S2/S2.pkl'
OUTPUT_PREFIX = 'WESAD/S2'  # We'll add suffixes like _baseline.wav
DATA_SAMPLING_RATE = 700
SEGMENT_DURATION_SEC = 60  # Duration for each emotional segment
warnings.filterwarnings('ignore')

# --- 1. Musical Constants ---
# C Major Scale (easier for smooth motion than pentatonic)
SCALE = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C4 to C5

# Chords (Root, 3rd, 5th)
CHORDS = {
    'C': [261.63, 329.63, 392.00], 'Am': [220.00, 261.63, 329.63],
    'F': [174.61, 220.00, 261.63], 'G': [196.00, 246.94, 293.66],
    'Em': [164.81, 196.00, 246.94]
}

# Progressions
VERSE_PROG = ['C', 'Am', 'C', 'Am']  # Calm, repetitive
CHORUS_PROG = ['F', 'G', 'Em', 'Am']  # Emotional, lifting


# --- 2. Improved Instruments ---
def get_kick(dur_ms=100):
    return Sine(60).to_audio_segment(duration=dur_ms).fade_out(60).apply_gain(0)


def get_snare(dur_ms=150):
    # Layered snare for more body
    low = Sine(180).to_audio_segment(duration=dur_ms).apply_gain(-8)
    high = WhiteNoise().to_audio_segment(duration=dur_ms).high_pass_filter(2000).apply_gain(-12)
    return low.overlay(high).fade_out(100)


def get_hihat(dur_ms=50):
    return WhiteNoise().to_audio_segment(duration=dur_ms).high_pass_filter(8000).fade_out(40).apply_gain(-18)


def get_piano_note(freq, dur_ms=400):
    # "Electric Piano" sound using mixed waves
    sine = Sine(freq).to_audio_segment(duration=dur_ms)
    saw = Sawtooth(freq).to_audio_segment(duration=dur_ms).low_pass_filter(1000).apply_gain(-15)
    note = sine.overlay(saw).fade_in(5).fade_out(300)
    return note.apply_gain(-8)


def get_pad_chord(chord_name, dur_ms=2000):
    pad = AudioSegment.silent(duration=dur_ms)
    for freq in CHORDS[chord_name]:
        osc = Square(freq / 2).to_audio_segment(duration=dur_ms).low_pass_filter(500)  # Lower octave for bass
        pad = pad.overlay(osc)
    return pad.fade_in(500).fade_out(500).apply_gain(-22)


# --- 3. Generation Logic (Unchanged) ---

def generate_song_structure(ecg_rate, emg, rate, total_sec):
    print("Arranging structured pop song...")
    full_mix = AudioSegment.silent(duration=total_sec * 1000)

    # State trackers
    current_time_ms = 0
    beat_counter = 0
    chord_idx = 0
    last_melody_note_idx = 0  # Start at C4 (index 0)

    # Pre-load sounds for speed
    kick = get_kick()
    snare = get_snare()
    hihat = get_hihat()

    # Prepare EMG for melody
    emg_clean = nk.emg_amplitude(emg)
    emg_norm = (emg_clean - np.min(emg_clean)) / (np.max(emg_clean) - np.min(emg_clean))

    while current_time_ms < (total_sec * 1000) - 2000:
        # --- A. Determine Tempo & Section ---
        sec_idx = min(int((current_time_ms / 1000) * rate), len(ecg_rate) - 1)
        bpm = np.clip(ecg_rate[sec_idx], 65, 135)
        ms_per_beat = 60000 / bpm

        # Simple logic: High HR (>90) = Chorus, Low HR = Verse
        is_chorus = bpm > 90
        progression = CHORUS_PROG if is_chorus else VERSE_PROG

        # --- B. Rhythm Section ---
        # Always Hi-hats (8th notes for chorus, quarter for verse)
        full_mix = full_mix.overlay(hihat, position=current_time_ms)
        if is_chorus:
            full_mix = full_mix.overlay(hihat.apply_gain(-5), position=current_time_ms + (ms_per_beat / 2))

        # Kick/Snare pattern
        if beat_counter % 4 == 0: full_mix = full_mix.overlay(kick, position=current_time_ms)  # Beat 1
        if beat_counter % 4 == 2: full_mix = full_mix.overlay(kick, position=current_time_ms)  # Beat 3
        if beat_counter % 4 == 1 or beat_counter % 4 == 3:  # Beats 2 and 4
            full_mix = full_mix.overlay(snare, position=current_time_ms)

        # --- C. Harmony (Chords) ---
        # Change chord every 4 beats (1 bar)
        if beat_counter % 4 == 0:
            chord_name = progression[chord_idx % 4]
            pad = get_pad_chord(chord_name, dur_ms=ms_per_beat * 4)
            # Chorus pads are slightly louder
            if is_chorus: pad = pad.apply_gain(3)
            full_mix = full_mix.overlay(pad, position=current_time_ms)
            chord_idx += 1

        # --- D. Smooth Melody (The "Human" Element) ---
        # Check EMG activity at this exact beat
        emg_val = emg_norm[int((current_time_ms / 1000) * rate)]

        # Only play a note if muscle is active (threshold 0.2)
        if emg_val > 0.2:
            # Decide direction based on EMG intensity
            # High intensity (>0.6) = move pitch UP. Low intensity = move pitch DOWN.
            step = 1 if emg_val > 0.5 else -1

            # Move melody index smoothly (no jumps larger than 1 step)
            new_idx = np.clip(last_melody_note_idx + step, 0, len(SCALE) - 1)

            # Play the note
            note = get_piano_note(SCALE[new_idx], dur_ms=ms_per_beat)
            full_mix = full_mix.overlay(note, position=current_time_ms)

            last_melody_note_idx = new_idx  # Remember for next time

        # Advance time
        current_time_ms += ms_per_beat
        beat_counter += 1

    return full_mix


# --- 4. Main (MODIFIED) ---
def main():
    data = load_pkl_data(INPUT_FILE)
    if data is None:
        print(f"Error: Could not load data from {INPUT_FILE}")
        return

    # Get full signals ONCE
    try:
        ecg_full = data['signal']['chest']['ECG'].flatten()
        emg_full = data['signal']['chest']['EMG'].flatten()
        labels = data['label'].flatten()
    except KeyError:
        print("Error: Data file seems to be missing 'signal' or 'label' keys.")
        return
    
    # Define the segments we want to create
    # WESAD Labels: 1=baseline, 2=stress, 3=amusement (fun), 4=meditation
    segments_to_generate = {
        'baseline': 1,
        'stress': 2,
        'fun': 3,
        'meditation': 4
    }
    
    num_samples_needed = SEGMENT_DURATION_SEC * DATA_SAMPLING_RATE

    # Loop over each defined segment
    for label_name, label_id in segments_to_generate.items():
        print(f"\n--- Processing segment: {label_name.upper()} (Label ID: {label_id}) ---")
        
        # Find the first index where this label occurs
        indices = np.where(labels == label_id)[0]
        
        if len(indices) == 0:
            print(f"Warning: No data found for label '{label_name}' (ID {label_id}). Skipping.")
            continue
            
        start_index = indices[0]
        end_index = start_index + num_samples_needed
        
        # Check if we have enough data for a full segment from that start point
        if end_index > len(ecg_full):
            print(f"Warning: Not enough continuous data for '{label_name}'. Skipping.")
            continue

        # --- Slicing ---
        print(f"Slicing data from sample {start_index} to {end_index}...")
        ecg_segment = ecg_full[start_index:end_index]
        emg_segment = emg_full[start_index:end_index]

        # --- Processing (on the specific segment) ---
        print("Analyzing Heart Rate for tempo...")
        ecg_clean = nk.ecg_clean(ecg_segment, sampling_rate=DATA_SAMPLING_RATE)
        _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=DATA_SAMPLING_RATE)
        ecg_rate = nk.signal_rate(rpeaks, sampling_rate=DATA_SAMPLING_RATE, desired_length=len(ecg_segment))

        # --- Generation ---
        final_song = generate_song_structure(ecg_rate, emg_segment, DATA_SAMPLING_RATE, SEGMENT_DURATION_SEC)
        
        # --- Exporting ---
        output_filename = f"{OUTPUT_PREFIX}_{label_name}.wav"
        final_song.export(output_filename, format="wav")
        print(f"✅ Pop song for '{label_name}' saved to {output_filename}")


def load_pkl_data(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        return None


if __name__ == "__main__":
    main()