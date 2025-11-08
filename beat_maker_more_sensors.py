import pickle
import numpy as np
import neurokit2 as nk
from pydub import AudioSegment
from pydub.generators import Sine, Square, Sawtooth, WhiteNoise
import warnings
import os

# --- Configuration ---
# Double-check this path matches exactly where your S2.pkl is located relative to this script
INPUT_FILE = 'WESAD/S2/S2.pkl'
OUTPUT_PREFIX = 'WESAD/S2'
DATA_SAMPLING_RATE = 700
SEGMENT_DURATION_SEC = 60
warnings.filterwarnings('ignore')

# --- 1. Musical Constants & Modes ---
SCALE_MAJOR = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
SCALE_MINOR_HARM = [261.63, 293.66, 311.13, 349.23, 392.00, 415.30, 493.88, 523.25]
SCALE_LYDIAN = [261.63, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88, 523.25]

CHORDS = {
    'C': [261.63, 329.63, 392.00], 'Am': [220.00, 261.63, 329.63],
    'F': [174.61, 220.00, 261.63], 'G': [196.00, 246.94, 293.66],
}
BASE_PROG = ['C', 'Am', 'F', 'G']


# --- 2. Instruments ---
def get_kick(dur_ms=100):
    return Sine(60).to_audio_segment(duration=dur_ms).fade_out(60).apply_gain(0)


def get_snare(dur_ms=150):
    low = Sine(180).to_audio_segment(duration=dur_ms).apply_gain(-8)
    high = WhiteNoise().to_audio_segment(duration=dur_ms).high_pass_filter(2000).apply_gain(-12)
    return low.overlay(high).fade_out(100)


def get_hihat(dur_ms=50):
    return WhiteNoise().to_audio_segment(duration=dur_ms).high_pass_filter(8000).fade_out(40).apply_gain(-18)


def get_piano_note(freq, dur_ms=400):
    sine = Sine(freq).to_audio_segment(duration=dur_ms)
    saw = Sawtooth(freq).to_audio_segment(duration=dur_ms).low_pass_filter(1000).apply_gain(-15)
    return sine.overlay(saw).fade_in(5).fade_out(300).apply_gain(-8)


def get_guitar_chord(chord_name, intensity_0_to_1, dur_ms=2000):
    root_freqs = CHORDS.get(chord_name, CHORDS['C'])
    # Power chord = Root + 5th
    power_chord_freqs = [root_freqs[0], root_freqs[2]]
    guitar = AudioSegment.silent(duration=dur_ms)
    # EDA intensity controls filter brightness. Higher = brighter/buzzier.
    filter_cutoff = 500 + (intensity_0_to_1 * 3000)

    for freq in power_chord_freqs:
        # Sawtooth wave for electric guitar-like grit
        string = Sawtooth(freq / 2).to_audio_segment(duration=dur_ms)
        string = string.low_pass_filter(filter_cutoff)
        guitar = guitar.overlay(string)

    return guitar.fade_in(100).fade_out(500).apply_gain(-25 + (intensity_0_to_1 * 12))


def get_breathing_pad(chord_name, resp_val_0_to_1, dur_ms=500):
    pad_slice = AudioSegment.silent(duration=dur_ms)
    for freq in CHORDS.get(chord_name, CHORDS['C']):
        pad_slice = pad_slice.overlay(Sine(freq).to_audio_segment(duration=dur_ms))
    # Breathing value directly controls the volume of this pad slice
    volume_db = -40 + (resp_val_0_to_1 * 25)
    return pad_slice.apply_gain(volume_db).fade_in(100).fade_out(100)


def get_warmth_drone(temp_val_0_to_1, dur_ms=1000):
    # White noise filtered based on body temperature.
    # Warmer temp = higher cutoff frequency = "brighter" hiss.
    cutoff = 100 + (temp_val_0_to_1 * 800)
    volume = -35 + (temp_val_0_to_1 * 8)
    noise = WhiteNoise().to_audio_segment(duration=dur_ms).low_pass_filter(cutoff)
    return noise.apply_gain(volume).fade_in(500).fade_out(500)


# --- 3. Logic & Generation ---
def determine_musical_mode(hr_bpm, eda_norm, resp_rate_bpm, emg_norm):
    # Thresholds may need tuning based on specific subject data
    if hr_bpm < 75 and resp_rate_bpm < 15:
        return 'MEDITATION'
    if hr_bpm > 85 and eda_norm > 0.4:
        return 'STRESS'
    if hr_bpm > 75 and emg_norm > 0.2:
        return 'AMUSEMENT'
    return 'BASELINE'


def generate_song_structure(ecg_rate, emg, eda, resp, temp, rate, total_sec):
    print(f"--> Generating {total_sec}s of audio...")
    full_mix = AudioSegment.silent(duration=total_sec * 1000)

    current_time_ms = 0
    beat_counter = 0
    chord_idx = 0
    last_melody_idx = 0
    current_mode = 'BASELINE'

    # Pre-load drum samples for speed
    kick, snare, hihat = get_kick(), get_snare(), get_hihat()

    # --- Pre-process Signals ---
    # Normalize EMG (0-1)
    emg_clean = nk.emg_amplitude(emg)
    emg_norm = (emg_clean - emg_clean.min()) / (emg_clean.max() - emg_clean.min())

    # Normalize EDA (0-1)
    eda_clean = nk.eda_clean(eda, sampling_rate=rate)
    eda_norm = (eda_clean - eda_clean.min()) / (eda_clean.max() - eda_clean.min())

    # Process Respiration (Rate in BPM and normalized Swell 0-1)
    resp_clean = nk.rsp_clean(resp, sampling_rate=rate)
    resp_rate_sig = nk.rsp_rate(resp, sampling_rate=rate, window=rate * 10)
    resp_swell = (resp_clean - resp_clean.min()) / (resp_clean.max() - resp_clean.min())

    # Normalize Temperature (Fixed expected physiological range 30C-37C)
    temp_norm = np.clip((temp - 30.0) / (37.0 - 30.0), 0.0, 1.0)

    while current_time_ms < (total_sec * 1000) - 2000:
        # A. Get bio-data snapshot for this exact moment
        sec_idx = min(int((current_time_ms / 1000) * rate), len(ecg_rate) - 1)

        cur_bpm = np.clip(ecg_rate[sec_idx], 60, 140)
        ms_per_beat = 60000 / cur_bpm

        cur_eda = eda_norm[sec_idx]
        cur_emg = emg_norm[sec_idx]
        cur_resp_swell = resp_swell[sec_idx]
        cur_temp = np.mean(temp_norm[sec_idx:min(len(temp_norm), sec_idx + rate)])
        # Get average respiration rate over last 5 seconds for stability
        cur_resp_rate = np.mean(resp_rate_sig[max(0, sec_idx - rate * 5):sec_idx + 1]) if sec_idx > rate * 5 else 15

        # B. "Band Leader": Determine Mode every 4 beats
        if beat_counter % 4 == 0:
            new_mode = determine_musical_mode(cur_bpm, cur_eda, cur_resp_rate, cur_emg)
            if new_mode != current_mode:
                print(
                    f"[{int(current_time_ms / 1000)}s] Mode Switch: {current_mode} -> {new_mode} | HR: {cur_bpm:.0f}, RespRate: {cur_resp_rate:.1f}")
                current_mode = new_mode

        # --- C. MUSICAL LAYERS ---
        current_scale = SCALE_MAJOR

        # Layer 1: Always-on Textures (Temperature Drone & Breathing Pads)
        # Drone updates every beat for smooth temperature shifts
        full_mix = full_mix.overlay(get_warmth_drone(cur_temp, dur_ms=ms_per_beat), position=current_time_ms)
        # Pad swells match breathing exactly
        full_mix = full_mix.overlay(get_breathing_pad(BASE_PROG[chord_idx % 4], cur_resp_swell, dur_ms=ms_per_beat),
                                    position=current_time_ms)

        # Layer 2: Mode-Specific Instruments
        if current_mode == 'MEDITATION':
            current_scale = SCALE_LYDIAN
            # No drums. Sparse, echoing melody notes if muscles slightly active
            if beat_counter % 2 == 0 and cur_emg > 0.1:
                note = get_piano_note(current_scale[last_melody_idx % 8], dur_ms=2500).apply_gain(-15)
                full_mix = full_mix.overlay(note, position=current_time_ms)
                # Slow melody movement
                last_melody_idx += 1

        elif current_mode == 'STRESS':
            current_scale = SCALE_MINOR_HARM
            # Aggressive drums (Kick on every beat, busy hi-hats)
            full_mix = full_mix.overlay(kick.apply_gain(2), position=current_time_ms)
            full_mix = full_mix.overlay(hihat, position=current_time_ms)
            full_mix = full_mix.overlay(hihat.apply_gain(-5), position=current_time_ms + (ms_per_beat / 2))  # 8th notes
            if beat_counter % 2 == 1: full_mix = full_mix.overlay(snare, position=current_time_ms)
            # Distorted Guitar Power Chords on beat 1 of every bar
            if beat_counter % 4 == 0:
                guitar = get_guitar_chord(BASE_PROG[chord_idx % 4], cur_eda, dur_ms=ms_per_beat * 4)
                full_mix = full_mix.overlay(guitar, position=current_time_ms)

        elif current_mode == 'AMUSEMENT':
            # Upbeat Pop (Bouncy kick pattern, syncopated snare)
            if beat_counter % 4 == 0 or beat_counter % 4 == 2:  # Kick on 1 and 3
                full_mix = full_mix.overlay(kick, position=current_time_ms)
            if beat_counter % 4 == 1 or beat_counter % 4 == 3:  # Snare on 2 and 4
                full_mix = full_mix.overlay(snare, position=current_time_ms)
            # Off-beat hi-hats
            full_mix = full_mix.overlay(hihat, position=current_time_ms + (ms_per_beat / 2))

            # Active, plucky melody driven by EMG
            if cur_emg > 0.15:
                note = get_piano_note(current_scale[last_melody_idx % 8], dur_ms=300)
                full_mix = full_mix.overlay(note, position=current_time_ms)
                # Faster melody movement
                last_melody_idx += 1 if cur_emg > 0.4 else -1

        else:  # BASELINE
            # Chill beat
            if beat_counter % 4 == 0: full_mix = full_mix.overlay(kick, position=current_time_ms)
            if beat_counter % 4 == 2: full_mix = full_mix.overlay(snare, position=current_time_ms)
            full_mix = full_mix.overlay(hihat.apply_gain(-10), position=current_time_ms)

        # Advance time based on current subject heart rate
        current_time_ms += ms_per_beat
        beat_counter += 1
        if beat_counter % 4 == 0: chord_idx += 1

    return full_mix


# --- 4. Main ---
def main():
    print(f"Starting up... attempting to load {INPUT_FILE}")
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File not found at {os.path.abspath(INPUT_FILE)}")
        return

    try:
        with open(INPUT_FILE, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        return

    try:
        # KEY CORRECTION HERE BASED ON YOUR DIAGNOSTIC OUTPUT
        chest = data['signal']['chest']
        ecg = chest['ECG'].flatten()
        emg = chest['EMG'].flatten()
        eda = chest['EDA'].flatten()
        resp = chest['Resp'].flatten()  # Corrected from RESP
        temp = chest['Temp'].flatten()  # Corrected from TEMP
        labels = data['label'].flatten()
        print("✅ Data keys found and signals loaded.")
    except KeyError as e:
        print(f"❌ missing key in data structure: {e}")
        print("Double check standard WESAD keys: ECG, EMG, EDA, Resp, Temp")
        return

    # WESAD Label IDs
    segments_to_process = {
        'baseline': 1,
        'stress': 2,
        'fun': 3,
        'meditation': 4
    }

    samples_needed = SEGMENT_DURATION_SEC * DATA_SAMPLING_RATE

    for label_name, label_id in segments_to_process.items():
        print(f"\n--- finding {label_name.upper()} segment (ID: {label_id}) ---")
        indices = np.where(labels == label_id)[0]

        if len(indices) == 0:
            print(f"Warning: No data found for {label_name}")
            continue

        # Take the middle of the condition to ensure stable data
        mid_point_idx = len(indices) // 2
        start_index = indices[mid_point_idx]
        end_index = start_index + samples_needed

        if end_index > len(ecg):
            print("Not enough data remaining for a full segment.")
            continue

        print(f"Processing {SEGMENT_DURATION_SEC}s segment...")

        # 1. Calculate HR for this specific segment first
        ecg_segment = ecg[start_index:end_index]
        ecg_clean = nk.ecg_clean(ecg_segment, sampling_rate=DATA_SAMPLING_RATE)
        try:
            # Find peaks and calculate rate
            _, rpeaks = nk.ecg_peaks(ecg_clean, sampling_rate=DATA_SAMPLING_RATE)
            ecg_rate = nk.signal_rate(rpeaks, sampling_rate=DATA_SAMPLING_RATE, desired_length=len(ecg_segment))
        except Exception as e:
            print(f"Could not extract Heart Rate for this segment: {e}")
            continue

        # 2. Generate Song with all sensors
        final_song = generate_song_structure(
            ecg_rate=ecg_rate,
            emg=emg[start_index:end_index],
            eda=eda[start_index:end_index],
            resp=resp[start_index:end_index],
            temp=temp[start_index:end_index],
            rate=DATA_SAMPLING_RATE,
            total_sec=SEGMENT_DURATION_SEC
        )

        # 3. Export
        output_filename = f"{OUTPUT_PREFIX}_{label_name}_full_bio.wav"
        # Ensure directory exists
        os.makedirs(os.path.dirname(OUTPUT_PREFIX), exist_ok=True)

        try:
            final_song.export(output_filename, format="wav")
            print(f"🎹 Success! Saved to: {output_filename}")
        except Exception as e:
            print(f"❌ Could not export audio (did you install ffmpeg?): {e}")


if __name__ == "__main__":
    main()