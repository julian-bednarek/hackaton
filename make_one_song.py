import json
import numpy as np
from scipy.io.wavfile import write
import os

# --- CONFIGURATION ---
JSON_FILE = 'music_data_s2.json'
OUTPUT_FILE = 'static/audio/wesad_symphony_1min.wav'
SAMPLE_RATE = 44100
DURATION_LIMIT = 60  # Generate only 60 seconds for testing
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# --- MUSICAL UTILITIES ---
SCALE = [130.81, 155.56, 174.61, 196.00, 233.08, 261.63, 311.13, 349.23, 392.00, 466.16]


def get_note(value_0_to_1):
	index = int(value_0_to_1 * (len(SCALE) - 1))
	return SCALE[index]


# --- SYNTH INSTRUMENTS ---
def play_kick(vol):
	if vol < 0.1: return np.zeros(SAMPLE_RATE)
	t = np.linspace(0, 1.0, SAMPLE_RATE, False)
	freq = np.linspace(150, 50, SAMPLE_RATE)
	amp = np.exp(-5 * t) * vol
	return np.sin(2 * np.pi * freq * t) * amp


def play_bass(vol):
	t = np.linspace(0, 1.0, SAMPLE_RATE, False)
	wave = np.sign(np.sin(2 * np.pi * 65.41 * t)) * 0.3
	return wave * vol


def play_melody(val_0_to_1, vol):
	freq = get_note(val_0_to_1)
	t = np.linspace(0, 1.0, SAMPLE_RATE, False)
	vibrato = np.sin(2 * np.pi * 6 * t) * 0.01
	wave = np.sin(2 * np.pi * freq * (t + vibrato)) * 0.4
	return wave * vol


def play_hats(stress_vol):
	t = np.linspace(0, 1.0, SAMPLE_RATE, False)
	noise = np.random.uniform(-0.5, 0.5, SAMPLE_RATE)
	if stress_vol > 0.6:
		env = np.abs(np.sin(2 * np.pi * 8 * t)) * np.exp(-5 * t % 0.125)
	else:
		env = np.abs(np.sin(2 * np.pi * 4 * t)) * np.exp(-5 * t % 0.25)
	return noise * env * stress_vol * 0.5


def play_pad(vol):
	t = np.linspace(0, 1.0, SAMPLE_RATE, False)
	root = np.sin(2 * np.pi * 130.81 * t)
	third = np.sin(2 * np.pi * 155.56 * t)
	fifth = np.sin(2 * np.pi * 196.00 * t)
	return (root + third + fifth) * 0.2 * vol


# --- THE COMPOSER ---
def compose():
	print(f"🎼 READING {JSON_FILE}...")
	try:
		with open(JSON_FILE, 'r') as f:
			data = json.load(f)
	except FileNotFoundError:
		print("❌ ERROR: JSON not found.")
		return

	# LIMIT TO 60 SECONDS
	data = data[:DURATION_LIMIT]
	print(f"🎹 COMPOSING 1-MINUTE SNIPPET ({len(data)} seconds)...")

	full_song = []

	for i, sec in enumerate(data):
		if i % 10 == 0: print(f"Rendering second {i}...")

		# SAFE DATA READING (Handles None/null values by defaulting to 0.0)
		v_kick = float(sec.get('kick') or 0.0)
		v_bass = float(sec.get('bass') or 0.0)
		v_mel_val = float(sec.get('melody') or 0.0)
		v_hats = float(sec.get('hats') or 0.0)
		v_pads = float(sec.get('pads') or 0.0)

		kick = play_kick(v_kick)
		bass = play_bass(v_bass)
		melody = play_melody(v_mel_val, vol=0.6)
		hats = play_hats(v_hats)
		pad = play_pad(v_pads)

		mix_chunk = kick + bass + melody + hats + pad
		full_song.append(mix_chunk)

	print("💾 SAVING AUDIO...")
	master_track = np.concatenate(full_song)
	max_val = np.max(np.abs(master_track))
	if max_val > 0:
		master_track = master_track / max_val * 0.9

	write(OUTPUT_FILE, SAMPLE_RATE, np.int16(master_track * 32767))
	print(f"✅ DONE! 1-minute song saved to: {os.path.abspath(OUTPUT_FILE)}")


if __name__ == '__main__':
	compose()