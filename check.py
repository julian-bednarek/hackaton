import pickle
import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings

# --- CONFIGURATION ---
SUBJECT_ID = 'S2'
SAMPLING_RATE = 700
warnings.filterwarnings('ignore')

# --- 0. ROBUST FILE FINDER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
	os.path.join(BASE_DIR, 'WESAD', SUBJECT_ID, f"{SUBJECT_ID}.pkl"),
	os.path.join(BASE_DIR, SUBJECT_ID, f"{SUBJECT_ID}.pkl"),
	os.path.join(BASE_DIR, f"{SUBJECT_ID}.pkl"),
	# Add your specific hardcoded path here as a LAST RESORT:
	r'C:\Users\Mrlio\Desktop\Hackathon\hackaton\WESAD\S2\S2.pkl'
]

FILE_PATH = None
for path in POSSIBLE_PATHS:
	if os.path.exists(path):
		FILE_PATH = path
		print(f"✅ FOUND DATA AT: {FILE_PATH}")
		break

if FILE_PATH is None:
	print(f"\n❌ ERROR: Could not find {SUBJECT_ID}.pkl")
	exit()


# --- 1. LOAD DATA ---
def load_data(filepath):
	print("Loading pickle file... (this might take 10-20 seconds)")
	with open(filepath, 'rb') as f:
		data = pickle.load(f, encoding='latin1')
	return data


# --- 2. EXTRACT FEATURES (FIXED AGAIN) ---
def extract_features(data):
	print("Extracting signals...")
	chest = data['signal']['chest']
	labels = data['label'].flatten()

	keys = {k.lower(): k for k in chest.keys()}

	def get_signal(name): return chest[keys[name.lower()]].flatten()

	df = pd.DataFrame({
		'ECG': get_signal('ECG'),
		'EDA': get_signal('EDA'),
		'EMG': get_signal('EMG'),
		'RESP': get_signal('RESP'),
		'TEMP': get_signal('TEMP'),
		'Label': labels
	})

	# Filter for main conditions only
	df = df[df['Label'].isin([1, 2, 3, 4])].copy()

	print("Processing physiological features...")
	# NOTE: We use .to_numpy() to avoid Pandas indexing errors

	# 1. Heart Rate from ECG
	ecg_cleaned = nk.ecg_clean(df['ECG'].to_numpy(), sampling_rate=SAMPLING_RATE)
	_, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE)
	# signal_rate NEEDS desired_length because it works from peaks (just a list of points)
	df['Heart_Rate'] = nk.signal_rate(rpeaks, sampling_rate=SAMPLING_RATE, desired_length=len(df))

	# 2. Phasic/Tonic from EDA
	df['EDA_Level'] = nk.eda_process(df['EDA'].to_numpy(), sampling_rate=SAMPLING_RATE)[0]['EDA_Clean']

	# 3. Amplitude from EMG (Muscle tension)
	df['EMG_Amp'] = nk.emg_amplitude(df['EMG'].to_numpy())

	# 4. Respiration Rate - FIXED LINE BELOW
	rsp_cleaned = nk.rsp_clean(df['RESP'].to_numpy(), sampling_rate=SAMPLING_RATE)
	# rsp_rate automatically matches the length of rsp_cleaned input
	df['Resp_Rate'] = nk.rsp_rate(rsp_cleaned, sampling_rate=SAMPLING_RATE)

	# 5. Temperature
	df['Temp_Mean'] = df['TEMP']

	return df


# --- 3. PLOTTING ---
def plot_data(df):
	print("Generating inspection plots...")
	plt.figure(figsize=(12, 10))

	samples_to_plot = min(300 * SAMPLING_RATE, len(df))
	df_slice = df.iloc[:samples_to_plot]
	time_axis = np.arange(len(df_slice)) / SAMPLING_RATE

	signals = [
		('Heart_Rate', 'Heart Rate (BPM) -> KICK', 'red'),
		('EDA_Level', 'EDA (Sweat) -> HATS', 'blue'),
		('EMG_Amp', 'EMG (Muscle) -> BASS', 'orange'),
		('Resp_Rate', 'Resp Rate -> MELODY', 'green'),
		('Temp_Mean', 'Skin Temp -> PADS', 'purple')
	]

	for i, (col, title, color) in enumerate(signals):
		plt.subplot(5, 1, i + 1)
		plt.plot(time_axis, df_slice[col], color=color, label=col)
		plt.title(title)
		plt.grid(True, alpha=0.3)
		if i == 4: plt.xlabel("Time (seconds)")

	plt.tight_layout()
	plt.show()


# --- 4. EXPORT JSON ---
def export_for_music_app(df):
	print("\n--- STARTING EXPORT ---")
	print("1. Downsampling to 1Hz...")
	df['second'] = np.arange(len(df)) // SAMPLING_RATE
	music_df = df.groupby('second')[['Heart_Rate', 'EDA_Level', 'EMG_Amp', 'Resp_Rate', 'Temp_Mean']].mean()

	print("2. Normalizing...")
	scaler = MinMaxScaler(feature_range=(0.0, 1.0))
	norm_data = scaler.fit_transform(music_df)
	export_df = pd.DataFrame(norm_data, columns=['kick', 'hats', 'bass', 'melody', 'pads'])

	export_df['bio_state'] = np.where(
		(export_df['kick'] + export_df['hats']) > 1.3, 'High Arousal',
		np.where((export_df['kick'] + export_df['hats']) < 0.6, 'Calm', 'Neutral')
	)

	export_df['time'] = export_df.index

	output_filename = f"music_data_{SUBJECT_ID.lower()}.json"
	export_df.to_json(output_filename, orient='records')
	print(f"✅ SUCCESS! Exported {len(export_df)} seconds to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
	data = load_data(FILE_PATH)
	if data is not None:
		df_features = extract_features(data)
		plot_data(df_features)
		export_for_music_app(df_features)