import pickle
import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Configuration ---

# <<<<< CHANGE THIS TO YOUR SUBJECT ID >>>>>
SUBJECT_ID = 'S2' # Example: 'S2', 'S3', 'S4', etc.

# <<<<< CHANGE THIS TO THE PATH OF YOUR .pkl FILE >>>>>
# Build the path relative to this script's directory so the script
# can be run from any working directory and still find the subject .pkl
FILE_PATH = os.path.join(os.path.dirname(__file__), f"{SUBJECT_ID}.pkl")

# The RespiBAN sampling rate is 700 Hz [cite: 27]
SAMPLING_RATE = 700

# Suppress warnings from neurokit for cleaner output
warnings.filterwarnings('ignore')

# --- Functions ---

def load_data(filepath):
    """
    Loads the synchronized .pkl data file for a subject.
    """
    print(f"Loading data from {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        print("Please make sure the FILE_PATH variable is correct.")
        return None
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None

def extract_features_from_chest(data):
    """
    Extracts features from the chest data, which is synchronized
    with the 700Hz labels[cite: 60, 61].
    """
    print("Extracting signals and labels...")
    
    # Get labels and chest signals
    labels = data['label'].flatten()
    chest_signals = data['signal']['chest']

    # Normalize channel lookup to be case-insensitive and robust
    available = {k.lower(): k for k in chest_signals.keys()}

    def pick(key_name):
        """Return the actual key name for a desired channel (case-insensitive)."""
        k = available.get(key_name.lower())
        if k is None:
            raise KeyError(f"Required channel '{key_name}' not found. Available chest channels: {list(chest_signals.keys())}")
        return chest_signals[k].flatten()

    # Create a DataFrame using robust channel lookup
    df = pd.DataFrame({
        'ECG': pick('ECG'),
        'EDA': pick('EDA'),
        'EMG': pick('EMG'),
        'RESP': pick('RESP'),
        'Label': labels
    })
    
    print("Filtering for relevant conditions (1, 2, 3, 4)...")
    # Filter for the 4 main conditions 
    df = df[df['Label'].isin([1, 2, 3, 4])].copy()
    
    # Map labels to meaningful names
    label_map = {
        1: 'Baseline',
        2: 'Stress',
        3: 'Amusement',
        4: 'Meditation'
    }
    df['Condition'] = df['Label'].map(label_map)
    
    print("Processing physiological features (this may take a minute)...")
    
    # --- Feature Engineering ---
    # We process the signals to get meaningful features.
    
    # 1. ECG -> Heart Rate
    # Clean the ECG signal
    ecg_cleaned = nk.ecg_clean(df['ECG'].to_numpy(), sampling_rate=SAMPLING_RATE)
    # Find R-peaks (the 'spikes' of the heartbeat)
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=SAMPLING_RATE)
    # Calculate the instantaneous heart rate
    hr = nk.signal_rate(rpeaks, sampling_rate=SAMPLING_RATE, desired_length=len(df))
    df['Heart_Rate'] = hr
    
    # 2. EDA -> Skin Conductance Level (Tonic)
    # Process EDA to separate tonic (slow level) and phasic (fast peaks)
    eda_signals, _ = nk.eda_process(df['EDA'].to_numpy(), sampling_rate=SAMPLING_RATE)
    # eda_signals['EDA_Clean'] will be a numpy array or pandas Series; convert to numpy
    df['EDA_Level'] = np.asarray(eda_signals['EDA_Clean'])
    
    # 3. RESP -> Respiration Rate
    # Clean respiration signal
    resp_cleaned = nk.rsp_clean(df['RESP'].to_numpy(), sampling_rate=SAMPLING_RATE)
    # Calculate instantaneous respiration rate
    # nk.rsp_rate doesn't accept 'desired_length' in some versions, and
    # typically returns an array matching the input length — call without it.
    resp_rate = nk.rsp_rate(resp_cleaned, sampling_rate=SAMPLING_RATE)
    df['Resp_Rate'] = resp_rate
    
    # 4. EMG -> Muscle Activation
    # Calculate the amplitude/envelope of the muscle signal
    emg_amplitude = nk.emg_amplitude(df['EMG'].to_numpy())
    df['EMG_Amplitude'] = emg_amplitude
    
    print("Feature extraction complete.")
    return df

def plot_all_signals(data):
    """
    Plot all raw signals from both RespiBAN and E4 devices with condition markers.
    Shows ACC, ECG, EDA, EMG, RESP, TEMP from RespiBAN and 
    ACC, BVP, EDA, TEMP from E4 data correlated to sample numbers.
    Adds shaded regions to show different conditions (baseline, stress, etc).
    """
    print("Generating raw signal plots...")
    
    # Set the plot style to something clean and modern
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = True
    
    # Get signals
    chest = data['signal']['chest']
    wrist = data['signal']['wrist']
    labels = data['label'].flatten()
    
    # Sampling rates and time axes
    chest_rate = 700  # Hz for RespiBAN
    wrist_acc_rate = 32  # Hz for E4 ACC
    wrist_bvp_rate = 64  # Hz for E4 BVP
    wrist_eda_rate = 4   # Hz for E4 EDA/TEMP
    
    # Create time axes (in seconds)
    chest_time = np.arange(len(labels)) / chest_rate
    wrist_acc_time = np.arange(len(wrist['ACC'])) / wrist_acc_rate
    wrist_bvp_time = np.arange(len(wrist['BVP'])) / wrist_bvp_rate
    wrist_eda_time = np.arange(len(wrist['EDA'])) / wrist_eda_rate
    
    # Full condition mapping
    condition_map = {
        0: 'Not Defined',
        1: 'Baseline',
        2: 'Stress',
        3: 'Amusement',
        4: 'Meditation',
        5: 'Should not occur',
        6: 'Should not occur',
        7: 'Should not occur'
    }
    
    # Colors for conditions (skip 0, 5, 6, 7 as they shouldn't appear in the selected regions)
    condition_colors = {
        1: 'lightgreen',
        2: 'salmon',
        3: 'lightblue',
        4: 'lavender'
    }
    
    # Create subplots - 10 signals total (6 RespiBAN + 4 E4)
    fig, axes = plt.subplots(10, 1, figsize=(15, 25), sharex=True)
    fig.suptitle(f'Raw Physiological Signals from RespiBAN and E4 Devices (Subject {SUBJECT_ID})', 
                 fontsize=16, y=0.92)
    
    # Function to add condition backgrounds
    def add_condition_backgrounds(ax, time):
        for cond in [1, 2, 3, 4]:  # Only show main conditions
            mask = labels == cond
            if not any(mask):  # Skip if condition doesn't exist
                continue
            # Find continuous regions
            changes = np.diff(np.concatenate(([0], mask, [0])))
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            for start, end in zip(starts, ends):
                ax.axvspan(start/chest_rate, end/chest_rate, 
                          alpha=0.2, color=condition_colors[cond], 
                          label=condition_map[cond] if start == starts[0] else "")
    
    # Plot RespiBAN signals
    # 1. ACC
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axes[0].plot(chest_time, chest['ACC'][:, i], 
                    label=f'ACC-{axis}', alpha=0.7, linewidth=1)
    axes[0].set_title('RespiBAN ACC')
    axes[0].legend()
    add_condition_backgrounds(axes[0], chest_time)
    
    # 2. ECG
    axes[1].plot(chest_time, chest['ECG'].flatten(), 'g-', 
                 label='ECG', alpha=0.7, linewidth=1)
    axes[1].set_title('RespiBAN ECG')
    axes[1].legend()
    add_condition_backgrounds(axes[1], chest_time)
    
    # 3. EMG
    axes[2].plot(chest_time, chest['EMG'].flatten(), 'r-', 
                 label='EMG', alpha=0.7, linewidth=1)
    axes[2].set_title('RespiBAN EMG')
    axes[2].legend()
    add_condition_backgrounds(axes[2], chest_time)
    
    # 4. EDA (RespiBAN)
    axes[3].plot(chest_time, chest['EDA'].flatten(), 'b-', 
                 label='EDA', alpha=0.7, linewidth=1)
    axes[3].set_title('RespiBAN EDA')
    axes[3].legend()
    add_condition_backgrounds(axes[3], chest_time)
    
    # 5. Resp
    axes[4].plot(chest_time, chest['Resp'].flatten(), 'm-', 
                 label='RESP', alpha=0.7, linewidth=1)
    axes[4].set_title('RespiBAN RESP')
    axes[4].legend()
    add_condition_backgrounds(axes[4], chest_time)
    
    # 6. Temp (RespiBAN)
    axes[5].plot(chest_time, chest['Temp'].flatten(), 'k-', 
                 label='TEMP', alpha=0.7, linewidth=1)
    axes[5].set_title('RespiBAN Temperature')
    axes[5].legend()
    add_condition_backgrounds(axes[5], chest_time)
    
    # Plot E4 signals
    # 7. ACC
    for i, axis in enumerate(['X', 'Y', 'Z']):
        axes[6].plot(wrist_acc_time, wrist['ACC'][:, i], 
                    label=f'ACC-{axis}', alpha=0.7, linewidth=1)
    axes[6].set_title('E4 ACC')
    axes[6].legend()
    add_condition_backgrounds(axes[6], chest_time)
    
    # 8. BVP
    axes[7].plot(wrist_bvp_time, wrist['BVP'].flatten(), 'g-', 
                 label='BVP', alpha=0.7, linewidth=1)
    axes[7].set_title('E4 BVP')
    axes[7].legend()
    add_condition_backgrounds(axes[7], chest_time)
    
    # 9. EDA (E4)
    axes[8].plot(wrist_eda_time, wrist['EDA'].flatten(), 'b-', 
                 label='EDA', alpha=0.7, linewidth=1)
    axes[8].set_title('E4 EDA')
    axes[8].legend()
    add_condition_backgrounds(axes[8], chest_time)
    
    # 10. Temp (E4)
    axes[9].plot(wrist_eda_time, wrist['TEMP'].flatten(), 'k-', 
                 label='TEMP', alpha=0.7, linewidth=1)
    axes[9].set_title('E4 Temperature')
    axes[9].legend()
    add_condition_backgrounds(axes[9], chest_time)
    
    # Add legend for conditions at the bottom
    handles = [plt.Rectangle((0,0),1,1, color=color, alpha=0.2) 
              for color in condition_colors.values()]
    labels = [condition_map[cond] for cond in condition_colors.keys()]
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), 
              ncol=len(condition_colors), title='Conditions')
    
    # Adjust layout and add common x-label
    plt.xlabel('Time (seconds)')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for condition legend
    
    return fig

def plot_feature_correlations(df):
    """
    Generates box plots to compare features across conditions.
    """
    print("Generating plots...")
    
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Define the order for the x-axis
    condition_order = ['Baseline', 'Amusement', 'Stress', 'Meditation']
    
    # Create a 2x2 grid for the plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Physiological Responses by Condition for Subject {SUBJECT_ID}', fontsize=20, y=1.03)
    
    # Plot 1: Heart Rate
    sns.boxplot(ax=axes[0, 0], data=df, x='Condition', y='Heart_Rate', order=condition_order)
    axes[0, 0].set_title('Heart Rate (BPM)', fontsize=16)
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('BPM')
    
    # Plot 2: EDA Level
    sns.boxplot(ax=axes[0, 1], data=df, x='Condition', y='EDA_Level', order=condition_order)
    axes[0, 1].set_title('Skin Conductance (EDA) Level', fontsize=16)
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('microsiemens (µS)')
    
    # Plot 3: Respiration Rate
    sns.boxplot(ax=axes[1, 0], data=df, x='Condition', y='Resp_Rate', order=condition_order)
    axes[1, 0].set_title('Respiration Rate (Breaths/min)', fontsize=16)
    axes[1, 0].set_xlabel('Condition', fontsize=12)
    axes[1, 0].set_ylabel('Breaths/min')
    
    # Plot 4: EMG Amplitude
    sns.boxplot(ax=axes[1, 1], data=df, x='Condition', y='EMG_Amplitude', order=condition_order)
    axes[1, 1].set_title('Muscle Activation (EMG) Amplitude', fontsize=16)
    axes[1, 1].set_xlabel('Condition', fontsize=12)
    axes[1, 1].set_ylabel('Amplitude (a.u.)')
    
    plt.tight_layout()
    plt.show()

# --- Main execution ---

if __name__ == "__main__":
    data = load_data(FILE_PATH)
    
    if data:
        # First plot the raw signals from both devices
        print("\n--- Plotting Raw Signals from RespiBAN and E4 ---")
        raw_fig = plot_all_signals(data)
        plt.show()
        
        # Then extract and plot the features
        features_df = extract_features_from_chest(data)
        
        # Display summary statistics in the terminal
        print("\n--- Mean Feature Values by Condition ---")
        summary_stats = features_df[['Condition', 'Heart_Rate', 'EDA_Level', 'Resp_Rate', 'EMG_Amplitude']].groupby('Condition').mean()
        print(summary_stats.loc[['Baseline', 'Amusement', 'Stress', 'Meditation']])
        print("----------------------------------------\n")
        
        # Show the feature correlation plots
        plot_feature_correlations(features_df)