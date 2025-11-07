import pickle
import numpy as np
import neurokit2 as nk
from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from scipy.signal import find_peaks
import warnings

# --- Konfiguracja ---

# Nazwa pliku wejściowego (musi być w tym samym folderze)
INPUT_FILE = 'S2/S2.pkl'

# Nazwa pliku wyjściowego
OUTPUT_FILE = 'S2_beat.wav'

# Szybkość próbkowania danych z WESAD (z dokumentacji)
DATA_SAMPLING_RATE = 700  # 700 Hz [cite: 27]

# Szybkość próbkowania dla pliku audio (standardowa)
AUDIO_SAMPLING_RATE = 44100  # 44.1 kHz

# Tłumienie ostrzeżeń, aby logi były czystsze
warnings.filterwarnings('ignore')

# --- 1. Funkcje do generowania dźwięków ---

def create_kick(duration_ms=100):
    """Generuje prosty dźwięk bębna basowego (kick)."""
    # Dźwięk sinusoidalny o niskiej częstotliwości (60 Hz)
    kick_sound = Sine(60).to_audio_segment(duration=duration_ms)
    # Szybkie wyciszenie, aby brzmiało jak "kopnięcie"
    kick_sound = kick_sound.fade_out(60).apply_gain(-10)
    return kick_sound

def create_hihat(duration_ms=50):
    """Generuje prosty dźwięk hi-hatu (biały szum)."""
    hihat_sound = WhiteNoise().to_audio_segment(duration=duration_ms)
    # Szybkie wyciszenie i podgłośnienie, aby brzmiało jak "cyknięcie"
    hihat_sound = hihat_sound.fade_in(5).fade_out(25).apply_gain(-25)
    return hihat_sound

def create_snare(duration_ms=150):
    """Generuje prosty dźwięk werbla (snare)."""
    snare_sound = WhiteNoise().to_audio_segment(duration=duration_ms)
    # Filtrowanie i wyciszanie, aby brzmiało bardziej jak werbel
    # Symulujemy band-pass filter używając high-pass i low-pass
    snare_sound = snare_sound.high_pass_filter(1000)
    snare_sound = snare_sound.low_pass_filter(5000)
    snare_sound = snare_sound.fade_out(100).apply_gain(-15)
    return snare_sound

# --- 2. Główne funkcje przetwarzające ---

def load_wesad_data(filepath):
    """Ładuje i zwraca sygnały oraz etykiety z pliku .pkl."""
    print(f"Ładowanie danych z {filepath}...")
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{filepath}'. Upewnij się, że jest w tym samym folderze.")
        return None
    
    # Wyciągamy tylko potrzebne dane [cite: 59, 60, 61]
    signals = data['signal']['chest']
    labels = data['label'].flatten()
    
    # Ograniczamy dane do pierwszej minuty (700 Hz * 60 sekund)
    samples_per_minute = 700 * 60
    ecg = signals['ECG'].flatten()[:samples_per_minute]
    eda = signals['EDA'].flatten()[:samples_per_minute]
    emg = signals['EMG'].flatten()[:samples_per_minute]
    labels = labels[:samples_per_minute]
    
    print("Dane załadowane pomyślnie (pierwsza minuta)")
    return ecg, eda, emg, labels

def find_event_indices(signal, **kwargs):
    """Uniwersalna funkcja do znajdowania pików w sygnale."""
    # Używamy find_peaks, bo jest szybki i wystarczający
    indices, _ = find_peaks(signal, **kwargs)
    return indices

def process_signals(ecg, eda, emg, rate):
    """Przetwarza surowe sygnały na listy indeksów zdarzeń."""
    
    # 1. EKG -> Bęben Basowy (Kick)
    print("Przetwarzanie EKG (na bęben basowy)...")
    # Znajdujemy załamki R (piki) w EKG
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=rate)
    # Bierzemy co 32-gą próbkę aby skrócić czas przetwarzania
    kick_indices = rpeaks['ECG_R_Peaks']
    
    # 2. EDA -> Hi-Hat
    print("Przetwarzanie EDA (na hi-hat)...")
    # Przetwarzamy EDA, aby znaleźć piki fazowe (szybkie reakcje)
    eda_phasic = nk.eda_phasic(nk.eda_clean(eda, sampling_rate=rate), sampling_rate=rate)['EDA_Phasic']
    # Znajdujemy piki w sygnale fazowym (wysokość piku > 0.01)
    hihat_indices = find_event_indices(eda_phasic, height=0.01)
    
    # 3. EMG -> Werbel (Snare)
    print("Przetwarzanie EMG (na werbel)...")
    # Obliczamy amplitudę (siłę) sygnału EMG
    emg_amplitude = nk.emg_amplitude(emg)
    # Znajdujemy tylko silne skurcze (np. powyżej 95. percentyla)
    threshold = np.percentile(emg_amplitude, 95)
    snare_indices = find_event_indices(emg_amplitude, height=threshold)
    
    print("Przetwarzanie sygnałów zakończone.")
    return kick_indices, hihat_indices, snare_indices

def create_beat(labels, rate, kick_indices, hihat_indices, snare_indices, samples):
    """Tworzy plik audio, aranżując beat zgodnie z etykietami."""
    
    total_samples = len(labels)
    duration_ms = total_samples / rate * 1000
    
    print(f"Tworzenie pustej ścieżki audio (długość: {duration_ms / 1000:.2f} s)...")
    # Tworzymy cichą ścieżkę audio o pełnej długości
    output_audio = AudioSegment.silent(duration=duration_ms)
    
    # Funkcja pomocnicza do konwersji indeksu na milisekundy
    def to_ms(index):
        return (index / rate) * 1000

    # --- ARANŻACJA UTWORU ---
    
    print("Nakładanie bębna basowego (EKG)...")
    for index in kick_indices:
        label = labels[index]
        pos_ms = to_ms(index)
        
        # Bęben basowy gra zawsze, ale ciszej podczas medytacji
        if label == 4:  # 4 = Medytacja 
            output_audio = output_audio.overlay(samples['kick'] - 6, position=pos_ms) # Ciszej o 6dB
        else:
            output_audio = output_audio.overlay(samples['kick'], position=pos_ms)

    print("Nakładanie hi-hatu (EDA)...")
    for index in hihat_indices:
        label = labels[index]
        pos_ms = to_ms(index)
        
        # Hi-hat gra tylko podczas stresu i rozbawienia (wysokie pobudzenie)
        if label == 2 or label == 3:  # 2 = Stres, 3 = Rozbawienie 
            output_audio = output_audio.overlay(samples['hihat'], position=pos_ms)

    print("Nakładanie werbla (EMG)...")
    for index in snare_indices:
        label = labels[index]
        pos_ms = to_ms(index)
        
        # Werbel gra tylko podczas stresu (jako akcenty napięcia)
        if label == 2:  # 2 = Stres 
            output_audio = output_audio.overlay(samples['snare'], position=pos_ms)
            
    return output_audio

# --- 3. Główny blok wykonawczy ---

def main():
    print("--- Sonifikator Danych WESAD ---")
    
    # 1. Załaduj dane
    data_tuple = load_wesad_data(INPUT_FILE)
    if data_tuple is None:
        return
    ecg, eda, emg, labels = data_tuple
    
    # 2. Wygeneruj/Załaduj sample
    print("Generowanie sampli audio...")
    samples = {
        'kick': create_kick(),
        'hihat': create_hihat(),
        'snare': create_snare()
    }
    
    # 3. Znajdź zdarzenia w sygnałach
    kick_idx, hihat_idx, snare_idx = process_signals(ecg, eda, emg, DATA_SAMPLING_RATE)
    
    # 4. Stwórz utwór
    final_beat = create_beat(labels, DATA_SAMPLING_RATE, kick_idx, hihat_idx, snare_idx, samples)
    
    # 5. Zapisz plik
    print(f"Eksportowanie pliku do {OUTPUT_FILE}...")
    final_beat.export(OUTPUT_FILE, format="wav")
    
    print("\n--- GOTOWE! ---")
    print(f"Twój beat został zapisany jako '{OUTPUT_FILE}'.")
    print("\nSłuchaj uważnie, a usłyszysz:")
    print(" - Bęben basowy (EKG) przyspieszający podczas stresu.")
    print(" - Hi-haty (EDA) pojawiające się tylko w fazie stresu i rozbawienia.")
    print(" - Werbel (EMG) uderzający rzadko, tylko podczas pików napięcia mięśniowego w stresie.")
    print(" - Fazy medytacji będą znacznie cichsze i spokojniejsze.")

if __name__ == "__main__":
    main()