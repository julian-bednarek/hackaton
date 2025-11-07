import pickle
import numpy as np
import os
from django.db import transaction
from subjects.models import Subject, AffectCondition, PhysiologicalFeature

class WESADProcessor:
    
    SAMPLE_RATE = 700
    WINDOW_SECONDS = 30
    WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SECONDS

    def __init__(self, subject_code, data_directory='.'):
        self.subject_code = subject_code
        self.data_directory = data_directory
        self.filepath = os.path.join(data_directory, f'{subject_code}.pkl')

    def _load_data(self):
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found at {self.filepath}")
        
        with open(self.filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        return data

    def _get_db_objects(self):
        subject_obj = Subject.objects.get(code=self.subject_code)
        condition_map = {c.condition_id: c for c in AffectCondition.objects.all()}
        return subject_obj, condition_map

    def _calculate_device_features(self, device_data, device_location, window_slice):
        features = {
            'hrv_rmssd': None,
            'eda_mean': None,
            'acc_std_x': None,
            'bvp_mean': None,
        }
        
        if 'EDA' in device_data:
            features['eda_mean'] = np.mean(device_data['EDA'][window_slice].flatten())
        
        if 'ACC' in device_data:
            features['acc_std_x'] = np.std(device_data['ACC'][window_slice][:, 0])

        if device_location == 'CHEST' and 'ECG' in device_data:
            ecg_window = device_data['ECG'][window_slice].flatten()
            features['hrv_rmssd'] = np.std(ecg_window)

        elif device_location == 'WRIST' and 'BVP' in device_data:
            bvp_window = device_data['BVP'][window_slice].flatten()
            features['bvp_mean'] = np.mean(bvp_window)
            features['hrv_rmssd'] = np.std(bvp_window)

        return features

    def _extract_features(self, data, subject_obj, condition_map):
        labels = data['label']
        signal_data = data['signal']
        signal_length = len(labels)
        features_to_create = []

        for start_idx in range(0, signal_length - self.WINDOW_SAMPLES, self.WINDOW_SAMPLES):
            end_idx = start_idx + self.WINDOW_SAMPLES
            time_start_sec = start_idx / self.SAMPLE_RATE
            window_slice = slice(start_idx, end_idx)

            window_labels = labels[window_slice]
            most_common_label_id = np.argmax(np.bincount(window_labels))
            
            if most_common_label_id not in [1, 2, 3]: 
                continue

            condition_obj = condition_map.get(most_common_label_id)

            for device_location in ['CHEST', 'WRIST']:
                device_signal_data = signal_data[device_location.lower()] 

                calculated_features = self._calculate_device_features(
                    device_signal_data, 
                    device_location, 
                    window_slice
                )

                features_to_create.append(
                    PhysiologicalFeature(
                        subject=subject_obj,
                        condition=condition_obj,
                        device_location=device_location,
                        time_window_start_sec=time_start_sec,
                        window_length_sec=self.WINDOW_SECONDS,
                        
                        hrv_rmssd=calculated_features['hrv_rmssd'],
                        eda_mean=calculated_features['eda_mean'],
                        acc_std_x=calculated_features['acc_std_x'],
                        bvp_mean=calculated_features['bvp_mean'],
                    )
                )
        return features_to_create

    def process(self):
        try:
            data = self._load_data()
        except Exception as e:
            return False, f"File loading error: {e}"
        
        try:
            subject_obj, condition_map = self._get_db_objects()
        except Exception as e:
            return False, f"Database initialization error: {e}"

        try:
            features_to_create = self._extract_features(data, subject_obj, condition_map)
        except Exception as e:
            return False, f"Feature calculation error: {e}"

        try:
            with transaction.atomic():
                PhysiologicalFeature.objects.bulk_create(features_to_create, ignore_conflicts=True)
        except Exception as e:
            return False, f"Database bulk insertion error: {e}"

        return True, f"Successfully processed {len(features_to_create)} feature windows for {self.subject_code} (Chest & Wrist)."