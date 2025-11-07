import pickle
import os
from time import sleep
from celery import shared_task

@shared_task
def process_pickle_data(data_dict):
    """Simulated processing of pickle data — outputs a .pkl file."""
    sleep(5)  # Simulate heavy computation

    # Example of a transformation
    processed_data = {
        "processed_keys": list(data_dict.keys()),
        "length": len(data_dict)
    }

    # Ensure output directory exists
    output_dir = "output_pickles"
    os.makedirs(output_dir, exist_ok=True)

    return {"output_file": 123}
