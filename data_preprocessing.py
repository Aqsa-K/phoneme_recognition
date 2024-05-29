import os
import torchaudio
import pandas as pd
import torch
from torchaudio.transforms import MFCC
import glob
import librosa
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from loguru import logger

SAMPLE_RATE = 16000
FRAME_LENGTH = 0.025    # ms
FRAME_STEP = 0.01       # ms


def extract_frames(audio, frame_length=0.025, frame_step=0.010, sampling_rate=16000):
    # Extract frames from audio using librosa
    frames = librosa.util.frame(audio, frame_length=int(sampling_rate * frame_length), hop_length=int(sampling_rate * frame_step))
    return frames.T  # Transpose to have frames in the first dimension

def load_phoneme_labels(phn_path):
    phonemes = []
    with open(phn_path, 'r') as file:
        for line in file:
            start, end, phoneme = line.strip().split()
            phonemes.append((int(start), int(end), phoneme))
    return phonemes

def frame_phoneme_labels(phonemes, num_frames, wav_range):
    """Create phoneme labels based on frames"""
    frame_shift = SAMPLE_RATE * FRAME_STEP
    frame_size = SAMPLE_RATE * FRAME_LENGTH
    frame_labels = []
    for frame_idx in range(num_frames):
        frame_start = frame_idx * frame_shift + wav_range[0]
        frame_end = frame_start + frame_size
        frame_phonemes = [ph for ph in phonemes if ph[0] < frame_end and ph[1] > frame_start]
        if frame_phonemes:
            label = max(frame_phonemes, key=lambda ph: min(frame_end, ph[1]) - max(frame_start, ph[0]))[2]
        else:
            logger.error(f'frame_phonemes label not found.')
            exit(0)
        frame_labels.append(label)
    return frame_labels

def process_timit_file(wav_path, phn_path):
    """process a single timit file"""
    waveform, sample_rate = torchaudio.load(wav_path)
    assert sample_rate == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, but got {sample_rate}"

    phonemes = load_phoneme_labels(phn_path)
    # get label range based on phonemes
    wav_range = [phonemes[0][0], phonemes[-1][1]]
    audio = waveform[0][wav_range[0]:wav_range[1]]

    # frames.shape -> (num_frames, frame_len)
    frames = extract_frames(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, sampling_rate=SAMPLE_RATE)
    num_frames = frames.shape[0]

    # Generate phoneme labels on frame-based level
    frame_phonemes = frame_phoneme_labels(phonemes, num_frames, wav_range)

    speaker_ids = [os.path.basename(os.path.dirname(wav_path))] * num_frames

    return frames, frame_phonemes, speaker_ids

def process_timit_dataset(data_dir, data_type='train', split=False, val_ratio=0.1):
    print(data_dir)
    all_frames = []
    all_phonemes = []
    all_speaker_ids = []

    wav_files = sorted(glob.glob(os.path.join(data_dir, data_type, '*', '*', '*.wav')))

    for wav_path in tqdm(wav_files, desc=f'Loading "{data_type}" data'):
        phn_path = wav_path.replace('.wav', '.phn')
        if os.path.exists(phn_path):
            frames, frame_phonemes, speaker_ids = process_timit_file(wav_path, phn_path)
            all_frames.extend(frames)
            all_phonemes.extend(frame_phonemes)
            all_speaker_ids.extend(speaker_ids)
    all_frames = np.array(all_frames)
    all_speaker_ids = np.array(all_speaker_ids)

    phoneme_to_id = {ph: id for id, ph in enumerate(np.sort(np.unique(all_phonemes)))}
    id_to_phoneme = {id: ph for ph, id in phoneme_to_id.items()}

    # convert phoneme to id
    all_labels = np.array([phoneme_to_id[ph] for ph in all_phonemes])

    # return all_frames, all_labels
    if split:
        train_frames = []
        train_labels = []
        val_frames = []
        val_labels = []
        unique_speaker_ids = np.unique(all_speaker_ids)
        for id in tqdm(unique_speaker_ids, desc='Split data by speakers'):
            select_idxs = np.where(all_speaker_ids == id)[0]
            speaker_frames = all_frames[select_idxs]
            speaker_labels = all_labels[select_idxs]
            # shuffle the data
            indices = np.arange(len(speaker_frames))
            np.random.shuffle(indices)
            speaker_frames = speaker_frames[indices]
            speaker_labels = speaker_labels[indices]
            
            # split the data by the given ratio
            train_frames.extend(speaker_frames[:int(len(speaker_frames)*(1-val_ratio))])
            train_labels.extend(speaker_labels[:int(len(speaker_frames)*(1-val_ratio))])
            val_frames.extend(speaker_frames[int(len(speaker_frames)*(1-val_ratio)):])
            val_labels.extend(speaker_labels[int(len(speaker_frames)*(1-val_ratio)):])
        train_frames = np.array(train_frames)
        train_labels = np.array(train_labels)
        val_frames = np.array(val_frames)
        val_labels = np.array(val_labels)
        return train_frames, train_labels, val_frames, val_labels, phoneme_to_id
    else:
        return all_frames, all_labels, phoneme_to_id

class TimitDataset(Dataset):
    def __init__(self, timit_data=None) -> None:
        super().__init__()

        self.frames = timit_data['frames']
        self.labels = timit_data['labels']

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        frame = self.frames[index]
        label = self.labels[index]
        return frame, label

# def main():
#     timit_root = 'Users/ossamaziri/Desktop/project/timit/timit'

#     train_frames, train_labels, val_frames, val_labels = process_timit_dataset(timit_root, 'train', split=True, val_ratio=0.6)
#     #test_frames, test_labels = process_timit_dataset(timit_root, 'test', split=False)

#     # Save the preprocessed dataset as npz file
#     np.savez('Users/ossamaziri/Desktop/project/dataset_train_model_a.npz', frames=train_frames, labels=train_labels)
#     np.savez('Users/ossamaziri/Desktop/project/dataset_val_model_a.npz', frames=val_frames, labels=val_labels)
#     #np.savez('/Projects/Speech/proj/dataset_test.npz', frames=test_frames, labels=test_labels)

# if __name__ == '__main__':
#     main()

def main():
    timit_root = './timit/timit/timit'

    train_frames, train_labels, val_frames, val_labels, phoneme_to_id = process_timit_dataset(timit_root, 'train', split=True, val_ratio=0.1)
    test_frames, test_labels, phoneme_to_id2 = process_timit_dataset(timit_root, 'test', split=False)

    print(phoneme_to_id.keys() == phoneme_to_id2.keys())

    # Save the dictionary to a file
    np.save('phoneme_id.npy', phoneme_to_id)

    # Define the paths
    train_data_path = './datasets_processed/dataset_train.npz'
    val_data_path = './datasets_processed/dataset_val.npz'
    test_data_path = './datasets_processed/dataset_test.npz'

    # Ensure the directories exist
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_data_path), exist_ok=True)

    # Save the preprocessed dataset as npz file
    np.savez(train_data_path, frames=train_frames, labels=train_labels)
    np.savez(val_data_path, frames=val_frames, labels=val_labels)
    np.savez(test_data_path, frames=test_frames, labels=test_labels)

    train_frames, train_labels, val_frames, val_labels, phoneme_to_id = process_timit_dataset(timit_root, 'train', split=True, val_ratio=0.6)


    # Define the paths
    train_data_path = './datasets_processed/dataset_train_model_a_40.npz'
    val_data_path = './datasets_processed/dataset_train_model_a_60.npz'

    # Ensure the directories exist
    os.makedirs(os.path.dirname(train_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_data_path), exist_ok=True)

    # Save the preprocessed dataset as npz file
    np.savez(train_data_path, frames=train_frames, labels=train_labels)
    np.savez(val_data_path, frames=val_frames, labels=val_labels)

if __name__ == '__main__':
    main()







# need to install this : !pip install google-cloud-storage

# from google.colab import files

# This will prompt you to select the JSON key file through the file picker
# uploaded = files.upload()

# After uploading, you can check the filename of the uploaded file
# print("Uploaded file(s):", list(uploaded.keys()))

import os
import json
import io
import numpy as np
# from google.colab import files
# uploaded = files.upload()

# Assuming the filename of your key is 'service-account-file.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'dt2119-lab3-422421-d62072692aa2.json'

from google.cloud import storage

# This creates a client that uses the specified service account credentials
client = storage.Client()
# Now you can interact with Google Cloud Storage using this client

# Lists all the buckets
buckets = list(client.list_buckets())
print(buckets)


def read_from_storage(bucket_name, file_name):
    """Reads content from a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_name (str): The name of the file to read from the bucket.

    Returns:
        str: The content of the file.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_name)

    # Download the content as a byte string
    content = blob.download_as_bytes()

    # Convert to string if necessary (assuming the file is text-based)
    return content.decode('utf-8')

# Example usage:
# content = read_from_storage('experimentresults', 'new-file.txt')
# print(content)

def write_to_storage(bucket_name, file_name, data):
    """Writes data to a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        file_name (str): The name of the file to write in the bucket.
        data (str): The data to write to the file.

    Returns:
        None
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(file_name)

    # Upload the data
    blob.upload_from_string(data)


# Example usage:
# write_to_storage('experimentresults', 'new-file-1.txt', 'Hello, World!')


def create_and_write_file(file_name, text_string):
    """
    Creates a file and writes a specified text string to it.

    Args:
        file_name (str): The name of the file to create.
        text_string (str): The text string to write to the file.

    Returns:
        None
    """
    # Open the file in write mode ('w'). If the file doesn't exist, it will be created.
    # If the file exists, it will be overwritten.
    with open(file_name, 'w') as file:
        file.write(text_string)

# Example usage:
# create_and_write_file('example.txt', 'Hello, this is a sample text.')


def write_json_to_gcs(bucket_name, destination_blob_name, data):
    """
    Writes JSON data to a file in Google Cloud Storage.

    Args:
        bucket_name (str): The name of the GCS bucket.
        destination_blob_name (str): The path within the bucket to save the file.
        data (str): JSON string to write to the file.
    """
    # Create a temporary file
    temp_file = "/tmp/tempfile.json"
    with open(temp_file, "w") as file:
        file.write(data)

    # Initialize the Google Cloud Storage client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Upload the file
    blob.upload_from_filename(temp_file)

    # Optionally, remove the temporary file if not needed
    os.remove(temp_file)


def read_json_from_gcs(bucket_name, source_blob_name):
    """
    Reads a JSON file from Google Cloud Storage and parses the JSON content.

    Args:
        bucket_name (str): The name of the GCS bucket.
        source_blob_name (str): The name of the blob (file) in the GCS bucket.

    Returns:
        dict: The parsed JSON data as a dictionary.
    """
    # Initialize the Google Cloud Storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.bucket(bucket_name)

    # Get the blob
    blob = bucket.blob(source_blob_name)

    # Download the content as a string
    json_data = blob.download_as_string()

    # Parse the JSON string into a Python dictionary
    data = json.loads(json_data)

    return data


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    
bucket_name = 'projectdt2119'


train_data_path = './datasets_processed/dataset_train.npz'
val_data_path = './datasets_processed/dataset_val.npz'
test_data_path = './datasets_processed/dataset_test.npz'

upload_to_gcs(bucket_name, train_data_path, 'datasets_processed/dataset_train.npz')
upload_to_gcs(bucket_name, val_data_path, 'datasets_processed/dataset_val.npz')
upload_to_gcs(bucket_name, test_data_path, 'datasets_processed/dataset_test.npz')

train_data_path = './datasets_processed/dataset_train_model_a_40.npz'
val_data_path = './datasets_processed/dataset_train_model_a_60.npz'

upload_to_gcs(bucket_name, val_data_path, 'datasets_processed/dataset_train_model_a_60.npz')
upload_to_gcs(bucket_name, train_data_path, 'datasets_processed/dataset_train_model_a_40.npz')
