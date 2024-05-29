import numpy as np  

phoneme_ids_61 = np.load('phoneme_id.npy', allow_pickle=True).item()
print(phoneme_ids_61)

phoneme_map_61_to_39 = {
    'aa': 'aa', 'ae': 'ae', 'ah': 'ah', 'ao': 'aa', 'aw': 'aw', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er', 'ay': 'ay',
    'b': 'b', 'bcl': 'sil', 'ch': 'ch', 'd': 'd', 'dcl': 'sil', 'dh': 'dh', 'dx': 'dx', 'eh': 'eh', 'el': 'l',
    'em': 'm', 'en': 'n', 'eng': 'ng', 'epi': 'sil', 'er': 'er', 'ey': 'ey', 'f': 'f', 'g': 'g', 'gcl': 'sil',
    'h#': 'sil', 'hh': 'hh', 'hv': 'hh', 'ih': 'ih', 'ix': 'ih', 'iy': 'iy', 'jh': 'jh', 'k': 'k', 'kcl': 'sil',
    'l': 'l', 'm': 'm', 'n': 'n', 'ng': 'ng', 'nx': 'n', 'ow': 'ow', 'oy': 'oy', 'p': 'p', 'pau': 'sil', 'pcl': 'sil',
    'q': 'sil', 'r': 'r', 's': 's', 'sh': 'sh', 't': 't', 'tcl': 'sil', 'th': 'th', 'uh': 'uh', 'uw': 'uw', 'ux': 'uw',
    'v': 'v', 'w': 'w', 'y': 'y', 'z': 'z', 'zh': 'sh'
}

phoneme_id_39 = {
    'aa': 0, 'ae': 1, 'ah': 2, 'aw': 3, 'ay': 4, 'b': 5, 'ch': 6, 'd': 7, 'dh': 8, 'dx': 9, 'eh': 10, 'er': 11, 'ey': 12,
    'f': 13, 'g': 14, 'hh': 15, 'ih': 16, 'iy': 17, 'jh': 18, 'k': 19, 'l': 20, 'm': 21, 'n': 22, 'ng': 23, 'ow': 24,
    'oy': 25, 'p': 26, 'r': 27, 's': 28, 'sh': 29, 't': 30, 'th': 31, 'uh': 32, 'uw': 33, 'v': 34, 'w': 35, 'y': 36,
    'z': 37, 'sil': 38
}

id_map_61_to_39 = {phoneme_ids_61[k]: phoneme_id_39[v] for k, v in phoneme_map_61_to_39.items()}


# print(id_map_61_to_39)

npz_dataset_train = np.load('./datasets_processed/dataset_train.npz', allow_pickle=True)
npz_dataset_val = np.load('./datasets_processed/dataset_val.npz', allow_pickle=True)
npz_dataset_test = np.load('./datasets_processed/dataset_test.npz', allow_pickle=True)

npz_model_a_40 = np.load('./datasets_processed/dataset_train_model_a_40.npz', allow_pickle=True)
npz_model_a_60 = np.load('./datasets_processed/dataset_train_model_a_60.npz', allow_pickle=True)

model_a_40_labels = npz_model_a_40['labels']
model_a_40_frames = npz_model_a_40['frames']

model_a_60_labels = npz_model_a_60['labels']
model_a_60_frames = npz_model_a_60['frames']

test_labels = npz_dataset_test['labels']
test_frames = npz_dataset_test['frames']
# print("test")
# print(f"Length of frames: {len(test_frames)}")
# print(f"Length of labels: {len(test_labels)}")

val_labels = npz_dataset_val['labels']
val_frames = npz_dataset_val['frames']
# print("val")
# print(f"Length of frames: {len(val_frames)}")
# print(f"Length of labels: {len(val_labels)}")

train_frames = npz_dataset_train['frames']
train_labels = npz_dataset_train['labels']
# print("train")
# print(f"Length of frames: {len(train_frames)}")
# print(f"Length of labels: {len(train_labels)}")

# aa = 10
mapped_labels_test = np.array([id_map_61_to_39[label] for label in test_labels])
np.savez('./datasets_processed_lp/dataset_test.npz', frames=test_frames, labels=mapped_labels_test)

mapped_labels_val = np.array([id_map_61_to_39[label] for label in val_labels])
np.savez('./datasets_processed_lp/dataset_val.npz', frames=val_frames, labels=mapped_labels_val)

mapped_labels_train = np.array([id_map_61_to_39[label] for label in train_labels])
np.savez('./datasets_processed_lp/dataset_train.npz', frames=train_frames, labels=mapped_labels_train)

mapped_labels_model_a_40 = np.array([id_map_61_to_39[label] for label in model_a_40_labels])
np.savez('./datasets_processed_lp/dataset_train_model_a_40.npz', frames=model_a_40_frames, labels=mapped_labels_model_a_40)

mapped_labels_model_a_60 = np.array([id_map_61_to_39[label] for label in model_a_60_labels])
np.savez('./datasets_processed_lp/dataset_train_model_a_60.npz', frames=model_a_60_frames, labels=mapped_labels_model_a_60)

print("finished")


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

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    
bucket_name = 'projectdt2119'

train_data_path = './datasets_processed_lp/dataset_train.npz'
val_data_path = './datasets_processed_lp/dataset_val.npz'
test_data_path = './datasets_processed_lp/dataset_test.npz'

upload_to_gcs(bucket_name, train_data_path, 'datasets_processed_lp/dataset_train.npz')
upload_to_gcs(bucket_name, val_data_path, 'datasets_processed_lp/dataset_val.npz')
upload_to_gcs(bucket_name, test_data_path, 'datasets_processed_lp/dataset_test.npz')

train_data_path = './datasets_processed_lp/dataset_train_model_a_40.npz'
val_data_path = './datasets_processed_lp/dataset_train_model_a_60.npz'

upload_to_gcs(bucket_name, train_data_path, 'datasets_processed_lp/dataset_train_model_a_40.npz')
upload_to_gcs(bucket_name, val_data_path, 'datasets_processed_lp/dataset_train_model_a_60.npz')
