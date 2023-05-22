import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import librosa
import librosa.display
import pdb

# file_list = os.listdir("/data/audios")
data_dir = "/home/tianchong/Downloads"

file_list = os.listdir(data_dir + "/audios")
file_list.sort()
pdb.set_trace()
res = {}

for i in range(10):

    # Load the .wav file
    filename = data_dir + "/audios/" + file_list[i]  # replace with your .wav file
    y, sr = librosa.load(filename)

    # Compute a Mel-scaled spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length = 80)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    assert log_mel_spectrogram.shape[1] >= 2584

    log_mel_spectrogram = log_mel_spectrogram[:, :2584]
    log_mel_spectrogram = log_mel_spectrogram.T

    audio_name = file_list[i].split('.')[0]
    res[audio_name] = log_mel_spectrogram

# save to pickle
with open(data_dir + '/spec.pickle', 'wb') as f:
    pickle.dump(res, f)

# # save to pickle
# with open('/data/spec.pickle', 'w') as f:
#     pickle.dump(res, f)




