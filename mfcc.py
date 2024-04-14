import librosa
import librosa.display 
import numpy as np
import matplotlib.pyplot as plt

audiofile = "wav_trial_split/1727.wav" 

signal, sr = librosa.load(audiofile)

S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128,
                                   fmax=8000)

mfccs = librosa.feature.mfcc(y = signal, n_mfcc = 13, sr = sr, S=librosa.power_to_db(S))

print(mfccs.shape)

plt.figure(figsize=(25,10))
librosa.display.specshow(mfccs, x_axis = "time", sr = sr)
plt.colorbar(format = "%+2f")
plt.show()
