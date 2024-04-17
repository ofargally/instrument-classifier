import librosa
import librosa.display 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

directory = './wav_trial_split/'
save_path = './mfccs/training'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):

        audiofile = f

        signal, sr = librosa.load(audiofile) 

        length = len(signal)/ sr #length of the audio file used to get the intervals 
        
        #THESE LIKELY MAY NEED TO BE TUNED LATER!!!
        hop_length = int(sr * 0.0116) #hop length 11.6 ms
        step = hop_length / sr 
        n_fft = int(sr * .0464) #block size 46.4 ms 
        n_mels = 96 #number of mel bands
        f_max = sr / 2 #frequency max 
        f_min = 20 #frequency min 

        #it seems like this outputs n_mfccs coefficient rows and each column is the time sections the mfcc chunks it into. 
        mfccs = librosa.feature.mfcc(y = signal, n_mfcc = 13, sr = sr, hop_length = hop_length, n_mels = n_mels, fmin= f_min, fmax = f_max)

        intervals_s = np.arange(start=0, stop=length, step=step) #this is the time stamps of each interval made by mfcc in seconds 
        print(mfccs.shape)

        df = pd.DataFrame(mfccs[1])
        file_name = os.path.splitext(filename)[0]
        filepath = os.path.join(save_path, f'{file_name}_mfccs.csv')

        #print(mfccs)
        df.to_csv(filepath, index=False)
