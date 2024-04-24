import librosa
import librosa.display 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from label_preprocesser import time_intervals_to_csv


directory = './wav_trial_split/'
save_path = './mfccs/training'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if os.path.isfile(f):

        audiofile = f

        signal, sr = librosa.load(audiofile)  #signal itself and the signal rate 
        #print(sr)
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
        #print(mfccs.shape)
        intervalLength = intervals_s[1] - intervals_s[0]

        timeInterval = intervalLength * sr
        

        #print(filename)
        #print(intervals_s)
        df = pd.DataFrame(mfccs[1])
        file_name = os.path.splitext(filename)[0]
        filepath = os.path.join(save_path, f'{file_name}_mfccs.csv')

        #print(filepath)
        csv_filename = 'labels/train_labels/' + file_name + '.csv' # THIS IS HARDCODED! FIX IT LATER!
        print(csv_filename)

        time_intervals_to_csv(csv_filename, hop_length * 2)

        #print(mfccs)
        df.to_csv(filepath, index=False)

#directory = '/Users/dankim/Documents/COSC410/FinalProj/instrument-classifier/labels/train_labels'
#for filename in os.listdir(directory):
#    print(filename)
#    time_intervals_to_csv(filename, sr)

        


