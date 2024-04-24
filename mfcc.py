import librosa
import librosa.display 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import json

from label_preprocesser import time_intervals_to_csv


directory = './wav_trial_split/'
save_path = './mfccs/training'

def instruments_to_mfcc(data_filename : str, mfcc_filename : str) :
    """
    Takes the name of a train_labels or test_labels csv and a corresponding MFCC csv, and
    assigns instrument groups to each entry of the MFCC csv
    """
    data_df = pd.read_csv(data_filename)
    mfcc_df = pd.read_csv(mfcc_filename)
    mfcc_df = mfcc_df.rename(columns={mfcc_df.columns[0]: "Coefficients"})
    mfcc_df['Instruments'] = None
    # Iterate over each row in data_df
    for index, row in data_df.iterrows():
        instrument_group = row['Instrument Group']
        time_chunks_string = row['Time Chunks']
        
        # Convert the string representation of list to a Python list
        time_chunks_list = json.loads(time_chunks_string)
        
        # Iterate over each time chunk in the list
        for time_chunk in time_chunks_list:
            # Convert time chunk to integer
            time_chunk = int(time_chunk)
            
            # Add instrument group to corresponding row in 'Instruments' column of mfcc_df
            mfcc_df.loc[time_chunk, 'Instruments'] = instrument_group
    print(mfcc_df.head(10))
    mfcc_df.to_csv(mfcc_filename, index=False)

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
        data_filename = 'labels/train_labels/' + file_name + '.csv' # THIS IS HARDCODED! FIX IT LATER!
        mfcc_filename = 'mfccs/training/' + file_name + '_mfccs.csv' # ALSO HARDCODED
        print(data_filename)
        print(mfcc_filename)

        time_intervals_to_csv(data_filename, hop_length * 2)
        instruments_to_mfcc(data_filename, mfcc_filename)

        #print(mfccs)
        df.to_csv(filepath, index=False)