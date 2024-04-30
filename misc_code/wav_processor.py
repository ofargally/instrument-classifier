import soundfile as sf
import numpy as np
import os

def split_wav_file(filename, chunk_length=2, output_dir='chunks'):
    """
    Split a .wav file into chunks of `chunk_length` seconds each using PySoundFile.

    :param filename: Path to the .wav file
    :param chunk_length: Length of each chunk in seconds
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data, samplerate = sf.read(filename)
    chunk_size = samplerate * chunk_length
    total_chunks = np.ceil(len(data) / chunk_size).astype(int)

    for i in range(total_chunks):
        start = int(i * chunk_size)
        end = int(start + chunk_size)
        chunk_data = data[start:end]

        chunk_filename = os.path.join(output_dir, f"{os.path.basename(filename)}_chunk_{i}.wav")
        sf.write(chunk_filename, chunk_data, samplerate)
        print(f"Created chunk: {chunk_filename}")

# Example usage
split_wav_file('./wav_trial_split/1727.wav', chunk_length=2, output_dir='./1727_chunks')