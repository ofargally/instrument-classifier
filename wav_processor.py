import wave
import audioop

def split_wav_file(filename, chunk_length=2):
    """
    Split a .wav file into chunks of `chunk_length` seconds each.

    :param filename: Path to the .wav file
    :param chunk_length: Length of each chunk in seconds
    """
    with wave.open(filename, 'rb') as original_wav:
        # Read parameters from the original file
        n_channels, sampwidth, framerate, n_frames, comptype, compname = original_wav.getparams()
        # Calculate the number of frames per chunk
        frames_per_chunk = int(chunk_length * framerate)
        for i in range(0, n_frames, frames_per_chunk):
            chunk_frames = original_wav.readframes(frames_per_chunk)
            chunk_filename = f"{filename}_chunk_{i//frames_per_chunk}.wav"
            with wave.open(chunk_filename, 'wb') as chunk_wav:
                chunk_wav.setparams((n_channels, sampwidth, framerate, len(chunk_frames), comptype, compname))
                chunk_wav.writeframes(chunk_frames)
            print(f"Created chunk: {chunk_filename}")
            
split_wav_file('path/to/your/file.wav')