from scipy.io import wavfile
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import os
import pandas as pd
import soundfile as sf
import sys

fs = 100000

def run_and_save(audio_path):
    spikes = get_spikes(audio_path)
    spike_path = os.path.join('spikes', audio_path) + '.npy'
    print(spike_path)
    np.save(spike_path, spikes, allow_pickle=False)

def single_channel(audio):
    return audio[[0],:]

def process_audio_and_save(audio_path):
    import torchaudio, torch
    waveform, sample_rate = torchaudio.load(audio_path)
    one_channel = single_channel(waveform) # some wav files are dual channel audio
    resampled_waveform = torchaudio.functional.resample(one_channel, orig_freq=sample_rate, new_freq=fs)
    smaller_data = 0.05*(resampled_waveform/resampled_waveform.max())
    numpy_data = smaller_data.numpy()[0]
    
    total_path = os.path.join('spikes', audio_path)
    print('saving to ' + total_path)
    wavfile.write(total_path, fs, numpy_data)


def run_on_all_data(root, fn=run_and_save):
    for folder in os.listdir(root):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path): continue

        for datapoint in os.listdir(folder_path):
            data_path = os.path.join(folder_path, datapoint)
            if not os.path.isfile(data_path): continue
            fn(data_path)

def process_spike_train(vals, output_len):
    # to reduce it to 4000 hz, go every 0.00025
    output = np.zeros((output_len))
    ind = 0
    for i in range(output_len):
        cap = 0.00025 * (i+1)
        while ind < len(vals) and vals[ind] < cap:
            output[i] += 1
            ind += 1
        output[i] /= 0.00025
    return output

def get_spikes(audio_path):
    import cochlea
    # loads the audio, runs our model on it, saves it to spikes/audio_path
    samplerate, data = wavfile.read(os.path.join('spikes', audio_path))
    assert(samplerate == fs)
    
    print('processing...')
    anf_trains = cochlea.run_zilany2014(
        np.array(data, dtype=np.float64),
        fs,
        anf_num=(1, 0, 0),
        cf=[10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000, 2000, 1000],
        seed=0,
        species='human'
    )

    output_len = len(data)/25 # *4k / 100k
    output = np.zeros((10, output_len))
    for i in range(10):
        output[i] = process_spike_train(anf_trains['spikes'].iloc[i], output_len)

    return output

if __name__ == '__main__':
    print("sup g")

    if sys.version_info[0] < 3:
        # we are running the model
        run_on_all_data('dataset', fn=run_and_save)
    else:
        # we are resampling all the data
        run_on_all_data('dataset', fn=process_audio_and_save)
