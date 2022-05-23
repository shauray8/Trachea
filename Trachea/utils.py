import numpy as np
import pandas as pd
import os
import csv
import wave
import torchaudio
import torch
import matplotlib.pyplot as plt
import time

def wavw():
    wavefile = wave.open("E:/data/LJSpeech-1.1/wavs/LJ001-0001.wav","r")
    length = wavefile.getnframes()
    print(f"{length} frames")
    wavedata = np.frombuffer(wavefile.readframes(length), np.int16)
    print(len(wavedata))
    print(wavedata[0:10])

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    time.sleep(10)

if __name__ == "__main__":
    #readme = pd.read_csv("E:/data/LJSpeech-1.1/metadata.csv", header=None)
    #print(readme.head()[1])

    fn =  "E:/data/LJSpeech-1.1/metadata.csv"
    wavefile = "E:/data/LJSpeech-1.1/wavs/LJ001-0001.wav"
    #lis = open(fn,"r").read().strip().split("\n")
    #print(lis)

    #with open(fn, newline='') as csvfile:
    #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #    for read in spamreader:
    #        print(read)
    #        break

    metadata = torchaudio.info(wavefile)
    print(metadata)

    waveform, sample_rate = torchaudio.load(wavefile)
    print(len(waveform[0]))
    plt.plot(waveform[0:100])
    plt.show()





