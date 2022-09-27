import numpy as np
import pandas as pd
import os
import csv
import torchaudio
import torch
import matplotlib.pyplot as plt
import time

#class dataset(object):
#    def __init__(self, config, vocab, text_col, audio_path):
#        self.config = config
#        self.vocab = vocab
#        self.text_col = text_col
#        self.audio_path = audio_path
#
#    def prepare_data(self, batch: set) -> batch:
#        pass
#

class wav_to_vec:
    def __init__(self, wavefile):
        self.wavefile = wavefile
        self.waveform,self.smaple_rate = torchaudio.load(self.wavefile, normalize=True)

    def __repr__(self) -> torch.tensor:
        spec_data = self.convert_to_spectrogram()
        return spec_data

    def __str__(self):
        return

    def convert_to_spectrogram(self) -> torch.tensor:
        transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
        mel_specgram = transform(self.waveform[None].type(torch.float32))
        mel_specgram = mel_specgram.reshape(1,80,-1)
        return mel_specgram

    def shape(self):
        return type(self)
        

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

def show():
    waveform, sample_rate = torchaudio.load(wavefile)
    print(len(waveform[0]))
    plt.plot(waveform[0:100])
    plt.show()

## outputs text from a different file
def load(fn):
    lis = open(fn,"r",encoding="utf-8").read().strip().split("\n")
    wav_file, trans = lis[0].strip().replace("\t","").split("|")[:-1]
    print(wav_file, trans)
    

def librosa_pre(wavefile):
    tt, sr= librosa.load(wavefile)
    #librosa.display.waveshow(tt, sr=sr)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(tt)), ref=np.max)
    #M = librosa.feature.melspectrogram(y=tt, sr=sr, hop_length=512)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time',
                               sr=sr)
    plt.show()
    print(tt)
    pass

def return_MFCC(wavefile):
    y, sr = librosa.load(wavefile)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    img = librosa.display.specshow(mfccs, x_axis='time')
    plt.show()
    pass

## return mel spectrograms and feed them to the network which outputs the text (not exactly)
## using pytorch instead
def return_spec(wavefile):

    fig, ax = plt.subplots()
    tt, sr= librosa.load(wavefile)
    spectrogram = librosa.feature.melspectrogram(tt)
    img = librosa.display.specshow(spectrogram)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()
    return


#def convert_to_spectrogram(waveform:torch.tensor,sample_rate:torch.tensor):
#
#    transform = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
#    mel_specgram = transform(waveform[None].type(torch.float32))
#    mel_specgram = mel_specgram.reshape(1,80,-1)
#    return mel_specgram
#    #print("log",torch.log10(mel_specgram).reshape(80,-1).shape)
#    #plt.imshow(torch.log10(mel_specgram))
#    #plt.show()
    


if __name__ == "__main__":
    #readme = pd.read_csv("E:/data/LJSpeech-1.1/metadata.csv", header=None)
    #print(readme.head()[1])

    fn =  "E:/data/LJSpeech-1.1/metadata_cp.txt"
    wavefile = "E:/data/LJSpeech-1.1/wavs/LJ001-0002.wav"
#    load(fn)


    #with open(fn, newline='') as csvfile:
    #    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #    for read in spamreader:
    #        print(read)
    #        break

    metadata = torchaudio.info(wavefile)
    waveform, sample_rate = torchaudio.load(wavefile, normalize=True)
    #print(metadata)
    #librosa_pre(wavefile)
    #plot_waveform(waveform, sample_rate)
    #wave = return_spec(wavefile)


    a = wav_to_vec(wavefile)
    print(a)


    



    ## convert the audio signal to spectrogram or MFCC whatever feels good and then feed it through a deep neural network and train it against tokeinized sentences 









