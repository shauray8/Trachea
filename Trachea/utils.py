import numpy as np
import pandas as pd
import os
import csv
import torchaudio
import torch
import matplotlib.pyplot as plt
import time

from torch.utils.data import Dataset

##----------------------- Gloabal Variables -----------------------##
DATASET = "E:\data\LJSpeech-1.1"
CHARSET = " abcdefghijklmnopqrstuvwxyz,.'"
XMAX = 1600
YMAX = 250
SAMPLE_RATE = 22050

class ListDataset(Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        
        inputs, yaw, pitch = self.path_list[index]
        yaw, pitch = onehot_vector(yaw, self.yaw_classes), onehot_vector(pitch, self.pitch_classes)
        inputs, yaw, pitch = loader(inputs, yaw, pitch)

        if self.transform is not None:
            inputs[0] = self.transform(inputs[0])
            inputs[1] = self.transform(inputs[1])

        return inputs, np.float32(yaw), np.float32(pitch)

    def __len__(self):
        return len(self.path_list)

def DATA_LOADER(root, split):        
    img_data = []
    mode = 5
    yaw_array = []
    pitch_array = []
    for i in range(mode):
        #input_img = (glob.glob(os.path.join(data_root, "wavs","*.wav"))
        #target_num = get_metadata()
        drive_img = []

        w = open(target_num[0],'r')
        drive_img.append(w.read())
        drive_img = drive_img[0].split("\n")
    
        for i in range(len(input_img)-1):
            yaw, pitch = drive_img[i].split(" ")
            yaw = 0 if yaw == "nan" else yaw
            pitch = 0 if pitch == "nan" else pitch
            img_data.append([[ input_img[i], input_img[i+1] ], float(yaw), float(pitch) ])
            yaw_array.append(float(yaw))
            pitch_array.append(float(pitch))
            
## ---------------- initializing class ararys with value 0. ---------------- ##

    yaw_array = np.sort(yaw_array)
    pitch_array = np.sort(pitch_array)
    yaw_classes, pitch_classes = [0.], [0.]

## ---------------- breaking into classes of 100 each ---------------- ##

    for yaw in range(len(yaw_array)):
        if yaw % 100 == 0 and yaw_array[yaw] > 0:
            yaw_classes.append(yaw_array[yaw])
    yaw_classes.append(1.)

    for pitch in range(len(pitch_array)):
        if pitch % 100 == 0 and pitch_array[pitch] > 0:
            pitch_classes.append(pitch_array[pitch])
    pitch_classes.append(1.)

## ---------------- train, validation split ---------------- ##

    train, test = [], []
    if split is not None:
        for sample in range( int(split*len(img_data)) ):
            train.append(img_data[sample])

        for sample in range( int(split*len(img_data)), len(img_data) ):
            test.append(img_data[sample])

    return train, test, yaw_classes, pitch_classes

## use transformed data to transform it and then return it to the data loader no fancy stuff 

def Transformed_data(data_root, transform=None, split=None):
    train, test, speech = DATA_LOADER(data_root, split)
    train_dataset = ListDataset(data_root, train, speech, transform)
    test_dataset = ListDataset(data_root, test, speech, transform)

    return train_dataset, test_dataset


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
    
def get_metadata():
    meta = []
    with open(os.path.join(DATASET, "metadata_cp.txt"), encoding="utf8") as txtfile:
        corpus = txtfile.read().split("\n")
        for row in corpus:
            row = row.split("|")
            tokens = [CHARSET.index(c)+1 for c in row[1].lower() if c in CHARSET]
            if len(tokens) <= YMAX:
                meta.append((os.path.join(DATASET, 'wavs', row[0]+".wav"), tokens))
    print("got metadata", len(meta))
    return meta

buffer = {}
def load_lj(wav_file):
    waveform, sample_rate = torchaudio.load(wav_file, normalize=True)
    if sample_rate not in buffer:
        hop_length = int(sample_rate/(1000/10))
        win_length = int(sample_rate/(1000/25))
        buffer[sample_rate] = torchaudio.transforms.MelSpectrogram(sample_rate, n_fft=win_length, win_length=win_length, hop_length=hop_length, n_mels=80)
    mel_specgram = buffer[sample_rate](waveform)
    return mel_specgram[0].T


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
    plot_waveform(waveform, sample_rate)
    #wave = return_spec(wavefile)
    #print(wave)


#    a = wav_to_vec(wavefile)


    



    ## convert the audio signal to spectrogram or MFCC whatever feels good and then feed it through a deep neural network and train it against tokeinized sentences 
