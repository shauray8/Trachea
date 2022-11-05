import torch
import torchaudio
import matplotlib.pyplot as plt
import os

DATASET = "E:\data\LJSpeech-1.1"
CHARSET = ' abcdefghijklmnopqrstuvwxyz,.'
XMAX = 870    # about 10 seconds
YMAX = 250
SAMPLE_RATE = 

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
    if wav_file in buffer:
        return buffer[wav_file]
    waveform, sample_rate = torchaudio.load(os.path.join(DATASET,"wavs" ,wav_file))
    transform = torchaudio.transforms.MelSpectrogram(sample_rate,n_fft=1024, win_length=1024, hop_length=256, n_mels=80)
    spectrogram = transform(waveform)
    buffer[wav_file] = spectrogram[0].T
    return spectrogram[0].T

load_lj("LJ001-0001.wav")
print(buffer)



