import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd

def x():
    x = "E:\data\LJSpeech-1.1\wavs\LJ001-0001.wav"
    wave, sam = torchaudio.load(x)
    trans = torchaudio.transforms.MelSpectrogram(sam)
    sepc = trans(wave)
    plt.imshow((sepc[0].T))
    plt.show()
    print(wave[0][:1], sam)

DATASET = "E:\data\LJSpeech-1.1"
CHARSET = ' abcdefghijklmnopqrstuvwxyz,.'
XMAX = 870    # about 10 seconds
YMAX = 250

def file():
    ret = []
    with open(os.path.join(DATASET, "metadata_cp.txt"), encoding="utf8") as txt:
        row = txt.read().split("\n")
        for more in row:
            get = more.split("|")
            answer = [CHARSET.index(c)+1 for c in get[1].lower() if c in CHARSET]
            if len(answer) <= YMAX:
                ret.append((os.path.join(DATASET, 'wavs', get[0]+".wav"), answer))
    print("got metadata", len(ret))
    return ret

ret = file()

