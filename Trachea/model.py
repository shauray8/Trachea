import torch
import wave
import numpy as np

## using conformer maybe

wavefile = wave.open("E:/data/LJSpeech-1.1/wavs/LJ037-0171.wav", "r")
length = wavefile.getnframes()
print(f"{length} frames")
wavedata = np.frombuffer(wavefile.readframes(length), np.int16)
print(len(wavedata))
print(wavedata[0:10])
