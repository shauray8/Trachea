## testing trained model
import numpy as np 
import torch
import torchaudio, pyaudio
from model import Rec
from utils import load_lj, CHARSET
import itertools
import matplotlib.pyplot as plt

model = Rec()
vals = torch.load('pretrained/tinyvoice_1652627131_165_0.11.pt', map_location=torch.device('cpu'))
model.load_state_dict({k[7:]:v for k,v in vals.items()})

def to_text(x):
    x = [k for k, g in itertools.groupby(x)]
    return ''.join([CHARSET[c-1] for c in x if c != 0])

model.eval()

data = "E:\data\LJSpeech-1.1\wavs\LJ011-0011.wav"
val = load_lj(data)
plt.imshow(torch.concat([torch.log10(val).T, torch.log10(val).T], axis=1))
print(val[None].shape)
mguess = model(val[None], torch.tensor([val.shape[0]]))[0]
print(mguess.shape)
pp = to_text(mguess[:, 0, :].argmax(dim=1).cpu().numpy())
print(pp)

