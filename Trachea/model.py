## Transformers 
"""
consists of an encoder and a decoder 
(x1,x2,x3,x4,....,xn) --> encoder --> (z1,z2,z3,z4,.....zn)
(z1,z2,z3,z4,.....,zn) --> decoder --> (y1,y2,y3,y4,.....,ym)

encoder -->
    composed of a stack of N = 6 identical layers   
    each layer has 2 sub layers 
        1. multi headed self attention mech
        2. simple fully connected
    uses a residual connnection around every 2 sublayers
    followed by layer norm
    output of every sublayer is LayerNorm(x+Sublayer(x)) --> d(model) = 512

decoder --> 
    also composed of N identical layers 
    with 3 sub layers additional sub layer performs multi head attention over the output of the sub layers
    and Layer Norm

Attention --> 
    mapping a query and a set of key-value pairs to an output (every thing is a ventor)

    takes in query,key,values

Dot product attention -->
    compute dot product of the query with all the keys and divide each by the root of |dimention of keys|
    every thing is packed into a matrix Q

    Attention(Q,K,V) = softmax(Q*K^T / root|dim k|)*V


"""

import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchaudio.models import Conformer

CHARSET = " abcdefghijklmnopqrstuvwxyz,.'"
class ResBlock(nn.Module):
  def __init__(self, c):
    super().__init__()
    self.block = nn.Sequential(
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c),
      nn.ReLU(c),
      nn.Conv2d(c, c, 3, padding='same'),
      nn.BatchNorm2d(c))
  
  def forward(self, x):
    return nn.functional.relu(x + self.block(x))

class TemporalBatchNorm(nn.Module):
  def __init__(self, channels):
    super().__init__()
    self.bn = nn.BatchNorm1d(channels)
  def forward(self, x):
    return self.bn(x.permute(0,2,1)).permute(0,2,1)

class Rec(nn.Module):
  def __init__(self, expand=1):
    super().__init__()
    self.expand = expand

    """
    C, H = 16, 256
    self.encode = nn.Sequential(
      nn.Conv2d(1, C, 1, stride=2),
      nn.ReLU(),
      ResBlock(C),
      ResBlock(C),
      nn.Conv2d(C, C, 1, stride=2),
      nn.ReLU(),
      ResBlock(C),
      ResBlock(C),
    )
    self.flatten = nn.Linear(320, H)
    self.gru = nn.GRU(H, H, batch_first=True)
    """

    C = 64
    self.encode = nn.Sequential(
      nn.Conv2d(1, C, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(C, C, kernel_size=3, stride=2),
      nn.ReLU(),
    )

    #H = 512
    #H = 80
    #H = 256
    H = 144
    self.linear = nn.Sequential(
      nn.Linear(C*(((80 - 1) // 2 - 1) // 2), H),
      nn.Dropout(0.1),
    )

    #encoder_layer = nn.TransformerEncoderLayer(d_model=H, nhead=4, dim_feedforward=H*4)
    #self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=12)
    #self.gru = nn.GRU(H, H, batch_first=True)
    self.conformer = Conformer(input_dim=H, num_heads=4, ffn_dim=H*4, num_layers=16, depthwise_conv_kernel_size=31)

    self.decode = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(H, len(CHARSET)*self.expand)
    )

  def forward(self, x, y):
    # (time, batch, freq)
    #print(x.shape)
    """
    x = x[:, None] # (batch, time, freq) -> (batch, 1, time, freq)
    # (batch, C, H, W)
    x = self.encode(x).permute(0, 2, 1, 3) # (batch, H(time), C, W)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = self.flatten(x)
    x = self.gru(x)[0]
    """
    x = self.encode(x[:, None])
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(x.shape[0], x.shape[1], -1)
    y = (y>>2)-1
    x = x[:, :torch.max(y)] # might clip last conv feature
    x = self.linear(x)
    #x,zz = self.transformer(x), y
    #x,zz = self.gru(x)[0], y
    x,zz = self.conformer(x, y)
    x = self.decode(x).reshape(x.shape[0], x.shape[1]*self.expand, len(CHARSET))
    zz *= 4
    return torch.nn.functional.log_softmax(x, dim=2).permute(1,0,2), zz


if __name__ == "__main__":
    model = Rec()
    print(model)
