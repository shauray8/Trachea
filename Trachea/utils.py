import numpy as np
import pandas as pd

if __name__ == "__main__":
    readme = pd.read_csv("E:/data/LJSpeech-1.1/metadata.csv", header=None)
    print(readme.head()[1])
