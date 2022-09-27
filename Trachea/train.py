import torch
import numpy as np
import os
import matplotlib.pyplot as plt

## validation before training supremacy

print(f"=> fetching image pairs from {args.data}") 
train_set, test_set = Transformed_data(args.data, transform=input_transform, split = args.split_value)

print(f"=> {len(test_set) + len(train_set)} samples found, {len(train_set)} train samples and {len(test_set)} test samples")

train_loader = DataLoader(
        train_set, batch_size = args.batch_size, num_workers=args.workers,
        pin_memory=True, shuffle=True)

val_loader = DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, shuffle = False)
