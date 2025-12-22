import sys
import os
import time
import data
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path


def save_checkpoint(path, epoch, model, optimizer):
    checkpoint = {
        'epoch': epoch,
        'model':    model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def prepare_musdb_index(input_path, output_path, duration, overlap):
    frames = int(duration * 44100)
    hop    = int((duration - overlap) * 44100)
    
    with open(output_path, 'w') as w:
        for track in os.scandir(input_path):
            mixture_path = os.path.join(track, 'mixture.wav')
            info = torchaudio.info(mixture_path)
    
            assert info.sample_rate == 44100
         
            for t_start in range(0, info.num_frames - frames, hop):
                t_end = t_start + frames
                print(f'{track.path};{t_start};{t_end}', file=w)
                print(f'{track.path};{t_start};{t_end}')
    
match sys.argv:
    case [_, 'prepare']:
        duration = 4
        overlap  = 2
        prepare_musdb_index('./musdb/train/', './train_index.csv', duration, overlap)
        prepare_musdb_index('./musdb/test/', './test_index.csv',   duration, overlap)        
    case _:
        print('unknown command')
        print('usage:')
        print('  prepare | Prepares the train and test indexes from ./musdb/train and ./musdb/test directories')
        sys.exit(1)
        