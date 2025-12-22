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

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_index_path, n_fft, amplitude_only):
        lines = []
        with open(dataset_index_path, 'r') as r:
            lines = r.readlines()
            
        def line_to_fragment(line):
            [track, start, end] = line.split(';')
            return track, int(start), int(end)
        
        self.fragments = list(map(line_to_fragment, lines))
        self.n_fft = n_fft
        self.window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        self.amplitude_only = amplitude_only
        
    
    def __getitem__(self, index):
        track, start, end = self.fragments[index]
        
        def get(sub_path):
            samples, sample_rate = torchaudio.load(
                os.path.join(track, sub_path),
                frame_offset = start, 
                num_frames = end - start, 
                normalize = True, 
                channels_first = True
            )
            assert sample_rate == 44100
        
            return torch.stft(samples, n_fft=self.n_fft, return_complex=True, window=self.window)
            
            
            if self.amplitude_only:
                return spectrum_abs 
            else:
                return spectrum
                
        full = get('./mixture.wav')
        vocal = get('./vocals.wav')
        
        full_abs  = full.abs()
        vocal_abs = vocal.abs()

        full_max = full_abs.max()
        
        if self.amplitude_only:
            return full_abs / full_max, vocal_abs / full_max
        else:
            return full / full_max, vocal / full_max
        
    def __len__(self):
        return len(self.fragments)
    
        


train_path = './train_dataset.csv'
test_path = './test_dataset.csv'




    
match sys.argv:
    case [_, 'prepare']:
        duration = 4
        overlap  = 2
        prepare_musdb_index('./musdb/train/', train_path, duration, overlap)
        prepare_musdb_index('./musdb/test/',  test_path,  duration, overlap)      
    case [_, 'load_test']:
        n_fft = 1024
        
        waveform = torch.zeros((2, 44100 * 400), dtype=torch.float32)
        weights = torch.zeros((1, 44100 * 400),  dtype=torch.float32)
        
        
        musdb = MUSDB18Dataset(train_path, n_fft, False)
        
        current_track = None
        for i, (x, y) in enumerate(musdb):
            x *= 500
            y *= 500
            
            track, start, end = musdb.fragments[i]
            fragment = torch.istft(y, n_fft=musdb.n_fft, window=musdb.window, center=True)
      
            print(track, start, end)
      
            if current_track == None: current_track = track
            if current_track != track:
                current_track = track
                break
            
            waveform[:, start:start+fragment.shape[1]] += fragment[:]
            weights[:, start:start+fragment.shape[1]] += 1
                         
        
        waveform /= weights
        
        torchaudio.save('load_test.wav', waveform, 44100)
    case _:
        print(f'unknown command')
        print(f'usage:')
        print(f'  prepare   | Prepares the train and test dataset indexes from ./musdb/train and ./musdb/test directories')
        print(f'  load_test | Reconstructs first track as specified in train dataset index to load_test.wav' )
        sys.exit(1)
        