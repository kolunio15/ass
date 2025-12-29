import sys
import os
import time
import math
import torch
import torch.nn as nn
import soundfile
import torchinfo
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

def audio_info(path):
    with soundfile.SoundFile(path) as f:
        return f.samplerate, f.frames

def audio_read(path, start, end):
    samples, sample_rate = soundfile.read(
        path,
        start = start,
        stop  = end,
        dtype = np.float32,
        always_2d=True
    )
    assert sample_rate == 44100
    return torch.from_numpy(np.transpose(samples))




def prepare_musdb_index(input_path, output_path, duration, overlap):
    frames = int(duration * 44100)
    hop    = int((duration - overlap) * 44100)

    with open(output_path, 'w') as w:
        for track in os.scandir(input_path):
            path = os.path.join(track, 'mixture.wav')
            sample_rate, num_frames = audio_info(path)

            rms = audio_read(path, 0, num_frames).square().mean().sqrt()

            for t_start in range(0, num_frames - frames, hop):
                t_end = t_start + frames

                print(f'{track.path};{t_start};{t_end};{rms}', file=w)
                print(f'{track.path};{t_start};{t_end};{rms}')

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_index_path, n_fft, amplitude_only):
        lines = []
        with open(dataset_index_path, 'r') as r:
            lines = r.readlines()
            
        def line_to_fragment(line):
            [track, start, end, rms] = line.split(';')
            return track, int(start), int(end), float(rms)
        
        self.fragments = list(map(line_to_fragment, lines))
        self.n_fft = n_fft
        self.window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        self.amplitude_only = amplitude_only
        
    
    def __getitem__(self, index):
        track, start, end, rms = self.fragments[index]
        
        def get(sub_path):
            samples = audio_read(os.path.join(track, sub_path), start, end)
            samples /= max(rms, 1e-25)
            return torch.stft(samples, n_fft=self.n_fft, return_complex=True, window=self.window)

        full = get('./mixture.wav')
        vocal = get('./vocals.wav')
        
        full_abs  = full.abs()
        vocal_abs = vocal.abs()
        
        if self.amplitude_only:
            return full_abs, vocal_abs
        else:
            return full, vocal
        
    def __len__(self):
        return len(self.fragments)
    
def save_checkpoint(path, epoch, model, optimizer):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']

def load_latest_checkpoint(directory, model, optimizer):
    if not os.path.exists(directory): return -1
    checkpoints = os.listdir(directory)
    checkpoints.sort()
    checkpoint = os.path.join(directory, checkpoints[-1])
    print(f'loading "{checkpoint}"')
    
    return load_checkpoint(checkpoint, model, optimizer)

class PerceptualLoss(nn.Module):
    def __init__(self, sample_rate, frequency_bin_count):
        super(self.__class__, self).__init__()
        
        # ISO data
        freqs = np.array([
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
            630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
            10000, 12500
        ])
        spl_40 = np.array([99.7, 93.9, 88.2, 82.7, 77.8, 73.0, 68.3, 64.2, 60.4, 56.6, 53.3, 50.3, 47.6, 45.1, 43.1, 41.4, 40.0, 40.0, 41.8, 42.6, 39.2, 36.6, 35.5, 36.7, 40.1, 45.8, 51.6, 54.4, 51.3])

        log_freqs = np.log10(freqs)
        cs = CubicSpline(log_freqs, spl_40, bc_type='natural')
        
        frequencies = np.linspace(0, sample_rate / 2, frequency_bin_count)
        clamped_frequencies = np.minimum(np.maximum(frequencies, freqs[0]), freqs[-1])

        spl = torch.from_numpy(cs(np.log10(clamped_frequencies))).unsqueeze(1)
        spl_min = spl.min()
        spl_max = spl.max()
        frequency_loudness = 1 - (spl - spl_min) / (spl_max - spl_min)
        
        
        self.register_buffer('frequency_loudness', frequency_loudness, persistent=False)

    def forward(self, y_predicted, y_correct):
        loss = (torch.log1p(y_predicted) - torch.log1p(y_correct)).abs()        
        loss = (loss * self.frequency_loudness).mean()         
        return loss
        

class NaiveLSTM(nn.Module):
    def __init__(self, frequency_bin_count, hidden_layers):
        super(self.__class__, self).__init__()
        self.conv1d_channel_merge = nn.Conv1d(in_channels=2 * frequency_bin_count, out_channels=1 * frequency_bin_count, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(frequency_bin_count)
        self.lstm = nn.LSTM(input_size=frequency_bin_count, hidden_size=hidden_layers, num_layers=1, dropout=0.2, batch_first=True, bidirectional=True)
        self.last_fc = nn.Linear(hidden_layers * 2, frequency_bin_count)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        batches, channels, frequencies, timesteps = x.shape
        x = x.reshape(batches, channels * frequencies, timesteps) # [batches, channels * frequencies, timesteps]
        x = self.conv1d_channel_merge(x)                          # [batches, frequencies, timesteps] 
        x = self.relu1(x)
        x = self.bn(x)                                            
        x = x.transpose(1, 2)                                     # [batches, timesteps, frequencies]
        x, _ = self.lstm(x)                                       # [batches, timesteps, hidden_size]
        x = self.last_fc(x)                                       # [batches, timesteps, frequencies]
        x = x.transpose(1, 2)                                     # [batches, frequencies, timesteps]
        x = x.unsqueeze(1)                                        # [batches, 1, frequencies, timesteps]
        x = self.relu2(x)
        x = x.expand(-1, 2, -1, -1)                               # [batches, 2, frequencies, timesteps]
        return x
        
class NaiveConv(nn.Module):
    def __init__(self, frequency_bin_count): 
        super(self.__class__, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=2, out_channels=12, kernel_size=3, padding=1) 
        self.bn1 = nn.BatchNorm2d(12)
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))             
        
        
        self.conv2d_2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.max_pool2d_2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))                 
        
        self.conv2d_3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1)                          
        self.bn3 = nn.BatchNorm2d(40)
        self.conv2d_4 = nn.Conv2d(in_channels=40, out_channels=30, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(30)
        self.conv2d_5 = nn.Conv2d(in_channels=30, out_channels=20, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(20)

        self.conv2d_6 = nn.Conv2d(in_channels=20, out_channels=12, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(12)
        
        self.conv2d_7 = nn.Conv2d(in_channels=12, out_channels=2, kernel_size=3,  padding=1)
        
    def forward(self, x):
        batches, channels, frequencies, timesteps = x.shape
        
        
        x = x.transpose(2, 3)                                                # [batches, 2, timesteps, frequencies]
        saved_2_channel = x
        
        x = self.conv2d_1(x)                                                 # [batches, 12, timesteps, frequency_bin_count]
        x = self.bn1(x)
        saved_12_channel = x
        x = nn.functional.relu(x)
        x = self.max_pool2d_1(x)                                             # [batches, 12, timesteps/3, frequency_bin_count/3]
        
        x = self.conv2d_2(x)                                                 # [batches, 20, timesteps/3, frequency_bin_count/9]
        x = self.bn2(x)
        saved_20_channel = x
        x = nn.functional.relu(x)
        x = self.max_pool2d_2(x)                                             # [batches, 20, timesteps/3, frequency_bin_count/9]
        
        x = self.conv2d_3(x)                                                 # [batches, 40, timesteps/3, frequency_bin_count/9]
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.conv2d_4(x)                                                 # [batches, 30, timesteps/3, frequency_bin_count/9]
        x = self.bn4(x)
        x = nn.functional.relu(x)
        
        x = self.conv2d_5(x)                                                 # [batches, 20, timesteps/3, frequency_bin_count/9]
        x = self.bn5(x)
        x = nn.functional.relu(x)
        x = nn.functional.interpolate(x, scale_factor=(1, 3), mode='nearest')# [batches, 20, timesteps/3, frequency_bin_count/3]
        x = x * torch.sigmoid(saved_20_channel)
        
        x = self.conv2d_6(x)                                                 # [batches, 12, timesteps/3, frequency_bin_count/3]
        x = self.bn6(x)
        x = nn.functional.relu(x)
        x = nn.functional.interpolate(x, scale_factor=(3, 3), mode='nearest')# [batches, 12, timesteps,   frequency_bin_count]
        x = x * torch.sigmoid(saved_12_channel)

        x = self.conv2d_7(x)                                                 # [batches  2,  timesteps,   frequency_bin_count]
        x = nn.functional.relu(x)
        x = x * torch.sigmoid(saved_2_channel)
        x = x.transpose(2, 3)                                                # [batches, 2, frequencies, timesteps]
        
        return x
        
        
class BandSplitEncoder(nn.Module):
    def __init__(self, frequency_bins, features):
        super(self.__class__, self).__init__()
        self.layer_norm = nn.LayerNorm(frequency_bins)
        self.fc = nn.Linear(in_features=frequency_bins, out_features=features)
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc(x)
        return x
class BandSplitDecoder(nn.Module):
    def __init__(self, features, frequency_bins):
        super(self.__class__, self).__init__()
        self.layer_norm = nn.LayerNorm(features)
        self.fc1 = nn.Linear(in_features=features, out_features=features*4)
        self.fc2 = nn.Linear(in_features=features*4, out_features=frequency_bins)
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = nn.functional.sigmoid(x)
        x = self.fc2(x)
        return x
    
        
class BandSplitDualPath(nn.Module):
    def __init__(self, features): 
        super(self.__class__, self).__init__()
        self.time_norm = nn.GroupNorm(1, features)
        self.band_norm = nn.GroupNorm(1, features)
        
        self.time_lstm = nn.LSTM(input_size=features, hidden_size=features//2, bidirectional=True, batch_first=True)
        self.band_lstm = nn.LSTM(input_size=features, hidden_size=features//2, bidirectional=True, batch_first=True) 
        
        self.time_fc = nn.Linear(in_features=features, out_features=features)
        self.band_fc = nn.Linear(in_features=features, out_features=features)
        
    def forward(self, x):  
        batches, bands, timesteps, features = x.shape
        x = x.reshape(batches * bands, timesteps, features)
        s1   = x                # [batches * bands, timesteps, features] 
        x    = self.time_norm(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.time_lstm(x)
        x    = s1 + self.time_fc(x)
        
        x = x.reshape(batches, bands, timesteps, features)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batches * timesteps, bands, features)
        
        s2   = x                # [batches * timesteps, bands, features] 
        x    = self.band_norm(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.band_lstm(x)
        x    = s2 + self.band_fc(x)
        
        x = x.reshape(batches, timesteps, bands, features)
        x = x.transpose(1, 2)
        
        return x
        
        
class BandSplitRNN(nn.Module):
    def __init__(self, sample_rate, frequency_bin_count, dual_path_count, features):
        super(self.__class__, self).__init__()    
       
        bin_diff = (sample_rate / 2) / frequency_bin_count
        recommended_bands_frequencies = [(2, 150), (8, 500), (16, 2000), (16, 6000), (8, 12000), (4, 22050)]
        
        self.band_split_ranges = []
        start = 0
        for bands, split_freq in recommended_bands_frequencies:
            end = math.floor(split_freq / bin_diff)
            
            bins = end - start
            bands = min(bands, bins)
            
            base_width = bins // bands
            remainder = bins % bands
            
            s = start
            for i in range(bands):
                width = base_width + (1 if i < remainder else 0)
                e = s + width
                
                # print(f'{s}, {e}, {s * bin_diff:5.0f}, {e * bin_diff:5.0f}, diff: {e - s}')
                self.band_split_ranges.append((s, e))
                s = e
            start = end
        last_s, _ = self.band_split_ranges[-1]
        self.band_split_ranges[-1] = last_s, frequency_bin_count

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()        
        for start, end in self.band_split_ranges:
            self.encoders.append(
                BandSplitEncoder(frequency_bins=2*(end-start), features=features)
            )
            self.decoders.append(
                BandSplitDecoder(features=features, frequency_bins=2*(end-start))
            )
        
        self.dual_path = nn.Sequential(*[BandSplitDualPath(features) for _ in range(dual_path_count)])
 
        
    def forward(self, x):
        saved_input = x

        batches, channels, frequencies, timesteps = x.shape
        x = x.transpose(2, 3) # [batch, channels, timesteps, frequencies]

        bands = []
        for i, (start, end) in enumerate(self.band_split_ranges):
            band = x[:, :, :, start:end]
            band = band.permute(0, 2, 1, 3) # [batch, timesteps, channels, frequencies]
            band = band.reshape(batches, timesteps, channels * (end-start)) # [batch, timesteps, channels * frequencies]
            band = self.encoders[i](band)
            bands.append(band)
        
        x = torch.stack(bands, dim=1) # [batch, bands, timesteps, frequencies]
        
        batches, bands, timesteps, features = x.shape
        
        x = self.dual_path(x)
            
        bands = []
        for i, (start, end) in enumerate(self.band_split_ranges):
            band = self.decoders[i](x[:, i])
            band = band.reshape(batches, timesteps, channels, end-start)
            band = band.permute(0, 2, 3, 1)
            bands.append(band)
                        
        x = torch.cat(bands, dim=2)  # [batch, 2, frequency_bin_count, timesteps]

        x = saved_input * torch.sigmoid(x)
        x = torch.relu(x)

        return x
        

class WideConv(nn.Module):
    def __init__(self, sample_rate, frequency_bin_count):
        super(self.__class__, self).__init__()
        # w = (w ​+ 2 * padding - kernel​) // stride + 1
        # encoder
        self.conv_ft1 = nn.Conv2d(in_channels= 2, out_channels=20, kernel_size=(5,5), stride=(1,1), padding=2)

        self.conv_t1 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(1,3), stride=(1,3),  padding=0)
        self.conv_f1 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=(5,1), stride=(5,1),  padding=0)

        self.conv_ft2 = nn.Conv2d(in_channels=60, out_channels=80, kernel_size=(5,5), stride=(1,1), padding=2)

        self.conv_t2 = nn.Conv2d(in_channels=80, out_channels=100,  kernel_size=(1,5), stride=(1,5), padding=0)
        self.conv_f2 = nn.Conv2d(in_channels=100, out_channels=120, kernel_size=(5,1), stride=(5,1), padding=0)

        # bottleneck
        self.conv_ft3 = nn.Conv2d(in_channels=120, out_channels=120, kernel_size=(5,5), stride=(1,1), padding=2)

        # decoder
        self.conv_df2 = nn.Conv2d(in_channels=120, out_channels=100, kernel_size=(5,1), stride=(1,1), padding=(2, 0))
        self.conv_dt2 = nn.Conv2d(in_channels=100, out_channels= 80, kernel_size=(1,5), stride=(1,1), padding=(0, 2))

        self.conv_dft2 = nn.Conv2d(in_channels=80, out_channels=60, kernel_size=(5,5), stride=(1,1), padding=2)

        self.conv_df1 = nn.Conv2d(in_channels=60, out_channels=40, kernel_size=(5,1), stride=(1,1), padding=(2, 0))
        self.conv_dt1 = nn.Conv2d(in_channels=40, out_channels=20, kernel_size=(1,3), stride=(1,1), padding=(0, 1))

        self.conv_dft1 = nn.Conv2d(in_channels=20, out_channels=2, kernel_size=(5,5), stride=(1,1), padding=2)


    def forward(self, x):
        def gate(x, skip):
            return torch.cat(x, skip, dim=1)

        # [B, 2, Freq, T]
        skip_0 = x

        # encoder
        x = self.conv_ft1(x) # [B,  20, 1025, 345]
        skip_ft1 = x
        x = nn.functional.gelu(x)


        x = self.conv_t1(x)  # [B,  40, 1025, 115]
        skip_t1 = x
        x = nn.functional.gelu(x)


        x = self.conv_f1(x)  # [B,  60,  205, 115]
        skip_f1 = x
        x = nn.functional.gelu(x)


        x = self.conv_ft2(x) # [B,  80,  205, 115]
        skip_ft2 = x
        x = nn.functional.gelu(x)


        x = self.conv_t2(x)  # [B, 100,  205, 23]
        skip_t2 = x
        x = nn.functional.gelu(x)

        x = self.conv_f2(x)  # [B, 120,   41, 23]
        skip_f2 = x
        x = nn.functional.gelu(x)

        # bottleneck
        x = self.conv_ft3(x) # [B, 120,   41, 23]
        x = nn.functional.gelu(x)

        # decoder
        x *= torch.sigmoid(skip_f2)
        x = self.conv_df2(x) # [B, 100,   41, 23]
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(5, 1), mode='nearest')  # [B, 100,  205, 23]

        x = gate(x, skip_t2)
        x = self.conv_dt2(x) # [B,  80,   41, 23]
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(1, 5), mode='nearest') #  [B,  80,  205, 115]


        x = gate(x, skip_ft2)
        x = self.conv_dft2(x)     #  [B,  60,  205, 115]
        x = nn.functional.gelu(x)


        x = gate(x, skip_f1)
        x = self.conv_df1(x) # [B,  40,  205, 115]
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(5, 1), mode='nearest') #  [B,  40, 1025, 115]

        x = gate(x, skip_t1)
        x = self.conv_dt1(x) # [B,  20, 1025, 115]
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(1, 3), mode='nearest') #  [B,  20, 1025, 345]


        x = gate(x, skip_ft1)
        x = self.conv_dft1(x) # [B, 2, 1025, 345]
        x = nn.functional.gelu(x)

        x = gate(x, skip_0)
        x = torch.relu(x)

        return x


class BetterConv(nn.Module):
    def __init__(self, sample_rate, frequency_bin_count):
        super(self.__class__, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels=2, out_channels=12, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(12)
        self.max_pool2d_1 = nn.MaxPool2d(kernel_size=(5, 5), stride=(5, 5))


        self.conv2d_2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(20)
        self.max_pool2d_2 = nn.MaxPool2d(kernel_size=(1, 5), stride=(1, 5))

        self.conv2d_3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(40)
        self.conv2d_4 = nn.Conv2d(in_channels=40, out_channels=30, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(30)
        self.conv2d_5 = nn.Conv2d(in_channels=30, out_channels=20, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(20)

        self.conv2d_6 = nn.Conv2d(in_channels=20, out_channels=12, kernel_size=5, padding=2)
        self.bn6 = nn.BatchNorm2d(12)

        self.conv2d_7 = nn.Conv2d(in_channels=12, out_channels=2, kernel_size=5,  padding=2)

        self.lstm = nn.LSTM(input_size=frequency_bin_count//25, hidden_size=128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, frequency_bin_count//25)

    def forward(self, x):
        def gate(x, skip):
            return x * torch.tanh(skip)

        batches, channels, frequencies, timesteps = x.shape

        x = x.transpose(2, 3)                                                # [batches, 2, timesteps, frequencies]
        saved_2_channel = x

        x = self.conv2d_1(x)                                                 # [batches, 12, timesteps, frequency_bin_count]
        x = self.bn1(x)
        saved_12_channel = x
        x = nn.functional.gelu(x)
        x = self.max_pool2d_1(x)                                             # [batches, 12, timesteps/3, frequency_bin_count/3]

        x = self.conv2d_2(x)                                                 # [batches, 20, timesteps/3, frequency_bin_count/9]
        x = self.bn2(x)
        saved_20_channel = x
        x = nn.functional.gelu(x)
        x = self.max_pool2d_2(x)                                             # [batches, 20, timesteps/3, frequency_bin_count/9]

        x = self.conv2d_3(x)                                                 # [batches, 40, timesteps/3, frequency_bin_count/9]
        x = self.bn3(x)
        x = nn.functional.gelu(x)
        x = self.conv2d_4(x)                                                 # [batches, 30, timesteps/3, frequency_bin_count/9]
        x = self.bn4(x)
        x = nn.functional.gelu(x)

        batches, channels, steps, freqs = x.shape
        x = x.reshape(batches * channels, steps, freqs)
        x, _  = self.lstm(x)
        x = nn.functional.gelu(x)
        x = self.fc(x)
        x = x.reshape(batches, channels, steps, freqs)

        x = self.conv2d_5(x)                                                 # [batches, 20, timesteps/3, frequency_bin_count/9]
        x = self.bn5(x)
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(1, 5), mode='nearest')# [batches, 20, timesteps/3, frequency_bin_count/3]
        x = gate(x, saved_20_channel)

        x = self.conv2d_6(x)                                                 # [batches, 12, timesteps/3, frequency_bin_count/3]
        x = self.bn6(x)
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(5, 5), mode='nearest')# [batches, 12, timesteps,   frequency_bin_count]
        x = gate(x, saved_12_channel)

        x = self.conv2d_7(x)                                                 # [batches  2,  timesteps,   frequency_bin_count]
        x = nn.functional.gelu(x)
        x = torch.tanh(x) * saved_2_channel  # masking
        x = x.transpose(2, 3)                                                # [batches, 2, frequencies, timesteps]

        x = torch.relu(x)
        return x


    
train_path = './train_dataset.csv'
test_path = './test_dataset.csv'

n_fft = 2048
frequency_bin_count = n_fft // 2 + 1
device = torch.device('cpu') if len(sys.argv) > 1 and sys.argv[1] == 'infer' else torch.device('cuda:0')
model = BandSplitRNN(sample_rate=44100, frequency_bin_count=frequency_bin_count, dual_path_count=8, features=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1.8e-4)
batch_size = 2
batch_limit = 10000000
#batch_limit = 40

match sys.argv:
    case [_, 'prepare']:
        duration = 4
        overlap  = 2
        prepare_musdb_index('./musdb/train/', train_path, duration, overlap)
        prepare_musdb_index('./musdb/test/',  test_path,  duration, overlap)     
    case [_, 'info']:
        duration = 4 
        sample_rate=44100        
        timesteps = (duration * sample_rate) // (n_fft // 4) + 1
        torchinfo.summary(model, input_size=(batch_size, 2, frequency_bin_count, timesteps))
    case [_, 'load_test']:
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
        
    case [_, 'train']:
        def train():
            model.train()
            total_loss = 0
            last_file = 0
            batch_count = 0
            for batch, (x, y_correct) in enumerate(train_dataset):
                x = x.to(device)
                y_correct = y_correct.to(device)
                batch_count += 1                
                optimizer.zero_grad()
                y_predicted = model(x)
                
                loss = loss_fn(y_predicted, y_correct)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # gradient clipping
                optimizer.step()
                   
                total_loss += loss.item()
                    
                print(f'epoch {epoch:3d}/{end_epoch-1} batch: {batch:4d} train single: {loss.item():.5e} avg: {total_loss / batch_count:.5e}')
                
                if batch == batch_limit: break
            return total_loss / batch_count
        def test():
            model.eval()
            total_loss = 0
            batch_count = 0
            with torch.no_grad():
                for batch, (x, y_correct) in enumerate(test_dataset):
                    x = x.to(device)
                    y_correct = y_correct.to(device)
                    batch_count += 1
                    y_predicted = model(x)
                    
                    loss = loss_fn(y_predicted, y_correct)   
                    total_loss += loss.item()
                        
                    print(f'epoch {epoch:3d}/{end_epoch-1} batch: {batch:4d} test single: {loss.item():.5e} avg: {total_loss / batch_count:.5e}')
                    if batch == batch_limit: break
          
            return total_loss / batch_count
        
        
        train_dataset = torch.utils.data.DataLoader(MUSDB18Dataset(train_path, n_fft, True), batch_size=batch_size, pin_memory=True, shuffle=True)
        test_dataset = torch.utils.data.DataLoader(MUSDB18Dataset(test_path, n_fft, True), batch_size=batch_size, pin_memory=True)
       
        loss_fn = PerceptualLoss(sample_rate=44100, frequency_bin_count=frequency_bin_count).to(device)
        
        
        checkpoint_dir = './BandSplitRNN_checkpoint'
        log_path = './BandSplitRNN_training.csv'
        
        start_epoch = load_latest_checkpoint(checkpoint_dir, model, optimizer) + 1
        end_epoch = 200
        
        start_time = time.perf_counter()
        last_train_loss = 0
        last_test_loss = 0
        
        sum_epoch_duration = 0
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2, factor=0.6)
        current_lr = scheduler.get_last_lr()
        
        with open(log_path, 'a') as log:
            for epoch in range(start_epoch, end_epoch):
                train_start_time = time.perf_counter()
                train_loss = train()
                train_end_time = time.perf_counter()
                
                save_checkpoint(os.path.join(checkpoint_dir, 'latest.zip'), epoch, model, optimizer)
                if epoch != 0 and epoch % 1 == 0:
                    save_checkpoint(os.path.join(checkpoint_dir, f'{epoch}.zip'), epoch, model, optimizer)
                
                test_start_time = time.perf_counter()
                test_loss = test()
                test_end_time = time.perf_counter()
                
                train_delta = train_loss - last_train_loss if epoch != start_epoch else 0
                test_delta  = test_loss - last_test_loss   if epoch != start_epoch else 0
            
                last_train_loss = train_loss
                last_test_loss = test_loss
            
                epoch_duration = test_end_time  - train_start_time
                train_duration = train_end_time - train_start_time
                test_duration  = test_end_time  - test_start_time
                elapsed        = test_end_time  - start_time
                
                sum_epoch_duration += epoch_duration
                avg_epoch_duration = sum_epoch_duration / (epoch - start_epoch + 1)
                remaining = (end_epoch - epoch - 1) * avg_epoch_duration
                
                print(f'epoch: {epoch:3d}/{end_epoch-1} train: {train_loss:.5e} test: {test_loss:.5e} Δtrain: {train_delta:.5e} Δtest: {test_delta:.5e} train duration: {train_duration / 60:2.1f}m test duration: {test_duration / 60:2.1f}m  epoch duration: {epoch_duration/60:2.1f}m elapsed: {elapsed/60/60:2.1f}h eta: {remaining/60/60:2.1f}h')
                print(f'{epoch};{train_loss};{test_loss};{train_duration};{test_duration};{epoch_duration};{current_lr[0]}', file=log)
                log.flush()
                
                scheduler.step(test_loss)
                if scheduler.get_last_lr() != current_lr:
                    print(f'lowered learning rate from {current_lr} to {scheduler.get_last_lr()}')
                    current_lr = scheduler.get_last_lr()
                
                
                
    case [_, 'infer', checkpoint, input_path, output_path]:
        model.eval()
        with torch.no_grad():
            duration = 4
            overlap  = 2
            
            frames = int(duration * 44100)
            hop    = int((duration - overlap) * 44100)
        
            sample_rate, num_frames = audio_info(input_path)
            assert sample_rate == 44100
            
            load_checkpoint(checkpoint, model, optimizer)
            
            vocal_waveform = torch.zeros((2, num_frames), dtype=torch.float32)
            instr_waveform = torch.zeros((2, num_frames), dtype=torch.float32)
            weights = torch.zeros((1, num_frames),  dtype=torch.float32)
            
            window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)

            full_samples = audio_read(input_path, 0, num_frames)
            rms = full_samples.square().mean().sqrt()

            full_samples_normalised = full_samples / rms

            for start in range(0, num_frames, hop):
                end = start + frames
                
                print(f'{start:10d}/{num_frames}')
                
                samples = full_samples_normalised[:, start:end]
                samples = nn.functional.pad(samples, (0, frames - samples.shape[1]))

                full = torch.stft(samples, n_fft=n_fft, return_complex=True, window=window)
                full_abs = full.abs()
                
                phase = full / torch.clamp(full_abs, min=1e-24)
                
                x = full_abs.unsqueeze(0).to(device)
                y = model(x)[0].cpu()
           
                vocal = phase * torch.clamp(y, min=0)
                instr = phase * torch.clamp(full_abs - y, min=0)
                
                vocal_reconstructed = torch.istft(vocal, n_fft=n_fft, window=window, center=True) * rms
                instr_reconstructed = torch.istft(instr, n_fft=n_fft, window=window, center=True) * rms
                
                size = min(vocal_reconstructed.shape[1], num_frames-start)

                vocal_waveform[:, start:start+size] += vocal_reconstructed[:, 0:size]
                instr_waveform[:, start:start+size] += instr_reconstructed[:, 0:size]
                weights       [:, start:start+size] += 1
                             
            vocal_waveform /= weights
            instr_waveform /= weights


            path_obj = Path(output_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            soundfile.write(os.path.join(path_obj.parent, path_obj.stem+ '_vocal' + path_obj.suffix), vocal_waveform.t(), 44100)
            soundfile.write(os.path.join(path_obj.parent, path_obj.stem+ '_instr' + path_obj.suffix), instr_waveform.t(), 44100)
            print('complete')
    case _:
        print(f'unknown command')
        print(f'usage:')
        print(f'  prepare   | Prepares the train and test dataset indexes from ./musdb/train and ./musdb/test directories')
        print(f'  load_test | Reconstructs first track as specified in train dataset index to load_test.wav' )
        sys.exit(1)
        