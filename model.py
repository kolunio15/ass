import sys
import os
import time
import math
import torch
import torch.nn as nn
import soundfile
import torchinfo
import numpy as np
import shutil
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



def is_chunk_silent(chunk):
    rms = chunk.square().mean().sqrt()
    return rms < 1e-3


def prepare_musdb_index(input_path, output_path, duration, overlap):
    frames = int(duration * 44100)
    hop    = int((duration - overlap) * 44100)

    with open(output_path, 'w') as w:
        for track in os.scandir(input_path):
            bass_path    = os.path.join(track, 'bass.wav')
            drums_path   = os.path.join(track, 'drums.wav')
            other_path   = os.path.join(track, 'other.wav')
            vocals_path  = os.path.join(track, 'vocals.wav')

            sample_rate, num_frames = audio_info(bass_path)

            for t_start in range(0, num_frames - frames, hop):
                t_end = t_start + frames

                bass_silent   = is_chunk_silent(audio_read(bass_path, t_start, t_end))
                drums_silent  = is_chunk_silent(audio_read(drums_path, t_start, t_end))
                other_silent  = is_chunk_silent(audio_read(other_path, t_start, t_end))
                vocals_silent = is_chunk_silent(audio_read(vocals_path, t_start, t_end))

                print(f'{track.path};{t_start};{t_end};{bass_silent};{drums_silent};{other_silent};{vocals_silent}', file=w)
                print(f'{track.path};{t_start};{t_end};{bass_silent};{drums_silent};{other_silent};{vocals_silent}')

class MUSDB18Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_index_path, n_fft, amplitude_only):
        lines = []

        self.bass = []
        self.drums = []
        self.other = []
        self.vocals = []

        with open(dataset_index_path, 'r') as r:
            for line in r:
                track, start, end, bass_silent, drums_silent, other_silent, vocals_silent = line.split(';')
                start = int(start)
                end   = int(end)

                if 'False' == bass_silent.strip():   self.bass  .append((os.path.join(track, 'bass.wav'),   start, end))
                if 'False' == drums_silent.strip():  self.drums .append((os.path.join(track, 'drums.wav'),  start, end))
                if 'False' == other_silent.strip():  self.other .append((os.path.join(track, 'other.wav'),  start, end))
                if 'False' == vocals_silent.strip(): self.vocals.append((os.path.join(track, 'vocals.wav'), start, end))

        self.n_fft = n_fft
        self.window = torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32)
        self.amplitude_only = amplitude_only
        
    
    def __getitem__(self, index):
        bass   = audio_read(*self.bass  [torch.randint(high=len(self.bass),   size=(1,))]) * (torch.rand(1) * 1.25)
        drums  = audio_read(*self.drums [torch.randint(high=len(self.drums),  size=(1,))]) * (torch.rand(1) * 1.25)
        other  = audio_read(*self.other [torch.randint(high=len(self.other),  size=(1,))]) * (torch.rand(1) * 1.25)
        vocals = audio_read(*self.vocals[torch.randint(high=len(self.vocals), size=(1,))]) * (torch.rand(1) * 1.25)

        mix = bass + drums + other + vocals

        if mix.abs().max() > 1.0:
            maximum = mix.abs().max()
            mix    /= maximum
            # bass   /= maximum
            # drums  /= maximum
            # other  /= maximum
            vocals /= maximum

        mix    = torch.stft(mix,    n_fft=self.n_fft, return_complex=True, window=self.window)
        vocals = torch.stft(vocals, n_fft=self.n_fft, return_complex=True, window=self.window)

        if self.amplitude_only:
            return mix.abs(), vocals.abs()
        else:
            return mix, vocals
        
    def __len__(self):
        return max(len(self.bass), len(self.drums), len(self.other), len(self.vocals))

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
        frequency_loudness = 1.0 + (1 - (spl - spl_min) / (spl_max - spl_min))
        
        self.register_buffer('frequency_loudness', frequency_loudness, persistent=False)

    def forward(self, y_predicted, y_correct):
        loss = (torch.log1p(y_predicted) - torch.log1p(y_correct)).abs()        
        loss = (loss * self.frequency_loudness).mean()         
        return loss
        

class NaiveLSTM(nn.Module):
    def __init__(self, sample_rate, frequency_bin_count, hidden_layers):
        super(self.__class__, self).__init__()
        self.conv1d_channel_merge = nn.Conv1d(in_channels=2 * frequency_bin_count, out_channels=1 * frequency_bin_count, kernel_size=1)
        self.relu1 = nn.ReLU()
        self.bn = nn.BatchNorm1d(frequency_bin_count)
        self.lstm = nn.LSTM(input_size=frequency_bin_count, hidden_size=hidden_layers, num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)
        self.last_fc = nn.Linear(hidden_layers * 2, frequency_bin_count)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        saved = x
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
        x = torch.tanh(x) * saved

        return x
        
class NaiveConv(nn.Module):
    def __init__(self, sample_rate, frequency_bin_count):
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
        
        x = self.conv2d_5(x)                                                 # [batches, 20, timesteps/3, frequency_bin_count/9]
        x = self.bn5(x)
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(1, 3), mode='nearest')# [batches, 20, timesteps/3, frequency_bin_count/3]
        x = x * torch.tanh(saved_20_channel)

        
        x = self.conv2d_6(x)                                                 # [batches, 12, timesteps/3, frequency_bin_count/3]
        x = self.bn6(x)
        x = nn.functional.gelu(x)
        x = nn.functional.interpolate(x, scale_factor=(3, 3), mode='nearest')# [batches, 12, timesteps,   frequency_bin_count]
        x = x * torch.tanh(saved_12_channel)

        x = self.conv2d_7(x)                                                 # [batches  2,  timesteps,   frequency_bin_count]
        x = nn.functional.relu(x)

        x = torch.tanh(x) * saved_2_channel
        x = x.transpose(2, 3)                                                # [batches, 2, frequencies, timesteps]

        x = torch.relu(x)

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
            return x * torch.tanh(skip)

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

        x = torch.tanh(x) * skip_0
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


class TDC_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
    def forward(self, x): # [batches * timesteps, channels, frequencies]
        x = self.norm(x)
        x = self.conv(x)
        x = nn.functional.gelu(x)
        return x

class TDC(nn.Module): # Time-Distributed Convolutions (convolutions on frequencies only)
    def __init__(self, in_channels, growth_rate, layer_count, kernel_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.projection = nn.Conv1d(in_channels=in_channels+growth_rate*layer_count, out_channels=in_channels, kernel_size=1, padding=0)

        for _ in range(layer_count):
            self.layers.append(TDC_Layer(in_channels=in_channels, out_channels=growth_rate, kernel_size=kernel_size))
            in_channels = in_channels + growth_rate

    def forward(self, x):                                         # [batches, channels, frequencies, timesteps]
        def fwd(x):
            batches, channels, frequencies, timesteps = x.shape
            x = x.permute(0, 3, 1, 2)                                 # [batches, timesteps, channels, frequencies]
            x = x.reshape(batches * timesteps, channels, frequencies) # [batches * timesteps, channels, frequencies]
            for layer in self.layers:
                x = torch.cat((x, layer(x)), dim=1)
            x = self.projection(x)
            x = x.reshape(batches, timesteps, channels, frequencies)
            x = x.permute(0, 2, 3, 1)                                 # [batches, channels, frequencies, timesteps]

            return x

        return torch.utils.checkpoint.checkpoint(fwd, x, use_reentrant=False)

class TFC_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, time_kernel_size, freq_kernel_size):
        super().__init__()
        self.norm = nn.GroupNorm(1, in_channels)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(time_kernel_size, freq_kernel_size), padding=(time_kernel_size//2, freq_kernel_size//2))
    def forward(self, x): # [batches, channels, frequencies, timesteps]
        x = self.norm(x)
        x = self.conv(x)
        x = nn.functional.gelu(x)
        return x
class TFC(nn.Module): # Time-Frequency Convolutions
    def __init__(self, in_channels, growth_rate, layer_count, time_kernel_size, freq_kernel_size):
        super().__init__()
        self.layers = nn.ModuleList()
        self.projection = nn.Conv2d(in_channels=in_channels+growth_rate*layer_count, out_channels=in_channels, kernel_size=1, padding=0)

        for _ in range(layer_count):
            self.layers.append(TFC_Layer(in_channels=in_channels, out_channels=growth_rate, time_kernel_size=time_kernel_size, freq_kernel_size=freq_kernel_size))
            in_channels = in_channels + growth_rate
    def forward(self, x):
        def fwd(x):
            for layer in self.layers:
                x = torch.cat((x, layer(x)), dim=1)
            x = self.projection(x)
            return x

        return torch.utils.checkpoint.checkpoint(fwd, x, use_reentrant=False)


class TFC_TDC(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.in_channels = in_channels

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1))

        layer_count = 3

        self.tfc_1 = TFC(in_channels=in_channels, growth_rate=growth_rate, layer_count=layer_count, time_kernel_size=3, freq_kernel_size=3)

        self.tdc = TDC(in_channels=in_channels, growth_rate=growth_rate, layer_count=layer_count, kernel_size=3)

        self.tfc_2 = TFC(in_channels=in_channels, growth_rate=growth_rate, layer_count=layer_count, time_kernel_size=3, freq_kernel_size=3)

    def forward(self, x):
        skip_0 = x

        x = self.tfc_1(x)
        skip_1 = x

        x = self.tdc(x)
        x = x + skip_1

        x = self.tfc_2(x)
        x = x + self.conv(skip_0)

        return x

class TDF_Layer(nn.Module):
    def __init__(self, channels, feature_count, bottleneck_factor):
        super().__init__()
        self.norm = nn.GroupNorm(1, channels)
        self.linear = nn.Sequential(
            nn.Linear(feature_count, feature_count // bottleneck_factor),
            nn.GELU(),
            nn.Linear(feature_count // bottleneck_factor, feature_count),
            nn.GELU()
        )

    def forward(self, x): # [batches * timesteps, channels, frequencies]
        x = self.norm(x)
        x = self.linear(x)
        return x

class TDF(nn.Module):
    def __init__(self, in_channels, feature_count, bottleneck_factor):
        super().__init__()
        self.tdf_layer = TDF_Layer(in_channels, feature_count, bottleneck_factor)

    def forward(self, x):                                         # [batches, channels, frequencies, timesteps]
        def fwd(x):
            batches, channels, frequencies, timesteps = x.shape

            x = x.permute(0, 3, 1, 2)                             # [batches, timesteps, channels, frequencies]
            x = x.reshape(batches * timesteps, channels, frequencies)

            identity = x
            x = self.tdf_layer(x)
            x = x + identity

            x = x.reshape(batches, timesteps, channels, frequencies)
            x = x.permute(0, 2, 3, 1)                             # [batches, channels, frequencies, timesteps]

            return x

        return torch.utils.checkpoint.checkpoint(fwd, x, use_reentrant=False)


class TFC_TDF(nn.Module):
    def __init__(self, in_channels, growth_rate, feature_count):
        super().__init__()

        self.in_channels = in_channels

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3), padding=(1, 1))

        layer_count = 3

        self.tfc_1 = TFC(in_channels=in_channels, growth_rate=growth_rate, layer_count=layer_count, time_kernel_size=3, freq_kernel_size=3)

        self.tdf = TDF(in_channels=in_channels, feature_count=feature_count, bottleneck_factor=8)

        self.tfc_2 = TFC(in_channels=in_channels, growth_rate=growth_rate, layer_count=layer_count, time_kernel_size=3, freq_kernel_size=3)

    def forward(self, x):
        skip_0 = x

        x = self.tfc_1(x)
        skip_1 = x

        x = x + self.tdf(x)

        #x = self.tdf(x)
        #x = x + skip_1

        x = self.tfc_2(x)
        x = x + self.conv(skip_0)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, feature_count):
        super().__init__()

        self.tfc_tdf = TFC_TDF(in_channels, growth_rate, feature_count)
        self.down_sample = nn.Conv2d(in_channels=in_channels, out_channels=in_channels+growth_rate, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))

    def forward(self, x):
        x = self.tfc_tdf(x)
        skip = x
        x = self.down_sample(x)
        return skip, x



class Encoder(nn.Module):
    def __init__(self, in_channels, growth_rate, frequency_bin_count, encoder_layer_count):
        super().__init__()
        self.growth_rate = growth_rate
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=1, padding=0)
        self.layers = nn.ModuleList()

        in_channels = growth_rate
        feature_count = frequency_bin_count
        for _ in range(encoder_layer_count):
            self.layers.append(EncoderLayer(in_channels, growth_rate, feature_count=feature_count))
            in_channels += growth_rate
            feature_count = math.ceil(feature_count / 2)


    def forward(self, x): # [batches, channels, frequencies, timesteps]
        x = self.conv(x)
        skips = []
        for layer in self.layers:
            skip, x = layer(x)
            skips.append(skip)
        return skips, x

class DecoderLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, feature_count):
        super().__init__()
        self.in_channels = in_channels
        self.up_sample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels-growth_rate, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.tfc_tdc = TFC_TDF(in_channels-growth_rate, growth_rate, feature_count)
    def forward(self, x, skip):
        x = self.up_sample(x, output_size=skip.shape)
        x *= torch.tanh(skip)
        x = self.tfc_tdc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate, frequency_bin_count, encoder_layer_count):
        super().__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels - growth_rate * encoder_layer_count, out_channels=out_channels, kernel_size=1, padding=0)
        self.layers = nn.ModuleList()

        feature_counts = []
        f = frequency_bin_count
        for _ in range(encoder_layer_count):
            feature_counts.append(f)
            f = math.ceil(f / 2)
        feature_counts.reverse()

        for feature_count in feature_counts:
            self.layers.append(DecoderLayer(in_channels=in_channels, growth_rate=growth_rate, feature_count=feature_count))
            in_channels -= growth_rate

    def forward(self, skips, x):
        for layer, skip in zip(self.layers, skips[::-1]):
            x = layer(x, skip)
        x = self.conv(x)
        return x


class DenseConvDualPath(nn.Module):
    def __init__(self, features):
        super(self.__class__, self).__init__()
        self.time_norm = nn.GroupNorm(1, features)
        self.band_norm = nn.GroupNorm(1, features)

        self.time_lstm = nn.LSTM(input_size=features, hidden_size=features, bidirectional=True, batch_first=True)
        self.band_lstm = nn.LSTM(input_size=features, hidden_size=features, bidirectional=True, batch_first=True)

        self.time_fc = nn.Linear(in_features=features*2, out_features=features)
        self.band_fc = nn.Linear(in_features=features*2, out_features=features)

    def forward(self, x):
        batches, channels, features, timesteps = x.shape
        x = x.transpose(2, 3)
        x = x.reshape(batches * channels, timesteps, features)
        s1   = x                # [batches * channels, timesteps, features]
        x    = self.time_norm(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.time_lstm(x)
        x    = s1 + self.time_fc(x)

        x = x.reshape(batches, channels, timesteps, features)
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(batches * timesteps, channels, features)

        s2   = x                # [batches * timesteps, channels, features]
        x    = self.band_norm(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.band_lstm(x)
        x    = s2 + self.band_fc(x)

        x = x.reshape(batches, timesteps, channels, features)
        x = x.permute(0, 2, 3, 1)

        return x

class DenseConvBottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate, features, head_count, layer_count):
        super(self.__class__, self).__init__()

        self.head_count = head_count

        self.tfc_tdf = TFC_TDF(in_channels=in_channels, growth_rate=growth_rate, feature_count=features)
        self.layers = nn.ModuleList()
        for _ in range(layer_count):
            self.layers.append(DenseConvDualPath(features=features))

    def forward(self, x):
        x = self.tfc_tdf(x)

        # split
        batches, channels, frequencies, timesteps = x.shape
        new_channel_count = channels // self.head_count

        x = x.permute(0, 2, 3, 1) # [batches, frequencies, timesteps, channels]
        x = x.reshape(batches, frequencies, timesteps, self.head_count, new_channel_count)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(batches * self.head_count, new_channel_count, frequencies, timesteps)

        for layer in self.layers:
            x = layer(x)

        x = x.reshape(batches, self.head_count, new_channel_count, frequencies, timesteps)
        x = x.permute(0, 3, 4, 1, 2)
        x = x.reshape(batches, frequencies, timesteps, channels)
        x = x.permute(0, 3, 1, 2)

        return x

class DenseConv(nn.Module):
    def __init__(self, frequency_bin_count, growth_rate, encoder_layer_count):
        super().__init__()
        self.encoder = Encoder(in_channels=2, growth_rate=growth_rate, frequency_bin_count=frequency_bin_count, encoder_layer_count=encoder_layer_count)
        self.decoder = Decoder(in_channels=growth_rate*(encoder_layer_count+1), out_channels=2, growth_rate=growth_rate, encoder_layer_count=encoder_layer_count, frequency_bin_count=frequency_bin_count)

        self.bottleneck = DenseConvBottleneck(in_channels=growth_rate*(encoder_layer_count+1), growth_rate=growth_rate, features=math.ceil(frequency_bin_count / (2**encoder_layer_count)), head_count=1, layer_count=4)
    def forward(self, x):
        initial = x
        skips, x = self.encoder(x)

        x = self.bottleneck(x)

        x = self.decoder(skips, x)

        x = initial * torch.tanh(x)
        x = torch.relu(x)
        return x


train_path = './train_dataset.csv'
test_path = './test_dataset.csv'

n_fft = 1024
frequency_bin_count = n_fft // 2 + 1
device = torch.device('cpu')
if torch.cuda.is_available(): device = torch.device('cuda:0')
model = DenseConv(frequency_bin_count=frequency_bin_count, growth_rate=32, encoder_layer_count=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1.8e-4)

batch_size = 1
batch_limit = 200

checkpoint_dir = './DenseConv5_checkpoint/'
log_path    = './DenseConv5_training.csv'

match sys.argv:
    case [_, 'gui']:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        import threading
        import librosa
        import re
        import matplotlib.cm as cm

        try:
            from PIL import Image, ImageTk
        except ImportError:
            print("ERROR: Pillow (PIL) is not installed. Run: pip install pillow")
            sys.exit(1)

        try:
            import sounddevice as sd

            has_sd = True
        except ImportError:
            has_sd = False
            print("Warning: sounddevice is not installed. Audio playback will be disabled.")

        try:
            from tkinterdnd2 import TkinterDnD, DND_FILES

            TkClass = TkinterDnD.Tk
            has_dnd = True
        except ImportError:
            TkClass = tk.Tk
            has_dnd = False
            print("Warning: tkinterdnd2 is not installed. Drag and Drop will be disabled.")


        THEMES = {
            "Cyberpunk (Default)": {
                "BG_COLOR": "#0f0f17",
                "PANEL_BG": "#1a1a24",
                "TEXT_COLOR": "#c0caf5",
                "ACCENT_CYAN": "#00ffcc",
                "ACCENT_PINK": "#ff007f",
                "CURSOR_COLOR": "#ffffff"
            },
            "Light Minimal": {
                "BG_COLOR": "#f0f2f5",
                "PANEL_BG": "#ffffff",
                "TEXT_COLOR": "#2d3748",
                "ACCENT_CYAN": "#3182ce",
                "ACCENT_PINK": "#e53e3e",
                "CURSOR_COLOR": "#1a202c"
            },
            "Solarized Light": {
                "BG_COLOR": "#fdf6e3",
                "PANEL_BG": "#eee8d5",
                "TEXT_COLOR": "#657b83",
                "ACCENT_CYAN": "#268bd2",
                "ACCENT_PINK": "#d33682",
                "CURSOR_COLOR": "#073642"
            },
            "Dracula": {
                "BG_COLOR": "#282a36",
                "PANEL_BG": "#44475a",
                "TEXT_COLOR": "#f8f8f2",
                "ACCENT_CYAN": "#8be9fd",
                "ACCENT_PINK": "#ff79c6",
                "CURSOR_COLOR": "#ffffff"
            },
            "Monokai": {
                "BG_COLOR": "#272822",
                "PANEL_BG": "#3e3d32",
                "TEXT_COLOR": "#f8f8f2",
                "ACCENT_CYAN": "#a6e22e",
                "ACCENT_PINK": "#f92672",
                "CURSOR_COLOR": "#ffffff"
            },
            "Gruvbox": {
                "BG_COLOR": "#282828",
                "PANEL_BG": "#3c3836",
                "TEXT_COLOR": "#ebdbb2",
                "ACCENT_CYAN": "#8ec07c",
                "ACCENT_PINK": "#fb4934",
                "CURSOR_COLOR": "#ffffff"
            }
        }

        class AudioSeparationGUI(TkClass):
            def __init__(self):
                super().__init__()
                self.title("Audio Separation - GUI")
                self.geometry("1200x850")
                
                self.current_theme_name = tk.StringVar(value="Cyberpunk (Default)")

                self.queue = []
                self.is_running = False
                self.is_paused = False
                self.current_thread = None

                self.tracks_data = {}
                self.current_track = None

                self.audio_org = np.zeros(0, dtype=np.float32)
                self.audio_voc = np.zeros(0, dtype=np.float32)
                self.audio_inst = np.zeros(0, dtype=np.float32)

                self.full_pil_images = []
                self.tk_images = []  
                self.loading_spec = False

                self.is_playing = False
                self.stream = None
                self.play_idx = 0
                self.view_start = 0.0
                self.cursor_time = 0.0

                self.vol_org = 0.0
                self.vol_voc = 1.0
                self.vol_inst = 0.0

                self.stft_sep_running = False
                self.stft_sep_thread = None

                self.create_widgets()
                self.apply_theme()
                self.bind("<space>", self.on_space_key)
                self.bind("1", self.on_key_1)
                self.bind("2", self.on_key_2)
                self.bind("3", self.on_key_3)

                self.scan_folder(self.out_dir_var.get())
                self.update_playback_cursor()

            def get_theme_colors(self):
                return THEMES[self.current_theme_name.get()]

            def apply_theme(self):
                colors = self.get_theme_colors()
                
                self.configure(bg=colors["BG_COLOR"])
                
                style = ttk.Style(self)
                style.theme_use('clam')

                style.configure(".", background=colors["BG_COLOR"], foreground=colors["TEXT_COLOR"], font=('Segoe UI', 10))
                style.configure("TFrame", background=colors["BG_COLOR"])
                style.configure("TNotebook", background=colors["BG_COLOR"], borderwidth=0)
                style.configure("TNotebook.Tab", background=colors["PANEL_BG"], foreground=colors["TEXT_COLOR"], padding=[15, 5])
                style.map("TNotebook.Tab", background=[("selected", colors["ACCENT_PINK"])], foreground=[("selected", "#ffffff")])

                style.configure("TLabelframe", background=colors["PANEL_BG"], foreground=colors["ACCENT_CYAN"], bordercolor=colors["ACCENT_PINK"],
                                borderwidth=1)
                style.configure("TLabelframe.Label", background=colors["PANEL_BG"], foreground=colors["ACCENT_CYAN"],
                                font=('Segoe UI', 10, 'bold'))

                style.configure("TButton", background=colors["PANEL_BG"], foreground=colors["ACCENT_CYAN"], bordercolor=colors["ACCENT_CYAN"],
                                focuscolor=colors["ACCENT_PINK"], padding=5)
                style.map("TButton", background=[("active", colors["ACCENT_CYAN"])], foreground=[("active", colors["BG_COLOR"])])

                style.configure("TEntry", fieldbackground=colors["PANEL_BG"], foreground=colors["ACCENT_CYAN"], bordercolor=colors["ACCENT_CYAN"])
                style.configure("Horizontal.TScrollbar", background=colors["PANEL_BG"], bordercolor=colors["BG_COLOR"],
                                arrowcolor=colors["ACCENT_CYAN"], troughcolor=colors["BG_COLOR"])

                style.configure("Treeview", background=colors["PANEL_BG"], fieldbackground=colors["PANEL_BG"], foreground=colors["TEXT_COLOR"],
                                rowheight=25, borderwidth=0)
                style.configure("Treeview.Heading", background=colors["BG_COLOR"], foreground=colors["ACCENT_PINK"],
                                font=('Segoe UI', 10, 'bold'))
                style.map("Treeview", background=[("selected", colors["ACCENT_PINK"])], foreground=[("selected", "#ffffff")])

                self.track_listbox.config(bg=colors["PANEL_BG"], fg=colors["TEXT_COLOR"], selectbackground=colors["ACCENT_PINK"], highlightcolor=colors["ACCENT_CYAN"])
                self.canvas.config(bg=colors["BG_COLOR"])
                
                self.render_plot()
                self.update_file_status_ui()
                
            def on_theme_change(self, *args):
                self.apply_theme()

            def on_space_key(self, event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry)): return
                if "button" in str(event.widget.winfo_class()).lower(): return
                self.toggle_audio_playback()
                
            def on_key_1(self, event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry)): return
                if self.notebook.index(self.notebook.select()) != 1: return # Only active on Tab 2
                self.vol_org_var.set(1.0)
                self.vol_voc_var.set(0.0)
                self.vol_inst_var.set(0.0)
                self.update_volumes()

            def on_key_2(self, event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry)): return
                if self.notebook.index(self.notebook.select()) != 1: return
                self.vol_org_var.set(0.0)
                self.vol_voc_var.set(1.0)
                self.vol_inst_var.set(0.0)
                self.update_volumes()

            def on_key_3(self, event):
                if isinstance(event.widget, (tk.Entry, ttk.Entry)): return
                if self.notebook.index(self.notebook.select()) != 1: return
                self.vol_org_var.set(0.0)
                self.vol_voc_var.set(0.0)
                self.vol_inst_var.set(1.0)
                self.update_volumes()

            def create_widgets(self):
                top_bar = ttk.Frame(self)
                top_bar.pack(fill=tk.X, padx=10, pady=(5, 0))
                
                ttk.Label(top_bar, text="Theme:").pack(side=tk.LEFT, padx=(0, 5))
                theme_combo = ttk.Combobox(top_bar, textvariable=self.current_theme_name, values=list(THEMES.keys()), state="readonly", width=20)
                theme_combo.pack(side=tk.LEFT)
                theme_combo.bind("<<ComboboxSelected>>", self.on_theme_change)

                self.notebook = ttk.Notebook(self)
                self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                # ================= Queue Tab =================
                infer_frame = ttk.Frame(self.notebook)
                self.notebook.add(infer_frame, text="Separation Queue")

                control_frame = ttk.LabelFrame(infer_frame, text=" Settings & Controls ")
                control_frame.pack(fill=tk.X, padx=10, pady=10)

                ttk.Label(control_frame, text="Checkpoint Path:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
                base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                default_ckpt = os.path.join(base_dir, "DenseConv5_checkpoint_76a079c", "latest.zip")
                self.ckpt_var = tk.StringVar(value=default_ckpt)
                ttk.Entry(control_frame, textvariable=self.ckpt_var, width=70).grid(row=0, column=1, padx=10, pady=10)
                ttk.Button(control_frame, text="Browse", command=self.browse_ckpt).grid(row=0, column=2, padx=10,
                                                                                        pady=10)

                ttk.Label(control_frame, text="Output Dir:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
                self.out_dir_var = tk.StringVar(value=os.path.join(base_dir, "output"))
                ttk.Entry(control_frame, textvariable=self.out_dir_var, width=70).grid(row=1, column=1, padx=10,
                                                                                       pady=10)
                ttk.Button(control_frame, text="Browse", command=self.browse_out_dir).grid(row=1, column=2, padx=10,
                                                                                           pady=10)

                time_frame = ttk.Frame(control_frame, style="TFrame")
                time_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W, padx=10, pady=5)

                ttk.Label(time_frame, text="Start Sec (0=start):").pack(side=tk.LEFT)
                self.start_var = tk.StringVar(value="0")
                ttk.Entry(time_frame, textvariable=self.start_var, width=10).pack(side=tk.LEFT, padx=10)

                ttk.Label(time_frame, text="End Sec (0=end):").pack(side=tk.LEFT, padx=(30, 0))
                self.end_var = tk.StringVar(value="0")
                ttk.Entry(time_frame, textvariable=self.end_var, width=10).pack(side=tk.LEFT, padx=10)

                btn_frame = ttk.Frame(infer_frame)
                btn_frame.pack(fill=tk.X, padx=10, pady=5)
                ttk.Button(btn_frame, text="Add Files...", command=self.add_files).pack(side=tk.LEFT, padx=5)
                self.btn_run = ttk.Button(btn_frame, text="Run Separation", command=self.toggle_run)
                self.btn_run.pack(side=tk.LEFT, padx=5)
                self.btn_pause = ttk.Button(btn_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
                self.btn_pause.pack(side=tk.LEFT, padx=5)
                ttk.Button(btn_frame, text="Clear Queue", command=self.clear_queue).pack(side=tk.LEFT, padx=5)

                tree_frame = ttk.Frame(infer_frame)
                tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                self.tree = ttk.Treeview(tree_frame, columns=("File", "Progress", "Status"), show="headings")
                self.tree.heading("File", text="File Path")
                self.tree.heading("Progress", text="Progress")
                self.tree.heading("Status", text="Status")
                self.tree.column("File", width=550)
                self.tree.column("Progress", width=250, anchor=tk.CENTER)
                self.tree.column("Status", width=150, anchor=tk.CENTER)
                self.tree.pack(fill=tk.BOTH, expand=True)

                self.tree.bind('<Double-1>', self.on_queue_double_click)

                if has_dnd:
                    self.tree.drop_target_register(DND_FILES)
                    self.tree.dnd_bind('<<Drop>>', self.handle_drop)

                # =================  Player Tab =================
                spec_frame = ttk.Frame(self.notebook)
                self.notebook.add(spec_frame, text="Interactive Player")

                paned = ttk.PanedWindow(spec_frame, orient=tk.HORIZONTAL)
                paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                left_panel = ttk.Frame(paned, style="TFrame")
                paned.add(left_panel, weight=1)

                self.lbl_converted = ttk.Label(left_panel, text="Converted Tracks", font=('Segoe UI', 12, 'bold'))
                self.lbl_converted.pack(anchor=tk.W, pady=(0, 5))
                self.track_listbox = tk.Listbox(left_panel, exportselection=False, bd=0, highlightthickness=1)
                self.track_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
                self.track_listbox.bind('<Double-1>', self.on_track_select)

                ttk.Button(left_panel, text="Scan Output Dir...",
                           command=lambda: self.scan_folder(self.out_dir_var.get())).pack(fill=tk.X, pady=5)

                right_panel = ttk.Frame(paned, style="TFrame")
                paned.add(right_panel, weight=4)

                top_controls = ttk.Frame(right_panel)
                top_controls.pack(fill=tk.X, pady=(0, 5))

                ttk.Label(top_controls, text="View (s) [Ctrl+Scroll zooms]:").pack(side=tk.LEFT)
                self.window_var = tk.StringVar(value="15.0")
                ttk.Entry(top_controls, textvariable=self.window_var, width=5).pack(side=tk.LEFT, padx=5)
                self.window_var.trace_add("write", lambda *args: [self.update_scrollbar(), self.delayed_render()])

                ttk.Label(top_controls, text="FFT:").pack(side=tk.LEFT, padx=(15, 0))
                self.fft_var = tk.StringVar(value=str(n_fft))
                ttk.Entry(top_controls, textvariable=self.fft_var, width=6).pack(side=tk.LEFT, padx=5)

                player_controls = ttk.Frame(top_controls)
                player_controls.pack(side=tk.RIGHT)

                vol_frame = ttk.Frame(player_controls)
                vol_frame.pack(side=tk.LEFT, padx=10)

                self.vol_org_var = tk.DoubleVar(value=0.0)
                self.vol_voc_var = tk.DoubleVar(value=1.0)
                self.vol_inst_var = tk.DoubleVar(value=0.0)

                ttk.Label(vol_frame, text="Original [1]:").pack(side=tk.LEFT)
                ttk.Scale(vol_frame, from_=0, to=1.5, variable=self.vol_org_var, orient=tk.HORIZONTAL, length=60,
                          command=self.update_volumes).pack(side=tk.LEFT, padx=(0, 5))
                ttk.Label(vol_frame, text="Vocal [2]:").pack(side=tk.LEFT)
                ttk.Scale(vol_frame, from_=0, to=1.5, variable=self.vol_voc_var, orient=tk.HORIZONTAL, length=60,
                          command=self.update_volumes).pack(side=tk.LEFT, padx=(0, 5))
                ttk.Label(vol_frame, text="Instrumental [3]:").pack(side=tk.LEFT)
                ttk.Scale(vol_frame, from_=0, to=1.5, variable=self.vol_inst_var, orient=tk.HORIZONTAL, length=60,
                          command=self.update_volumes).pack(side=tk.LEFT, padx=(0, 5))

                self.btn_play = ttk.Button(player_controls, text="▶ Play (Space)", command=self.toggle_audio_playback)
                self.btn_play.pack(side=tk.LEFT, padx=(10, 5))
                self.time_label = ttk.Label(player_controls, text="Time: 0.0s", font=('Consolas', 10))
                self.time_label.pack(side=tk.LEFT, padx=5)

                self.file_status_frame = ttk.Frame(right_panel, style="TFrame")
                self.file_status_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

                self.fs_lbls = {}
                self.fs_btns = {}

                for i, ttype in enumerate(['original', 'vocal', 'instr']):
                    frame = ttk.Frame(self.file_status_frame, style="TFrame")
                    frame.pack(side=tk.LEFT, padx=(0, 15))

                    ttk.Label(frame, text=f"{ttype.capitalize()}:").pack(side=tk.LEFT)
                    lbl = ttk.Label(frame, text="Missing")
                    lbl.pack(side=tk.LEFT, padx=(5, 5))
                    btn = ttk.Button(frame, text="Browse", command=lambda t=ttype: self.browse_missing(t))
                    btn.pack(side=tk.LEFT)

                    self.fs_lbls[ttype] = lbl
                    self.fs_btns[ttype] = btn

           
                self.canvas_frame = ttk.Frame(right_panel, style="TFrame")
                self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                self.scrollbar = ttk.Scrollbar(right_panel, orient=tk.HORIZONTAL, command=self.on_scroll)
                self.scrollbar.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=0)

                self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0)
                self.canvas.pack(fill=tk.BOTH, expand=True)

                self.canvas.bind('<ButtonPress-1>', self.on_canvas_click)
                self.canvas.bind('<ButtonPress-2>', self.on_canvas_pan_start)
                self.canvas.bind('<ButtonPress-3>', self.on_canvas_pan_start)
                self.canvas.bind('<B2-Motion>', self.on_canvas_pan_drag)
                self.canvas.bind('<B3-Motion>', self.on_canvas_pan_drag)
                self.canvas.bind('<MouseWheel>', self.on_canvas_scroll)
                self.canvas.bind('<Configure>', lambda e: self.render_plot())

                # ================= Simple STFT Center Remove Tab =================
                stft_frame = ttk.Frame(self.notebook)
                self.notebook.add(stft_frame, text="Simple STFT Center Remove")

                stft_ctrl = ttk.LabelFrame(stft_frame, text=" Center removal (200–4000 Hz) ")
                stft_ctrl.pack(fill=tk.X, padx=10, pady=10)

                ttk.Label(stft_ctrl, text="Input WAV (stereo):").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
                self.stft_in_var = tk.StringVar()
                ttk.Entry(stft_ctrl, textvariable=self.stft_in_var, width=70).grid(row=0, column=1, padx=10, pady=10)
                ttk.Button(stft_ctrl, text="Browse...", command=self.browse_stft_input).grid(row=0, column=2, padx=10,
                                                                                             pady=10)

                ttk.Label(stft_ctrl, text="Output WAV:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
                self.stft_out_var = tk.StringVar()
                ttk.Entry(stft_ctrl, textvariable=self.stft_out_var, width=70).grid(row=1, column=1, padx=10, pady=10)
                ttk.Button(stft_ctrl, text="Browse...", command=self.browse_stft_output).grid(row=1, column=2, padx=10,
                                                                                              pady=10)

                stft_btn_frame = ttk.Frame(stft_frame)
                stft_btn_frame.pack(fill=tk.X, padx=10, pady=5)
                self.stft_run_btn = ttk.Button(stft_btn_frame, text="Run Center Removal",
                                               command=self.run_stft_separation)
                self.stft_run_btn.pack(side=tk.LEFT, padx=5)

                self.stft_status = ttk.Label(stft_btn_frame, text="Idle")
                self.stft_status.pack(side=tk.LEFT, padx=20)


            def on_queue_double_click(self, event):
                selection = self.tree.selection()
                if not selection: return
                item = selection[0]
                file_path = self.tree.item(item, "values")[0]

                target_name = None
                for name, data in self.tracks_data.items():
                    if data.get('original') == file_path:
                        target_name = name
                        break

                if not target_name:
                    self.add_track_to_data(file_path, "", "", 0.0, 0.0)
                    self.refresh_track_listbox()
                    target_name = list(self.tracks_data.keys())[-1]

                idx = list(self.tracks_data.keys()).index(target_name)
                self.track_listbox.selection_clear(0, tk.END)
                self.track_listbox.selection_set(idx)

                self.notebook.select(1)
                self.on_track_select(None)

            def get_max_audio_duration(self):
                max_len = max(len(self.audio_org), len(self.audio_voc), len(self.audio_inst))
                return max_len / 44100.0 if max_len > 0 else 10.0

            def update_volumes(self, *args):
                self.vol_org = self.vol_org_var.get()
                self.vol_voc = self.vol_voc_var.get()
                self.vol_inst = self.vol_inst_var.get()

            def browse_ckpt(self):
                self.focus_set()
                f = filedialog.askopenfilename(filetypes=[("Zip Checkpoint", "*.zip"), ("All Files", "*.*")])
                if f: self.ckpt_var.set(f)

            def browse_out_dir(self):
                self.focus_set()
                d = filedialog.askdirectory()
                if d: self.out_dir_var.set(d)

            def add_files(self):
                self.focus_set()
                files = filedialog.askopenfilenames(
                    filetypes=[("Audio Files", "*.wav *.flac *.mp3"), ("All Files", "*.*")])
                for f in files:
                    self.tree.insert("", tk.END, values=(f, "[░░░░░░░░░░] 0%", "Pending"))

            def clear_queue(self):
                self.focus_set()
                if not self.is_running:
                    for item in self.tree.get_children():
                        self.tree.delete(item)

            def toggle_run(self):
                self.focus_set()
                if not self.is_running:
                    self.is_running = True
                    self.is_paused = False
                    self.btn_run.config(text="Stop Processing")
                    self.btn_pause.config(state=tk.NORMAL)

                    ckpt = self.ckpt_var.get()
                    out_dir = self.out_dir_var.get()
                    try:
                        s_sec = float(self.start_var.get())
                        e_sec = float(self.end_var.get())
                    except ValueError:
                        s_sec, e_sec = 0.0, 0.0

                    self.current_thread = threading.Thread(target=self.process_queue,
                                                           args=(ckpt, out_dir, s_sec, e_sec), daemon=True)
                    self.current_thread.start()
                else:
                    self.is_running = False
                    self.is_paused = False
                    self.btn_run.config(text="Run Separation")
                    self.btn_pause.config(state=tk.DISABLED, text="Pause")

            def toggle_pause(self):
                self.focus_set()
                self.is_paused = not self.is_paused
                if self.is_paused:
                    self.btn_pause.config(text="Resume")
                else:
                    self.btn_pause.config(text="Pause")

            def update_file_status_ui(self):
                if not self.current_track: return
                colors = self.get_theme_colors()
                for ttype in ['original', 'vocal', 'instr']:
                    path = self.current_track.get(ttype)
                    if path and os.path.exists(path):
                        self.fs_lbls[ttype].config(text="Loaded", foreground=colors["ACCENT_CYAN"])
                        self.fs_btns[ttype].config(text="Replace")
                    else:
                        self.fs_lbls[ttype].config(text="Missing", foreground=colors["ACCENT_PINK"])
                        self.fs_btns[ttype].config(text="Browse")
                
                self.lbl_converted.config(foreground=colors["ACCENT_CYAN"])
                self.time_label.config(foreground=colors["ACCENT_PINK"])
                self.stft_status.config(foreground=colors["ACCENT_CYAN"])

            def browse_missing(self, track_type):
                self.focus_set()
                if not self.current_track: return
                f = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.flac *.mp3"), ("All Files", "*.*")])
                if f:
                    self.current_track[track_type] = f
                    self.update_file_status_ui()
                    self.load_track_into_memory()
                    self.view_start = 0.0
                    self.update_scrollbar()
                    self.start_compute_thread()

            def scan_folder(self, folder_path=None):
                self.focus_set()
                if folder_path is None:
                    folder_path = filedialog.askdirectory(title="Select folder to scan")
                if not folder_path or not os.path.exists(folder_path): return

                found = 0
                for f in os.listdir(folder_path):
                    match = re.match(r"^(.*)_vocal_gui(?:_([0-9.]+)_([0-9.]+))?\.(wav|flac|mp3)$", f)
                    if match:
                        stem = match.group(1)
                        s_start = float(match.group(2)) if match.group(2) else 0.0
                        s_end = float(match.group(3)) if match.group(3) else 0.0
                        ext = match.group(4)

                        voc_path = os.path.join(folder_path, f)
                        inst_name = f"{stem}_instr_gui_{s_start}_{s_end}.{ext}" if match.group(
                            2) else f"{stem}_instr_gui.{ext}"
                        inst_path = os.path.join(folder_path, inst_name)
                        if not os.path.exists(inst_path): continue

                        org_path = None
                        for org_ext in ['.wav', '.flac', '.mp3']:
                            temp = os.path.join(folder_path, stem + org_ext)
                            if os.path.exists(temp):
                                org_path = temp
                                break

                        self.add_track_to_data(org_path or f"(Missing Original) {stem}", voc_path, inst_path, s_start,
                                               s_end)
                        found += 1

                if folder_path != os.getcwd() and found > 0 and folder_path != self.out_dir_var.get():
                    messagebox.showinfo("Scan complete", f"Found {found} separated track(s)!")
                self.refresh_track_listbox()

            def add_track_to_data(self, org, voc, inst, start_sec, end_sec):
                base_path = org if (org and os.path.exists(org)) else (voc if (voc and os.path.exists(voc)) else inst)
                name = os.path.basename(base_path) if base_path else "Unknown"
                if start_sec > 0 or end_sec > 0:
                    name += f" [{start_sec}s - {end_sec}s]"

                self.tracks_data[name] = {
                    "original": org,
                    "vocal": voc,
                    "instr": inst,
                    "frag_start": start_sec,
                    "frag_end": end_sec
                }

            def refresh_track_listbox(self):
                self.track_listbox.delete(0, tk.END)
                for name in self.tracks_data.keys():
                    self.track_listbox.insert(tk.END, name)

            def handle_drop(self, event):
                files = self.tk.splitlist(event.data)
                for f in files:
                    self.tree.insert("", tk.END, values=(f, "[░░░░░░░░░░] 0%", "Pending"))

            def update_status(self, item_id, status):
                try:
                    self.tree.set(item_id, column="Status", value=status)
                except tk.TclError:
                    pass

            def update_progress(self, item_id, pct, elapsed_sec=0, eta_sec=0):
                try:
                    bar_len = 10
                    filled = int((pct / 100) * bar_len)
                    bar = '█' * filled + '░' * (bar_len - filled)
                    if pct >= 100:
                        time_str = f" ({elapsed_sec}s)" if elapsed_sec > 0 else ""
                    else:
                        time_str = f" ({elapsed_sec}s | ETA: {eta_sec}s)" if elapsed_sec > 0 else ""
                    self.tree.set(item_id, column="Progress", value=f"[{bar}] {pct}%{time_str}")
                except tk.TclError:
                    pass

            def process_queue(self, checkpoint, out_dir, start_sec, end_sec):
                if not os.path.exists(checkpoint):
                    self.after(0, lambda: messagebox.showerror("Error", "Checkpoint not found!"))
                    self.after(0, self.toggle_run)
                    return

                try:
                    model.eval()
                    load_checkpoint(checkpoint, model, optimizer)
                except Exception as e:
                    self.after(0, lambda: messagebox.showerror("Error", f"Failed to load checkpoint: {e}"))
                    self.after(0, self.toggle_run)
                    return

                items = self.tree.get_children()
                for item in items:
                    if not self.is_running: break
                    status = self.tree.item(item, "values")[2]

                    if status == "Pending":
                        input_path = self.tree.item(item, "values")[0]
                        self.after(0, self.update_status, item, "Processing...")

                        try:
                            while self.is_paused and self.is_running: time.sleep(0.5)
                            if not self.is_running: break

                            proc_start_time = time.time()
                            self.run_infer_logic(input_path, out_dir, start_sec, end_sec, item, proc_start_time)

                            total_time = int(time.time() - proc_start_time)
                            self.after(0, self.update_status, item, f"Done ({total_time}s)")
                        except Exception as e:
                            self.after(0, self.update_status, item, f"Error: {e}")

                if self.is_running:
                    self.after(0, self.toggle_run)
                    self.after(0, lambda: messagebox.showinfo("Queue Finished", "All files processed."))

            def quiet_add_new_track(self, org, voc, inst, start_sec, end_sec):
                self.add_track_to_data(org, voc, inst, start_sec, end_sec)
                self.refresh_track_listbox()

            def run_infer_logic(self, input_path, out_dir, start_sec, end_sec, item_id, proc_start_time):
                with torch.no_grad():
                    duration = 4
                    overlap = 2
                    frames = int(duration * 44100)
                    hop = int((duration - overlap) * 44100)

                    sample_rate, total_frames = audio_info(input_path)
                    if sample_rate != 44100: raise ValueError("Sample rate must be 44100Hz")

                    start_frame = int(start_sec * 44100) if start_sec > 0 else 0
                    end_frame = int(end_sec * 44100) if end_sec > 0 else total_frames
                    if end_frame > total_frames or end_frame <= start_frame: end_frame = total_frames
                    num_frames = end_frame - start_frame

                    vocal_waveform = torch.zeros((2, num_frames), dtype=torch.float32)
                    instr_waveform = torch.zeros((2, num_frames), dtype=torch.float32)
                    weights = torch.zeros((1, num_frames), dtype=torch.float32)
                    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
                    full_samples = audio_read(input_path, start_frame, end_frame)

                    for start in range(0, num_frames, hop):
                        if not self.is_running: return
                        while self.is_paused: time.sleep(0.5)

                        end = start + frames
                        samples = full_samples[:, start:end]
                        if samples.shape[1] < frames:
                            samples = nn.functional.pad(samples, (0, frames - samples.shape[1]))

                        full = torch.stft(samples, n_fft=n_fft, return_complex=True, window=window)
                        full_abs = full.abs()
                        phase = full / torch.clamp(full_abs, min=1e-24)

                        x = full_abs.unsqueeze(0).to(device)
                        y = model(x)[0].cpu()

                        vocal = phase * torch.clamp(y, min=0)
                        instr = phase * torch.clamp(full_abs - y, min=0)

                        vocal_reconstructed = torch.istft(vocal, n_fft=n_fft, window=window, center=True)
                        instr_reconstructed = torch.istft(instr, n_fft=n_fft, window=window, center=True)

                        size = min(vocal_reconstructed.shape[1], num_frames - start)

                        vocal_waveform[:, start:start + size] += vocal_reconstructed[:, 0:size]
                        instr_waveform[:, start:start + size] += instr_reconstructed[:, 0:size]
                        weights[:, start:start + size] += 1

                        pct_float = ((start + hop) / num_frames) * 100.0
                        pct = min(100, int(pct_float))
                        elapsed = time.time() - proc_start_time

                        eta = int((elapsed / pct_float) * (100.0 - pct_float)) if pct_float > 0 else 0
                        self.after(0, self.update_progress, item_id, pct, int(elapsed), eta)

                    vocal_waveform /= weights
                    instr_waveform /= weights

                    self.after(0, self.update_progress, item_id, 100, int(time.time() - proc_start_time), 0)

                    path_obj = Path(input_path)
                    os.makedirs(out_dir, exist_ok=True)

                    is_fragment = (start_sec > 0 or end_sec > 0)
                    frag_suffix = f"_{start_sec}_{end_sec}" if is_fragment else ""

                    out_voc = os.path.join(out_dir, path_obj.stem + f'_vocal_gui{frag_suffix}' + path_obj.suffix)
                    out_inst = os.path.join(out_dir, path_obj.stem + f'_instr_gui{frag_suffix}' + path_obj.suffix)

                    soundfile.write(out_voc, vocal_waveform.t().numpy(), 44100)
                    soundfile.write(out_inst, instr_waveform.t().numpy(), 44100)
                    
                    out_org = os.path.join(out_dir, path_obj.name)
                    if not os.path.exists(out_org) or not os.path.samefile(input_path, out_org):
                        try:
                            shutil.copy2(input_path, out_org)
                        except Exception as e:
                            print(f"Warning: could not copy original file to output dir: {e}")

                    self.after(0, self.quiet_add_new_track, out_org, out_voc, out_inst, start_sec, end_sec)

            # ================= PLAYER LOGIC  =================
            def toggle_audio_playback(self):
                self.focus_set()
                if not has_sd: return
                if self.is_playing:
                    self.stop_audio()
                else:
                    self.play_audio()

            def audio_callback(self, outdata, frames, time_info, status):
                if not self.is_playing:
                    outdata.fill(0)
                    return
                end_idx = self.play_idx + frames

                org_c = self.audio_org[self.play_idx: end_idx]
                voc_c = self.audio_voc[self.play_idx: end_idx]
                inst_c = self.audio_inst[self.play_idx: end_idx]

                actual_len = max(len(org_c), len(voc_c), len(inst_c))

                if actual_len == 0:
                    outdata.fill(0)
                    raise sd.CallbackStop()

                mix = np.zeros(actual_len, dtype=np.float32)
                if len(org_c) == actual_len: mix += org_c * self.vol_org
                if len(voc_c) == actual_len: mix += voc_c * self.vol_voc
                if len(inst_c) == actual_len: mix += inst_c * self.vol_inst

                outdata[:actual_len, 0] = mix
                if actual_len < frames: outdata[actual_len:, 0] = 0

                self.play_idx += actual_len
                self.cursor_time = self.play_idx / 44100.0
                if actual_len < frames: raise sd.CallbackStop()

            def play_audio(self):
                if not self.current_track or self.get_max_audio_duration() <= 0: return
                self.stop_audio()
                self.play_idx = int(self.cursor_time * 44100)
                self.is_playing = True
                self.btn_play.config(text="⏹ Stop (Space)")
                try:
                    self.stream = sd.OutputStream(samplerate=44100, channels=1, callback=self.audio_callback,
                                                  blocksize=2048)
                    self.stream.start()
                except Exception as e:
                    print("Play error:", e)
                    self.stop_audio()

            def stop_audio(self):
                self.is_playing = False
                if self.stream is not None:
                    self.stream.stop()
                    self.stream.close()
                    self.stream = None
                self.btn_play.config(text="▶ Play (Space)")

            def update_playback_cursor(self):
                if self.is_playing:
                    try:
                        window_len = float(self.window_var.get())
                    except ValueError:
                        window_len = 15.0

                    if self.cursor_time > self.view_start + window_len * 0.95:
                        self.view_start = self.cursor_time
                        self.update_scrollbar()
                        self.render_plot()
                    elif self.cursor_time < self.view_start:
                        self.view_start = max(0.0, self.cursor_time - window_len * 0.1)
                        self.update_scrollbar()
                        self.render_plot()

                    self.update_cursor_position()
                    self.time_label.config(text=f"Time: {self.cursor_time:.1f}s")
                    if self.stream and not self.stream.active:
                        self.stop_audio()

                self.after(20, self.update_playback_cursor)

            def on_canvas_click(self, event):
                self.focus_set()
                if not self.full_pil_images: return
                w = self.canvas.winfo_width()
                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    window_len = 15.0

                click_time = self.view_start + (event.x / w) * window_len
                total_dur = self.get_max_audio_duration()

                self.cursor_time = max(0.0, min(click_time, total_dur))
                self.play_idx = int(self.cursor_time * 44100)
                self.time_label.config(text=f"Time: {self.cursor_time:.1f}s")
                self.update_cursor_position()

                if self.is_playing: self.play_audio()

            def on_canvas_pan_start(self, event):
                self.focus_set()
                self.drag_start_x = event.x
                self.drag_start_view = self.view_start

            def on_canvas_pan_drag(self, event):
                if not hasattr(self, 'drag_start_x'): return
                w = self.canvas.winfo_width()
                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    window_len = 15.0

                dx_sec = ((event.x - self.drag_start_x) / w) * window_len
                self.view_start = self.drag_start_view - dx_sec

                total_dur = self.get_max_audio_duration()
                if self.view_start > total_dur - window_len:
                    self.view_start = max(0.0, total_dur - window_len)
                if self.view_start < 0: self.view_start = 0.0

                self.update_scrollbar()
                self.render_plot()

            def on_canvas_scroll(self, event):
                self.focus_set()
                if self.get_max_audio_duration() == 0: return
                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    window_len = 15.0

                if (event.state & 0x0004):
                    zoom_factor = 0.8 if event.delta > 0 else 1.25
                    new_window = max(0.5, min(3600.0, window_len * zoom_factor))

                    w = self.canvas.winfo_width()
                    mouse_time = self.view_start + (event.x / w) * window_len
                    self.view_start = mouse_time - (event.x / w) * new_window

                    total_dur = self.get_max_audio_duration()
                    if self.view_start > total_dur - new_window:
                        self.view_start = max(0.0, total_dur - new_window)
                    if self.view_start < 0: self.view_start = 0.0

                    self.window_var.set(f"{new_window:.1f}")
                else:
                    shift = window_len * 0.1
                    if event.delta > 0:
                        self.view_start -= shift
                    else:
                        self.view_start += shift

                    total_dur = self.get_max_audio_duration()
                    if self.view_start > total_dur - window_len:
                        self.view_start = max(0.0, total_dur - window_len)
                    if self.view_start < 0: self.view_start = 0.0

                    self.update_scrollbar()
                    self.render_plot()

            def on_scroll(self, *args):
                if self.get_max_audio_duration() == 0: return
                total_duration = self.get_max_audio_duration()
                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    window_len = 15.0

                if args[0] == 'moveto':
                    self.view_start = float(args[1]) * total_duration
                elif args[0] == 'scroll':
                    amount = int(args[1])
                    if args[2] == 'pages':
                        self.view_start += amount * window_len * 0.9
                    else:
                        self.view_start += amount * window_len * 0.1

                if self.view_start > total_duration - window_len:
                    self.view_start = max(0.0, total_duration - window_len)
                if self.view_start < 0: self.view_start = 0.0

                self.update_scrollbar()
                self.render_plot()

            def update_scrollbar(self):
                total_duration = self.get_max_audio_duration()
                if total_duration <= 0: return
                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    return

                first = self.view_start / total_duration
                last = (self.view_start + window_len) / total_duration

                if last > 1.0: last = 1.0
                if first < 0.0: first = 0.0
                self.scrollbar.set(first, last)

            # ================= Spectogram rendering =================
            def load_track_into_memory(self):
                t = self.current_track
                sr = 44100

                org = np.zeros(0, dtype=np.float32)
                voc = np.zeros(0, dtype=np.float32)
                inst = np.zeros(0, dtype=np.float32)

                if t.get('original') and os.path.exists(t['original']):
                    try:
                        o, _ = soundfile.read(t['original'], always_2d=True)
                        org = o[:, 0].astype(np.float32)
                    except Exception as e:
                        print("Err org:", e)

                if t.get('vocal') and os.path.exists(t['vocal']):
                    try:
                        v, _ = soundfile.read(t['vocal'], always_2d=True)
                        voc = v[:, 0].astype(np.float32)
                    except Exception as e:
                        print("Err voc:", e)

                if t.get('instr') and os.path.exists(t['instr']):
                    try:
                        i, _ = soundfile.read(t['instr'], always_2d=True)
                        inst = i[:, 0].astype(np.float32)
                    except Exception as e:
                        print("Err inst:", e)

                start_idx = int(t.get('frag_start', 0.0) * sr)

                max_len = max(len(org), start_idx + len(voc), start_idx + len(inst))

                self.audio_org = np.zeros(max_len, dtype=np.float32)
                self.audio_voc = np.zeros(max_len, dtype=np.float32)
                self.audio_inst = np.zeros(max_len, dtype=np.float32)

                if len(org) > 0: self.audio_org[:len(org)] = org
                if len(voc) > 0: self.audio_voc[start_idx:start_idx + len(voc)] = voc
                if len(inst) > 0: self.audio_inst[start_idx:start_idx + len(inst)] = inst

            def start_compute_thread(self):
                self.canvas.delete("all")
                colors = self.get_theme_colors()
                self.canvas.create_text(50, 50, text="Loading & computing full spectrograms... Please wait.",
                                        fill=colors["ACCENT_CYAN"], font=("Segoe UI", 12, "bold"), anchor=tk.NW,
                                        tags="loading_text")
                self.loading_spec = True

                try:
                    n_fft_user = int(self.fft_var.get())
                except ValueError:
                    n_fft_user = 2048

                threading.Thread(target=self.compute_spectrograms_thread, args=(n_fft_user,), daemon=True).start()

            def on_track_select(self, event):
                self.focus_set()
                selection = self.track_listbox.curselection()
                if not selection: return
                self.stop_audio()

                name = self.track_listbox.get(selection[0])
                self.current_track = self.tracks_data[name]

                self.update_file_status_ui()
                self.load_track_into_memory()
                self.view_start = 0.0

                if self.current_track['frag_start'] > 0:
                    self.view_start = self.current_track['frag_start']
                    self.cursor_time = self.current_track['frag_start']
                else:
                    self.cursor_time = 0.0

                self.update_scrollbar()
                self.start_compute_thread()

            def compute_spectrograms_thread(self, n_fft_user):
                if self.get_max_audio_duration() == 0:
                    self.loading_spec = False
                    self.after(0, self.render_plot)
                    return

                sr = 44100
                hop_len = 1024
                if self.get_max_audio_duration() > 600:
                    hop_len = 4096

                n_mels = 128
                new_pils = []
                tracks = [self.audio_org, self.audio_voc, self.audio_inst]

                for y in tracks:
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft_user, hop_length=hop_len, n_mels=n_mels)
                    S_dB = librosa.power_to_db(S, ref=1.0)

                    S_dB = np.clip(S_dB, -80, 0)
                    normalized = (S_dB + 80) / 80.0

                    rgba = cm.magma(normalized)
                    img_arr = (rgba * 255).astype(np.uint8)

                    img = Image.fromarray(img_arr, 'RGBA')
                    img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    new_pils.append(img)

                self.full_pil_images = new_pils
                self.loading_spec = False
                self.after(0, self.render_plot)

            def delayed_render(self):
                if self.render_job is not None:
                    self.after_cancel(self.render_job)
                self.render_job = self.after(100, self.render_plot)

            def render_plot(self):
                self.canvas.delete("loading_text")
                if self.loading_spec or not self.full_pil_images: return

                w = self.canvas.winfo_width()
                h = self.canvas.winfo_height()
                if w < 10 or h < 10: return

                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    window_len = 15.0

                if self.get_max_audio_duration() > 600:
                    time_per_px = 4096 / 44100.0
                else:
                    time_per_px = 1024 / 44100.0

                px_start = int(self.view_start / time_per_px)
                px_end = int((self.view_start + window_len) / time_per_px)

                max_px = self.full_pil_images[0].width
                px_start = max(0, min(px_start, max_px - 1))
                px_end = max(1, min(px_end, max_px))

                actual_time_len = (px_end - px_start) * time_per_px
                if actual_time_len <= 0: return

                draw_w = int((actual_time_len / window_len) * w)
                if draw_w < 1: draw_w = 1

                sec_h = h // 3
                self.tk_images = []
                self.canvas.delete("spec_image")
                self.canvas.delete("spec_label")

                colors = self.get_theme_colors()

                for i, img in enumerate(self.full_pil_images):
                    crop = img.crop((px_start, 0, px_end, img.height))
                    resized = crop.resize((draw_w, sec_h), Image.NEAREST)
                    tk_img = ImageTk.PhotoImage(resized)
                    self.tk_images.append(tk_img)

                    y_offset = i * sec_h
                    self.canvas.create_image(0, y_offset, anchor=tk.NW, image=tk_img, tags="spec_image")

                    titles = ["Original", "Vocals", "Instrumental"]
                    self.canvas.create_text(10, y_offset + 5, text=titles[i], fill=colors["ACCENT_CYAN"],
                                            font=('Segoe UI', 10, 'bold'), anchor=tk.NW, tags="spec_label")

                    if i > 0:
                        self.canvas.create_line(0, y_offset, w, y_offset, fill=colors["ACCENT_PINK"], width=2, tags="spec_image")

                self.canvas.tag_lower("spec_image")
                self.update_cursor_position()

            def update_cursor_position(self):
                w = self.canvas.winfo_width()
                h = self.canvas.winfo_height()
                try:
                    window_len = float(self.window_var.get())
                except ValueError:
                    window_len = 15.0

                x = ((self.cursor_time - self.view_start) / window_len) * w

                colors = self.get_theme_colors()

                if not hasattr(self, 'cursor_line') or not self.canvas.find_withtag("cursor"):
                    self.cursor_line = self.canvas.create_line(x, 0, x, h, fill=colors["CURSOR_COLOR"], width=2, tags="cursor")
                else:
                    self.canvas.coords(self.cursor_line, x, 0, x, h)
                    self.canvas.itemconfig(self.cursor_line, fill=colors["CURSOR_COLOR"])
                self.canvas.tag_raise("cursor")


            def browse_stft_input(self):
                self.focus_set()
                f = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav"), ("All Files", "*.*")])
                if f:
                    self.stft_in_var.set(f)
                    # auto‑suggest output name
                    base, ext = os.path.splitext(f)
                    self.stft_out_var.set(base + "_center_removed.wav")

            def browse_stft_output(self):
                self.focus_set()
                f = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
                if f:
                    self.stft_out_var.set(f)

            def run_stft_separation(self):
                if self.stft_sep_running:
                    messagebox.showinfo("Busy", "A separation is already running.")
                    return
                in_path = self.stft_in_var.get().strip()
                out_path = self.stft_out_var.get().strip()
                if not in_path or not os.path.exists(in_path):
                    messagebox.showerror("Error", "Select a valid input WAV file.")
                    return
                if not out_path:
                    messagebox.showerror("Error", "Specify an output file path.")
                    return

                self.stft_sep_running = True
                self.stft_run_btn.config(state=tk.DISABLED)
                colors = self.get_theme_colors()
                self.stft_status.config(text="Processing...", foreground=colors["ACCENT_PINK"])

                def task():
                    try:
                        data, fs = soundfile.read(in_path, always_2d=True, dtype=np.float32)
                        if data.shape[1] < 2:
                            raise ValueError("Input must be a stereo file.")
                        self._stft_center_remove(data, fs, out_path)
                        self.after(0, lambda: self.stft_status.config(text="Done.", foreground=colors["ACCENT_CYAN"]))
                    except Exception as e:
                        self.after(0, lambda: self.stft_status.config(text=f"Error: {e}", foreground="#ff4444"))
                    finally:
                        self.after(0, lambda: self.stft_run_btn.config(state=tk.NORMAL))
                        self.stft_sep_running = False

                self.stft_sep_thread = threading.Thread(target=task, daemon=True)
                self.stft_sep_thread.start()

            def _stft_center_remove(self, data, fs, out_path):
                """Exact implementation of the provided STFT center removal."""
                FRAME_SIZE = 2048
                HOP_SIZE = FRAME_SIZE // 2  # 50% overlap

                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.max(np.abs(data))
                L = data[:, 0].copy()
                R = data[:, 1].copy()
                N = len(L)

                window = np.hanning(FRAME_SIZE)

                pad = FRAME_SIZE
                L = np.pad(L, (0, pad), mode='constant')
                R = np.pad(R, (0, pad), mode='constant')

                L_out = np.zeros_like(L)
                R_out = np.zeros_like(R)

                freqs = np.fft.rfftfreq(FRAME_SIZE, d=1 / fs)
                low_mask = freqs <= 200
                mid_mask = (freqs > 200) & (freqs <= 4000)
                high_mask = freqs > 4000

                total_frames = (len(L) - FRAME_SIZE) // HOP_SIZE + 1
                frame_count = 0

                colors = self.get_theme_colors()

                for i in range(0, len(L) - FRAME_SIZE + 1, HOP_SIZE):
                    frame_L = L[i:i + FRAME_SIZE] * window
                    frame_R = R[i:i + FRAME_SIZE] * window

                    L_fft = np.fft.rfft(frame_L)
                    R_fft = np.fft.rfft(frame_R)

                    L_low = L_fft * low_mask
                    L_mid = L_fft * mid_mask
                    L_high = L_fft * high_mask
                    R_low = R_fft * low_mask
                    R_mid = R_fft * mid_mask
                    R_high = R_fft * high_mask

                    mid_side = L_mid - R_mid
                    L_mid_new = mid_side
                    R_mid_new = -mid_side

                    L_fft_out = L_low + L_mid_new + L_high
                    R_fft_out = R_low + R_mid_new + R_high

                    frame_L_out = np.fft.irfft(L_fft_out)
                    frame_R_out = np.fft.irfft(R_fft_out)

                    L_out[i:i + FRAME_SIZE] += frame_L_out * window
                    R_out[i:i + FRAME_SIZE] += frame_R_out * window

                    frame_count += 1
                    # Update progress every 100 frames
                    if frame_count % 100 == 0:
                        pct = int((i / len(L)) * 100)
                        self.after(0, self.stft_status.config,
                                   f"Processing... {pct}%", colors["ACCENT_PINK"])

                # Remove padding
                L_out = L_out[:N]
                R_out = R_out[:N]

                # Normalize
                max_val = max(np.max(np.abs(L_out)), np.max(np.abs(R_out)))
                if max_val > 0:
                    L_out /= max_val
                    R_out /= max_val

                out = np.stack((L_out, R_out), axis=1)

                # Write 16-bit WAV (as original)
                soundfile.write(out_path, (out * 32767).astype(np.int16), fs, subtype='PCM_16')


        app = AudioSeparationGUI()
        app.mainloop()

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

            waveform[:, start:start + fragment.shape[1]] += fragment[:]
            weights[:, start:start + fragment.shape[1]] += 1

        waveform /= weights
        soundfile.write('load_test.wav', waveform.t().numpy(), 44100)

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
       
        loss_fn = nn.L1Loss() # PerceptualLoss(sample_rate=44100, frequency_bin_count=frequency_bin_count).to(device)
        
        
        start_epoch = load_latest_checkpoint(checkpoint_dir, model, optimizer) + 1
        end_epoch = 400
        
        start_time = time.perf_counter()
        last_train_loss = 0
        last_test_loss = 0
        
        sum_epoch_duration = 0
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=4, factor=0.9)
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


            for start in range(0, num_frames, hop):
                end = start + frames
                
                print(f'{start:10d}/{num_frames}')
                
                samples = full_samples[:, start:end]
                samples = nn.functional.pad(samples, (0, frames - samples.shape[1]))

                full = torch.stft(samples, n_fft=n_fft, return_complex=True, window=window)
                full_abs = full.abs()
                
                phase = full / torch.clamp(full_abs, min=1e-24)
                
                x = full_abs.unsqueeze(0).to(device)
                y = model(x)[0].cpu()
           
                vocal = phase * torch.clamp(y, min=0)
                instr = phase * torch.clamp(full_abs - y, min=0)
                
                vocal_reconstructed = torch.istft(vocal, n_fft=n_fft, window=window, center=True)
                instr_reconstructed = torch.istft(instr, n_fft=n_fft, window=window, center=True)
                
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
        print(f'  gui       | Runs the graphical user interface for inference and spectrogram viewing')
        print(f'  info      | Prints information about current model')
        print(f'  prepare   | Prepares the train and test dataset indexes from ./musdb/train and ./musdb/test directories')
        print(f'  load_test | Reconstructs first track as specified in train dataset index to load_test.wav' )
        print(f'  train     | Starts or continues training using files as specified by dataset index')
        print(f'  infer checkpoint_path input_path output_path | Infers the instrumental and vocal stems of specified track using a given checkpoint')
        sys.exit(1)
