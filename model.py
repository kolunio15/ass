import sys
import os
import time
import data
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

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
    checkpoint = torch.load(path)
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
                optimizer.step()
                   
                total_loss += loss.item()
                    
                print(f'epoch {epoch:3d}/{end_epoch-1} batch: {batch:4d} train')
                
                break
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
                        
                    print(f'epoch {epoch:3d}/{end_epoch-1} batch: {batch:4d} test')
                    break
          
          
            return total_loss / batch_count
        
        n_fft = 1024
        frequency_bin_count = n_fft // 2 + 1
        
        batch_size = 64
        train_dataset = torch.utils.data.DataLoader(MUSDB18Dataset(train_path, n_fft, True), batch_size=batch_size, pin_memory=True)
        test_dataset = torch.utils.data.DataLoader(MUSDB18Dataset(test_path, n_fft, True), batch_size=batch_size, pin_memory=True)            
       
       
        device = torch.device('cuda:0')
        model = NaiveLSTM(frequency_bin_count=frequency_bin_count, hidden_layers=128).to(device)
        loss_fn = PerceptualLoss(sample_rate=44100, frequency_bin_count=frequency_bin_count).to(device)
        optimizer = torch.optim.AdamW(model.parameters())
        
        checkpoint_dir = './checkpoint'
        log_path = './training.csv'
        
        start_epoch = load_latest_checkpoint(checkpoint_dir, model, optimizer) + 1
        end_epoch = 100
        
        start_time = time.perf_counter()
        last_train_loss = 0
        last_test_loss = 0
        
        sum_epoch_duration = 0
        
        with open(log_path, 'a') as log:
            for epoch in range(start_epoch, end_epoch):
                train_start_time = time.perf_counter()
                train_loss = train()
                train_end_time = time.perf_counter()
                
                save_checkpoint(os.path.join(checkpoint_dir, 'latest.zip'), epoch, model, optimizer)
                if epoch != 0 and epoch % 10 == 0:
                    save_checkpoint(os.path.join(checkpoint_dir, f'{epoch}.zip'), epoch, model, optimizer)
                
                test_start_time = time.perf_counter()
                test_loss = test()
                test_end_time = time.perf_counter()
                
                train_delta = train_loss - last_train_loss if epoch != start_epoch else 0
                test_delta  = test_loss - last_test_loss   if epoch != start_epoch else 0
            
                epoch_duration = test_end_time  - train_start_time
                train_duration = train_end_time - train_start_time
                test_duration  = test_end_time  - test_start_time
                elapsed        = test_end_time  - start_time
                
                sum_epoch_duration += epoch_duration
                avg_epoch_duration = sum_epoch_duration / (epoch - start_epoch + 1)
                remaining = (end_epoch - epoch - 1) * avg_epoch_duration
                
                print(f'epoch: {epoch:3d}/{end_epoch-1} train: {train_loss:.5e} test: {test_loss:.5e} Δtrain: {train_delta:.5e} Δtest: {test_loss:.5e} train duration: {train_duration / 60:2.1f}m test duration: {test_duration / 60:2.1f}m  epoch duration: {epoch_duration/60:2.1f}m elapsed: {elapsed/60/60:2.1f}h eta: {remaining/60/60:2.1f}h')
                print(f'{epoch};{train_loss};{test_loss};{train_duration};{test_duration};{epoch_duration}', file=log)
                log.flush()
    case [_, 'infer', checkpoint, input_path, output_path]:
        
    case _:
        print(f'unknown command')
        print(f'usage:')
        print(f'  prepare   | Prepares the train and test dataset indexes from ./musdb/train and ./musdb/test directories')
        print(f'  load_test | Reconstructs first track as specified in train dataset index to load_test.wav' )
        sys.exit(1)
        