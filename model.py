import sys
import os
import time
import data
import torch
import torch.nn as nn
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


class AmplitudePredict(nn.Module):
    def __init__(self, frequency_bin_count, hidden_layers):
        super(AmplitudePredict, self).__init__()
        self.lstm = nn.LSTM(frequency_bin_count, hidden_layers, 2, batch_first=True, bidirectional=False, dropout=0.2)
        self.linear = nn.Linear(in_features=hidden_layers, out_features=frequency_bin_count)
        self.relu = nn.ReLU()

    def forward(self, x : torch.Tensor, state):
        y, (hs, cs) = self.lstm(x, state)
        y = self.linear(y)
        y = self.relu(y)
                
        return y, (hs.detach(), cs.detach())
    

checkpoint_path = './checkpoint_100_tracks_1024_timestep_log.bin'
    
frequency_bin_count = 2048
timesteps = 1024
device    = torch.device('cuda:0')
model     = AmplitudePredict(frequency_bin_count=frequency_bin_count, hidden_layers=512).to(device)
criterion = nn.HuberLoss()
optimizer = torch.optim.Adam(model.parameters())


epoch_start, epoch_end = 0, 100
if (os.path.exists(checkpoint_path)):
    epoch_start = load_checkpoint(checkpoint_path, model, optimizer) + 1


def train():
    model.train()
    
    total_loss = 0
    chunk_count = 0

    last_file = 0
    state = None
    for file, chunk, x, y_correct in dataset('./train_dataset.bin'):
        if last_file != file: state = None
        last_file = file
        chunk_count += 1
        
        optimizer.zero_grad()
        
        y_predicted, state = model(x, state)
        
        loss = criterion(y_predicted, y_correct)
        loss.backward()
        optimizer.step()
           
        total_loss += loss.item()
            
        print(f'epoch {epoch:3d}/{epoch_end-1} file: {file:3d} chunk: {chunk:2d} train')
    return total_loss / chunk_count
    
def test():
    model.eval()
    total_loss = 0
    chunk_count = 0
    
    last_file = 0
    state = None
    
    with torch.no_grad():
        for file, chunk, x, y_correct in dataset('./test_dataset.bin'):
            if last_file != file: state = None
            last_file = file            
            chunk_count += 1
            
            y_predicted, state = model(x, state)
            
            loss = criterion(y_predicted, y_correct)   
            total_loss += loss.item()
                
            print(f'epoch {epoch:3d}/{epoch_end-1} file: {file:3d} chunk: {chunk:2d} test')
  
    return total_loss / chunk_count
        
def prepare(input_path, output_path):
    stft = data.create_stft(44100, frequency_bin_count)
    def file_stft(file):
        sample_rate, full_channels = data.load_wav(file)
        assert sample_rate == 44100
        return stft.stft(full_channels[:, 0])
    
    
    with open(output_path, 'wb') as w:
        tracks = os.listdir(input_path)

        data.write_i64(w, 2) # version
        data.write_i64(w, frequency_bin_count) # frequency_bin_count
        data.write_i64(w, len(tracks)) # track_count
        
        for i, track_dir_path in enumerate(tracks):
            print(f'file: [{i:3d}/{len(tracks)-1}] path:{track_dir_path}')
            full   = file_stft(os.path.join(input_path, track_dir_path, 'mixture.wav'))
            vocals = file_stft(os.path.join(input_path, track_dir_path, 'vocals.wav'))
            
            full_abs = np.abs(full)
            vocals_abs = np.abs(vocals)
            
            x = full_abs / np.max(full_abs)
            y = vocals_abs / np.max(vocals_abs)
                        
            data.write_i64(w, x.shape[1]) # track time length
           
            for t_idx in range(x.shape[1]):
                data.write_f32_array(w, x[:, t_idx])
                data.write_f32_array(w, y[:, t_idx])
                
def dataset(input_path):
    with open(input_path, 'rb') as r:
        assert data.read_i64(r) == 2 # version
        assert data.read_i64(r) == frequency_bin_count
        track_count = data.read_i64(r)
        
        for track in range(track_count):
            time_length = data.read_i64(r)
            assert time_length > 0
            
            for chunk, t_start in enumerate(range(0, time_length, timesteps)):
                t_end = min(t_start + timesteps, time_length)
                
                x = torch.empty((1, t_end - t_start, frequency_bin_count), dtype=torch.float32, device=device)
                y = torch.empty((1, t_end - t_start, frequency_bin_count), dtype=torch.float32, device=device)
                
               

                
                raw = torch.tensor(data.read_f32_array(r, frequency_bin_count * 2 * (t_end-t_start)))
                
                raw = torch.reshape(raw, shape=((t_end-t_start)*2, frequency_bin_count))
         
                raw = torch.log2(raw + 1)
         
                x[0, 0:t_end-t_start] = raw[0:2*(t_end-t_start):2]
                y[0, 0:t_end-t_start] = raw[1:2*(t_end-t_start):2]
                
                                
                
                # for t_idx in range(0, t_end-t_start):
                #     x[0, t_idx] = torch.tensor(data.read_f32_array(r, frequency_bin_count), device=device)
                #     y[0, t_idx] = torch.tensor(data.read_f32_array(r, frequency_bin_count), device=device)
                    
                yield track, chunk, x, y
                
match sys.argv:
    case [_, 'prepare']:
        prepare('./musdb/train', './train_dataset.bin')
        prepare('./musdb/test', './test_dataset.bin')
    case [_, 'infer', timesteps, checkpoint, input_path, output_path]:
        load_checkpoint(checkpoint, model, optimizer)
        timesteps = int(timesteps)
        model.eval()
        with torch.no_grad():
            stft = data.create_stft(44100, frequency_bin_count)
            sample_rate, full_channels = data.load_wav(input_path)
            assert sample_rate == 44100
            full = stft.stft(full_channels[:, 0])
            
            full_abs = np.abs(full)
            full_abs_max = np.max(full_abs)
            full_abs_normalised = full_abs / full_abs_max
            full_abs_normalised_log = np.log2(full_abs_normalised + 1)
        
            dest = np.zeros_like(full)
            
            state = None
            for t_start in range(0, full.shape[1], timesteps):
                t_end = min(t_start + timesteps, full.shape[1])
                
                x = np.empty((1, t_end-t_start, frequency_bin_count), dtype=np.float32)
                for i, t_idx in enumerate(range(t_start, t_end)):
                    x[0, i] = full_abs_normalised_log[:, t_idx]
                y, state = model(torch.from_numpy(x).to(device), state)
                y = y.cpu()
                
                for i, t_idx in enumerate(range(t_start, t_end)): 
                    dest[:, t_idx] = y[0, i]
                print(t_start)
                    
            full_angle = full / np.maximum(full_abs, 1e-30)
                    
            vocals = full_angle * full_abs_max * np.minimum(np.exp2(dest) - 1, 1.0)

            instrumental = full_angle * full_abs_max * np.minimum(1, np.maximum(0, (full_abs_normalised - np.minimum(np.exp2(dest) - 1, 1.0))))
            #instrumental = full - vocals
            
            path = Path(output_path)
            vocal_path =  path.stem + '_vocal' + path.suffix
            instrumental_path =  path.stem + '_instr' + path.suffix
            
            
            output_vocals = stft.istft(vocals, k1=full_channels[:, 0].shape[0])
            output_instrumental = stft.istft(instrumental, k1=full_channels[:, 0].shape[0])
            data.save_wav(vocal_path, 44100, output_vocals, output_vocals)
            data.save_wav(instrumental_path, 44100, output_instrumental, output_instrumental)
            
    case [_, 'train']:
        with open('training_log.csv', 'a') as log:
            time_start = time.perf_counter()
            sum_epoch_duration_s = 0
            
            last_train_loss = 1
            last_test_loss  = 1
            for epoch in range(epoch_start, epoch_end):
                epoch_start_s = time.perf_counter()
                
                train_loss = train()
                
                save_checkpoint(checkpoint_path, epoch, model, optimizer)
                test_loss = test()
                
                epoch_end_s = time.perf_counter()
                epoch_duration_s = epoch_end_s - epoch_start_s
                sum_epoch_duration_s += epoch_duration_s   
                elapsed = epoch_end_s - time_start
                remaining = (epoch_end - epoch - 1) * (sum_epoch_duration_s / (epoch - epoch_start + 1))
                
                print(f'epoch: {epoch:3d}/{epoch_end-1} train: {train_loss:.5e} test: {test_loss:.5e} Δtrain: {train_loss - last_train_loss:.5e} Δtest: {test_loss - last_test_loss:.5e} epoch duration: {epoch_duration_s/60:3.1f}m elapsed: {elapsed/60:3.1f}m eta: {remaining/60/60:3.1f}h')
                last_train_loss, last_test_loss = train_loss, test_loss   
                
                
                print(f'{epoch};{train_loss:.8e};{test_loss:.8e};{epoch_duration_s:.2f}', file=log)
                log.flush()
    case _:
        print('bad arguments')
        print(sys.argv[0], 'prepare|infer|train')
print('done')