import os
import struct
import scipy
import numpy as np
import scipy.io
from scipy.signal import ShortTimeFFT 
import scipy.signal.windows as windows

def load_wav(path):
    (sample_rate, samples) = scipy.io.wavfile.read(path)
    if samples.dtype == np.int16:
        samples = samples.astype(float) / (2 ** 15 - 1)
    elif samples.dtype == np.int32:
        samples = samples.astype(float) / (2 ** 31 - 1)
    elif samples.dtype == np.float16 or samples.dtype == np.float32:
        pass
    else:
        raise Exception('Unexpected format:', samples.dtype)

    return (sample_rate, samples)
def save_wav(path, sample_rate, l, r):
    samples = np.column_stack([l, r])
    print(samples.shape)
    scipy.io.wavfile.write(path, sample_rate, samples)

def create_stft(sample_rate, frequency_bin_count):
    gaussian_standard_deviation = sample_rate * 0.5 # in samples

    stft_window_length_samples = frequency_bin_count * 2 - 1
    stft_window = windows.gaussian(stft_window_length_samples, std=gaussian_standard_deviation)
    stft_hop_seconds = 0.025
    stft_hop_samples = int(stft_hop_seconds * sample_rate)
    return ShortTimeFFT(win=stft_window, hop=stft_hop_samples, fs=sample_rate, fft_mode='onesided')

def write_i64(w, value):
    s = struct.pack('q', value)
    w.write(s)
def write_f32(w, value):
    s = struct.pack('f', value)
    w.write(s)
def write_c32(w, value):
    write_f32(w, (value).real)
    write_f32(w, (value).imag)
def write_f32_array(w, values):
    s = values.astype('f')
    w.write(s)


def read_i64(r):
    data = r.read(8)
    return struct.unpack_from('q', data)[0]
def read_f32(r):
    data = r.read(4)
    return struct.unpack_from('f', data)[0]
    
def read_f32_array(r, size):
    data = r.read(4 * size)
    return struct.unpack_from(f'{size}f', data)


def convert_musdb_to_dataset_file(input_dir, output_file, frequency_bins_count):
    stft = create_stft(sample_rate=44100, frequency_bin_count=frequency_bins_count)

    def file_stft(file):
        sample_rate, full_channels = load_wav(file)
        assert sample_rate == 44100
        print('og_shape:', full_channels.shape)
        return stft.stft(full_channels[:, 0])
        
    with open(output_file, 'wb') as w:
        write_i64(w, 1) # version
        write_i64(w, 0) # size in time slices
        write_i64(w, frequency_bins_count)

        time_slices = 0
        for base_path in os.scandir(input_dir):            
           
            full   = file_stft(os.path.join(base_path, 'mixture.wav'))
            vocals = file_stft(os.path.join(base_path, 'vocals.wav'))
            
            full_max = np.max(np.abs(full))

            #full /= full_max
            #vocals /= full_max
            full   = np.log2(full   / full_max + 1e-8)
            vocals = np.log2(vocals / full_max + 1e-8) 

            time_slices += full.shape[1]
            out = np.empty(shape=(frequency_bins_count * 2), dtype=np.float32)
            
            for t_idx in range(full.shape[1]):
                out[0::2] = full[:, t_idx].real
                out[1::2] = full[:, t_idx].imag
                write_f32_array(w, out)
                
                out[0::2] = vocals[:, t_idx].real
                out[1::2] = vocals[:, t_idx].imag
                write_f32_array(w, out)

                
                #for v in full[:, t_idx]:
                #    write_c32(w, v)
                #for v in vocals[:, t_idx]:
                #    write_c32(w, v)
                if t_idx % 1000 == 0: print(f'{t_idx}/{full.shape[1]} of file: {base_path.path}')
        w.seek(8)
        print('time_slices:', time_slices)
        write_i64(w, time_slices)


if __name__ == '__main__':    
    convert_musdb_to_dataset_file('./musdb/train/', 'train_dataset.bin', frequency_bins_count=2048)
    convert_musdb_to_dataset_file('./musdb/test/', 'test_dataset.bin',   frequency_bins_count=2048)