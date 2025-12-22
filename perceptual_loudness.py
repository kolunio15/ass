import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


freqs = np.array([
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 
    10000, 12500
])

# spl_40 = np.array([
#     93.2, 88.5, 83.7, 78.9, 74.3, 69.8, 65.4, 61.3, 57.4, 53.7, 50.4, 
#     47.4, 44.7, 42.4, 40.6, 39.2, 38.4, 40.0, 41.6, 43.1, 44.6, 44.9, 
#     43.4, 40.6, 41.2, 47.1, 53.6, 58.2, 58.9
# ])

spl_40 = np.array([99.7, 93.9, 88.2, 82.7, 77.8, 73.0, 68.3, 64.2, 60.4, 56.6, 53.3, 50.3, 47.6, 45.1, 43.1, 41.4, 40.0, 40.0, 41.8, 42.6, 39.2, 36.6, 35.5, 36.7, 40.1, 45.8, 51.6, 54.4, 51.3])

log_freqs = np.log10(freqs)
cs = CubicSpline(log_freqs, spl_40, bc_type='natural')


sample_rate = 44100
frequency_bin_count = 2048

frequencies = np.linspace(0, sample_rate / 2, frequency_bin_count)
clamped_frequencies = np.minimum(np.maximum(frequencies, freqs[0]), freqs[-1])

spl = cs(np.log10(clamped_frequencies))

plt.figure(figsize=(10, 6))
plt.plot(freqs, spl_40, 'ro', label='ISO 226', markersize=5)
plt.plot(frequencies, spl, 'b-', label='Cubic Spline', linewidth=2)

plt.xlabel('[Hz]')
plt.ylabel('sound pressure level [dB]')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig('iso226_linear_axis.png')


plt.figure(figsize=(10, 5))
plt.semilogx(freqs, spl_40, 'ro', label='ISO 226 Points', markersize=6)
plt.semilogx(frequencies[1:], spl[1:], 'b-', label='Cubic Spline (Log-Freq)', linewidth=2)
plt.xlabel('[Hz]')
plt.ylabel('sound pressure level [dB]')
plt.grid(True, which="both", ls="-", alpha=0.5)

ticks = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
plt.xticks(ticks, [str(t) for t in ticks])
plt.savefig('iso226_log_axis.png')