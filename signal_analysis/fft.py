# -*- coding: utf-8 -*-
#%%
import numpy as np
import tensorflow as tf
import torch

import matplotlib.pyplot as plt
#%%
# 1. Create random signal

time = np.arange(0,1000, 0.001)
amplitude = np.sin(time)

print(amplitude.shape)
#%%
# 2. plot signal
plt.plot(time, amplitude)
plt.title("signal")
plt.xlabel("time")
plt.ylabel("Amplitude")
plt.axhline(y=0, color='k')
plt.show()
#%%
# 3. fft in numpy
spectrogram = None
n_fft = 1000 # sampling rate (max frequency)
for i in range(int(len(amplitude)/n_fft)):
    if i == 0:
        spectrogram = np.expand_dims(np.fft.fft(amplitude[n_fft*i:n_fft*(i+1)]), axis=0)
    else:
        temp = np.expand_dims(np.fft.fft(amplitude[n_fft*i:n_fft*(i+1)]), axis=0)
        spectrogram = np.concatenate((spectrogram, temp), axis=0)
        
print(spectrogram.shape)
#%%
magnitude = np.absolute(spectrogram)  # Calculate Magnitude

# Normalize Magnitude to 0 to 1
variance = np.max(magnitude) - np.min(magnitude)
minimum = np.min(magnitude)

magnitude -= minimum
magnitude /= variance
print("max: {} min : {}".format(np.max(magnitude), np.min(magnitude)))
print("shape: {}".format(magnitude.shape))
#%%
plt.imshow(magnitude)
plt.title("Spectrogram")
plt.xlabel("frequency")
plt.ylabel("time")
plt.show()
#%%