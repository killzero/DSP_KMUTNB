import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq, fftshift
from scipy import signal
import numpy as np

# rate, data = wav.read('cello.wav')
# rate, data = wav.read('ChillingMusic.wav')
rate, data = wav.read('NubiaCantaDalva.wav')

b, a = signal.butter(5, 3.6*2/300)

datafilter = signal.lfilter(b, a, data)

wav.write('output.wav',rate,datafilter.astype(np.int16))

plt.plot(data,'-g')
plt.plot(datafilter,'-r')
plt.show()
