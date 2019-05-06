import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft, fftfreq, fftshift
from scipy import signal
import numpy as np


rate, data = wav.read('cello.wav')

data = np.mean(data, axis=1)
N = data.shape[0]
w1 = signal.blackman(N)
w2 = signal.nuttall(N)

ywf = fft(data)
ywf2 = fft(data*w1)
ywf3 = fft(data*w2)

xf = fftfreq(N,1/rate)
# xf = [x for x in xf if abs(x) < 1000] #limit 1000

plt.plot(xf[:len(xf)//2],np.abs(ywf[:len(xf)//2]), '-b')
plt.plot(xf[:len(xf)//2],np.abs(ywf2[:len(xf)//2]), '-r')
plt.plot(xf[:len(xf)//2],np.abs(ywf3[:len(xf)//2]), '-g')
plt.show()
