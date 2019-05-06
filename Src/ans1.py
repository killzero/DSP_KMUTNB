####################################################################
# File: signal_fft_demo-4.py
# Date: 2019-02-19
####################################################################

import numpy as np
from scipy import signal
from scipy.fftpack import fft
import matplotlib.pyplot as plt

Fs = 200      # sampling frequency Hz
N = 512       # Number of sample points: select 256, 512, and 1024
T = 1.0 / Fs  # sample spacing

x = np.linspace(0.0, N*T, N)  # discrete-time sequence
y1 = 1.0*np.sin( 2.0*np.pi*x *  20 )  # 20 Hz
y2 = 0.5*np.sin( 2.0*np.pi*x *  40 )  # 40 Hz
y3 = 0.2*np.sin( 2.0*np.pi*x *  60 )  # 60 Hz
y = y1 + y2 + y3

# create a Blackman window
w = signal.blackman(N)
w2 = signal.nuttall(N)
w3 = signal.triang(N)


# apply FFT to the input sequence (multiplied with rectangular windowing function)
yf = fft(y)

# apply FFT to the input sequence multiplied with Blackman windowing function
ywf = fft(y*w)
yw2f = fft(y*w2)
yw3f = fft(y*w3)

# plot the absolute value of FFT coefficients for positive frequencies only

xf = np.linspace(0.0, 1.0/(2.0*T), N//2)  # frequency in Hz
# semilog plot 
plt.semilogy( xf,  2.0/N * np.abs(yf[0:N//2]),  '-b')
plt.semilogy( xf,  2.0/N * np.abs(ywf[0:N//2]), '-r')
plt.semilogy( xf,  2.0/N * np.abs(yw2f[0:N//2]),  '-g')
plt.semilogy( xf,  2.0/N * np.abs(yw3f[0:N//2]),  '-y')

plt.legend(['FFT', 'FFT with Blackman window', 'FFT with Nuttall window', 'FFT with triangular window'])
plt.grid()
#plt.savefig('fft_demo-4.png')
plt.show()

####################################################################

