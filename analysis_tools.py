import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import h5py

def PQ_analysis(t,x):
    peak_idxs = sg.find_peaks(x, prominence=0.001)[0]
    T_n = np.diff(t[peak_idxs][1:-3]).mean()
    # plt.sca(ax.flatten('F')[i_dof-1])
    # plt.plot(t, x, label=label + f' ($T_n$={T_n:.2f})', color=color, linestyle=style)
    # plt.plot(t[peak_idxs], x[peak_idxs], 'o', color=color, markersize=3)
    pks = x[peak_idxs]
    dphi = -np.diff(pks)
    mean_amp = 0.5*(pks[1:]+pks[:-1])
    y = np.divide(dphi,mean_amp)
    P,Q = np.polynomial.polynomial.polyfit(mean_amp,y,1)
    return P, Q

def bw_filter(data, cutoff, fs, order, switch):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = sg.butter(order, normal_cutoff, btype=switch, analog=False)
    y = sg.filtfilt(b, a, data)
    return y

def circular_perm(x, i_lag):
    n = len(x)
    i_lag = i_lag % n  # Optimize lag to avoid redundant computation
    if i_lag == 0:
        return x
    return np.concatenate((x[-i_lag:], x[:-i_lag]))

def circular_matrix(x):
    from scipy.linalg import circulant
    mat = circulant(x)
    m = np.transpose(mat)
    return(m)

# example = np.array([1, 2, 3, 2, 3, 4])
# print(circular_perm(example, 2))
# print(circular_perm(example, -2))
# print(circular_perm(example, len(example)))
# print(circular_perm(example, 0))
# print(circular_matrix(example))

def discrete_autocorrelation(x):
    n = len(x)
    Rxx_tau = np.zeros(2 * n - 1)
    for i_lag in range(2 * n - 1):
        lagged_x = circular_perm(x, i_lag - n)
        Rxx_tau[i_lag] = np.dot(x, lagged_x) / n
    return Rxx_tau

def mat_autocorrelation(x):
    m = circular_matrix(x)
    return(np.dot(m,x)/len(x))

# print(example * example)
# lnsp = np.linspace(0, 4, 1000)
# X = np.linspace(-lnsp[-1], lnsp[-1], 2*len(lnsp)-1)
# example2 = np.sin(lnsp*2*np.pi)
# plt.plot(X, discrete_autocorrelation(example2))
# plt.plot(lnsp, mat_autocorrelation(example2))

def FFT(timeserie,dt):
    N = len(timeserie)
    NFFT = 1 << ((N-1)//2).bit_length()  # Efficient power of 2 calculation
    # f, s= np.fft.rfftfreq(N, d=dt), np.abs(np.fft.rfft(timeserie, norm='forward'))*2
    # f, s= np.fft.rfftfreq(NFFT, d=dt), np.abs(np.fft.rfft(timeserie, n=NFFT, norm='forward'))*2
    # f, s= np.fft.fftfreq(NFFT, d=dt)[:NFFT//2], np.abs(np.fft.fft(timeserie, n=NFFT, norm='backward')[:NFFT//2])*2/NFFT
    f, s = np.fft.rfftfreq(NFFT, dt), np.abs(np.fft.rfft(timeserie, n=NFFT))*2/NFFT
    return f,s
    
def filt_FFT(timeserie, dt):
    f,s = FFT(timeserie, dt)
    s = bw_filter(s, cutoff=1/dt*2, fs=1/(f[1]-f[0]), order=2, switch='low')
    return f,s

# Z = np.linspace(0,np.pi*20,100000)
# sinewave = 12 * np.sin(2*np.pi*Z) + 3 * np.sin(100*2*np.pi*Z)
# ds = Z[1]-Z[0]
# F,S = FFT(sinewave,ds)

# plt.plot(F,S)

def PSD_wave(timeserie, dt):
    Rxx_tau = discrete_autocorrelation(timeserie)
    # Rxx_tau = mat_autocorrelation(timeserie)
    freq, Sp = FFT(Rxx_tau,dt)
    Sp *= 1
    return freq, Sp

def PSD_wave2(timeserie, dt):
    N = len(timeserie)
    NFFT = 1 << (N-1).bit_length()  # Efficient power of 2 calculation
    freq, Sp = sg.welch(timeserie,window=sg.windows.blackmanharris(N),fs= 1/dt) #window=sg.windows.blackmanharris(N), 
    return freq, Sp

def PSD_wave3(timeserie, dt):
    N = len(timeserie)
    NFFT = 1 << ((N-1)//2).bit_length()  # Efficient power of 2 calculation
    f_fft, x_fft = np.fft.fftfreq(NFFT, d=dt), np.fft.fft(timeserie, n=NFFT, norm='backward')
    Sp = x_fft * np.conjugate(x_fft)/NFFT
    Sp *= 2 * (dt)
    return f_fft[:NFFT//2], Sp[:NFFT//2]

def PSD_wave4(timeserie, dt):
    Nwin = len(timeserie)//4
    return (sg.welch(timeserie, window=sg.windows.blackmanharris(Nwin), fs=1/dt, nperseg=Nwin))

def PSD_wave5(timeserie, dt):
    N = len(timeserie)
    NFFT = 1 << ((N-1)//2).bit_length()  # Efficient power of 2 calculation
    f_fft, x_fft = np.fft.fftfreq(NFFT, d=dt), np.fft.fft(timeserie, n=NFFT, norm='backward')
    x_fft = bw_filter(x_fft, cutoff=0.1/f_fft[1], fs=1/f_fft[1], order=2, switch='low')
    # Sp = x_fft * np.conjugate(x_fft)/NFFT
    Sp = np.abs(x_fft)**2/NFFT
    Sp *= 2 * (dt)
    Sp = bw_filter(Sp, cutoff=0.1/f_fft[1], fs=1/f_fft[1], order=2, switch='low')
    f, Sp = f_fft[:NFFT//2], Sp[:NFFT//2]
    return f_fft[:NFFT//2], Sp[:NFFT//2]