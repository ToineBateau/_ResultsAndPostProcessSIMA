import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt

def PQ_analysis(x_pks):
    dphi = -np.diff(x_pks)
    mean_amp = 0.5*(x_pks[1:]+x_pks[:-1])
    y = np.divide(dphi,mean_amp)
    P,Q = np.polynomial.polynomial.polyfit(mean_amp,y,1)
    return P, Q, mean_amp, y

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

def compare_decay_tests(H_data, outputs, focus ,colors={},styles={},labels={},figsize=(16,7)):
    H_data = H_data.selection(
        {
            'model':focus['model'],
            'condition':focus['condition'],
            'analysis':focus['analysis']
        }
    )
    n_o = len(outputs)
    fig,ax = plt.subplots(n_o // 2 + n_o % 2, 2, figsize=figsize, dpi=600)
    legend_labels = []
    res = {}
    model = focus['model']
    for cond in H_data.conds:
        color = colors[cond] if len(colors) > 0 else None
        style = styles[cond] if len(styles) > 0 else '-'
        label = labels[cond] if len(labels) > 0 else cond
        legend_labels.append(label)
        dict_res = {}
        for i_dof, output in enumerate(outputs):
            t,x = H_data.timeserie({'model':model, 'condition':cond, 'analysis':'Dynamic'}, output, show=False)
            peak_idxs = sg.find_peaks(x, prominence=0.001)[0]
            T_n = np.diff(t[peak_idxs][1:-1]).mean()
            plt.sca(ax.flatten('F')[i_dof])
            plt.plot(t, x, label=label + f' ($T_n$={T_n:.2f})', color=color, linestyle=style)
            plt.plot(t[peak_idxs], x.values[peak_idxs], 'o', color=color, markersize=3)

            if output == focus['output']:   
                x_filt = x.values-np.mean(x.values)
                plt.plot(t, x_filt, color=color, alpha=0.3, linestyle=style)
                plt.plot(t[peak_idxs], x_filt[peak_idxs], 'o', color=color, alpha=0.3, markersize=3)
                P,Q, amp, y = PQ_analysis(x_filt[peak_idxs])
                print(output + ' analysis for cond ' + cond + ' : P = ' + str(P) + ' ; Q = ' + str(Q) + ' ; T_n = ' + str(T_n))
                dict_res[output] = (P,Q,amp,y,T_n)
            
            plt.xlabel('Time (s)')
            plt.ylabel(f'{output}')

        res[cond] = dict_res
    for axis in ax.flatten('F'):
        axis.grid()
        axis.legend(loc='upper right', fontsize='small')
    fig.tight_layout()
    return fig, ax, res