import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq


def get_featured_df(df, window_size=None, sampling_freq=None, n_top_freq=None):
    feature_df = pd.DataFrame()
    # sampling frequency and number of top frequency are needed 
    # for generating features in frequency domain
    if sampling_freq and n_top_freq:
        feature_df = __get_freqdomain_feature(df, sampling_freq, n_top_freq)
        # set window size for time domain feature equal to 
        # window size for frequency domain feature (where window size = length of df)
        window_size = len(df)
    
    # window size is needed for generating feature in time domain 
    if window_size:
        feature_df = pd.concat([feature_df, __get_timedomain_feature(df, window_size)],
                               axis=1)
    else: 
        print('Please provide window size (number of samples) for generating features in time domain')
    return feature_df


def __get_timedomain_feature(df, window_size):

    # get rolling mean
    avg = df.rolling(window_size, center=True).mean()
    avg.columns = [f'avg_{col}' for col in df.columns]
    
    # get the rolling standard deviation
    std = df.rolling(window_size, center=True).std()
    std.columns = [f'std_{col}' for col in df.columns]

    # get the rolling peak value
    peak = df.rolling(window_size, center=True).max()
    peak.columns = [f'peak_{col}' for col in df.columns] 

    # get the rolling rms
    rms_func = lambda d: np.sqrt ((d ** 2).sum() / d.size)
    rms = df.rolling(window_size, center=True).apply(rms_func)
    rms.columns = [f'rms_{col}' for col in df.columns] 

    # get the rolling skewness 
    skew = df.rolling(window_size, center=True).skew()
    skew.columns = [f'skew_{col}' for col in df.columns] 

    # get the rolling kurtosis
    kurt = df.rolling(window_size, center=True).kurt()
    kurt.columns = [f'kurt_{col}' for col in df.columns] 

    # get the rolling crest
    crest = pd.DataFrame(peak.values/rms.values)
    crest.columns = [f'crest_{col}' for col in df.columns] 
    
    # get the rolling clearance
    sub_func = lambda d: np.square ((d ** 0.5).sum() / d.size)
    sub = df.rolling(window_size, center=True).apply(sub_func)
    clear = pd.DataFrame(peak.values/sub.values)
    clear.columns = [f'clear_{col}' for col in df.columns] 
    
    # get the rolling shape
    shape = pd.DataFrame(rms.values/avg.values)
    shape.columns = [f'shape_{col}' for col in df.columns] 

    # get the rolling impulse
    imp = pd.DataFrame(peak.values/avg.values)
    imp.columns = [f'imp_{col}' for col in df.columns] 
    
    # get the rollinng coefficient of variation
    cv = pd.DataFrame(std.values/avg.values)
    cv.columns = [f'cv_{col}' for col in df.columns] 
    
    time_feature = pd.concat([avg, std, peak, rms, skew, kurt, 
                              crest, clear, shape, imp, cv], 
                             axis=1).dropna().reset_index(drop=True)
    return time_feature


def __get_freqdomain_feature(df, sampling_freq, n_top_freq):
    top_freq_df = pd.DataFrame()
    for col in df.columns:
        window_size = len(df[col])
        coef = np.abs(fft(df[col].values))[:window_size//2]
        freq = fftfreq(window_size, 1/sampling_freq)[:window_size//2]
        top_freq = pd.DataFrame(freq[coef.argsort()[-n_top_freq:][::-1]].reshape(1,-1))
        top_freq.columns = [f'freq_{n+1}_{col}' for n in range(n_top_freq)]
        top_freq_df = pd.concat([top_freq_df, top_freq], axis=1)
    return top_freq_df

