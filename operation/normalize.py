import numpy as np

def Normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

def Normalize_withB(data,B):
    mean = np.mean(B)
    std = np.std(B)
    return (data - mean) / std


def Normalize_mean_std(data,mean, std):
    return (data - mean) / std