import librosa

def cqt(data, hop_size):
    fs, x = data
    if len(x.shape) > 1: x = np.mean(x, axis=1)
    cqt = librosa.core.cqt(x, sr=fs, n_bins=88, hop_length=int(round(hop_size*fs)))
    cqt_delta = librosa.feature.delta(cqt)
    log = librosa.feature.rmse(y=x)
    log_delta = librosa.feature.delta(log)
    features = np.zeros((cqt.shape[1], 178))
    features[:, :88] = cqt.T
    features[:, 88] = log.T[:, 0]
    features[:, 89:177] = cqt_delta.T
    features[:, 177] = log_delta.T[:, 0]
    return features
