
import tensorflow as tf
import torch
from constants import *
import numpy as np

def split_sequences(data, n_input, n_output):
    """Expects input data to be of shape (n_series, series_len, n_features)
    
       Returns:
           - X of shape (n_series * (series_len - (n_input + n_output) + 1), n_input, n_features)
           - Y of shape (n_series * (series_len - (n_input + n_output) + 1), n_output, n_features)
    """
    X = []
    Y = []
    # For each time series, split into sequences of x of length n_input and y of length n_output immediately following x
    for i in range(data.shape[0]):
        # series.shape: (series_len, n_features)
        series = data[i]
        series_len = series.shape[0]
        for j in range(series_len - (n_input + n_output) + 1):
            given_wind_start = j
            given_wind_end = given_wind_start + n_input
            x = series[given_wind_start:given_wind_end, :]
            pred_wind_start = given_wind_end
            pred_wind_end = pred_wind_start + n_output
            y = series[pred_wind_start:pred_wind_end, :]
            
            X.append(x)
            Y.append(y)
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def generate_autoregressive(i=0):
    data = torch.load(kDataFile).numpy()
    rng = np.random.default_rng(seed=7)
    idx = np.arange(72)
    rng.shuffle(idx)
    data_val = data[idx[50:61], :, :]
    print(idx[50])
    n_input = 10
    n_output = 10
    to_gen_len = 500
    model = tf.keras.models.load_model(kAutoregressiveModelFile) 
    series = data_val[i, 0:n_input, :]
    for t in range(to_gen_len - (n_input)):
        y_hat = model.predict(series[-n_input:, :][None, :, :])
        series = np.concatenate((series, y_hat[:, 0, :]), axis=0)
    np.save(kAutoRegSeq, series) 

generate_autoregressive()