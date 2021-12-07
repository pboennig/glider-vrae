import numpy as np
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Reshape
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split

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


def partition_data(data):
	rng = np.random.default_rng(seed=7)
	idx = np.arange(72)
	rng.shuffle(idx)

	data_train = data[idx[0:50], :, :]
	data_val = data[idx[50:61], :, :]
	data_test = data[idx[61:72], :, :]

	return data_train, data_val, data_test
	

def main():
	DATA_FILE = './data/processed/x_without_artifact.pt'
	data = torch.load(DATA_FILE)
	data = data.numpy()
	data_train, data_val, data_test = partition_data(data)

	split_params = [2, 5, 10, 20, 50, 100]
	hidden_dims = [5, 20, 50, 100, 200]

	min_loss = float('inf')
	losses = {}

	for n_input in split_params:
		for n_output in split_params:
			for hidden_dim in hidden_dims:
				print(f"Evaluating model for n_input of {n_input} and n_output of {n_output} and hidden_dim {hidden_dim}")
				# Define split sequence parameters
				n_features = data.shape[-1]
				X_train, Y_train = split_sequences(data_train, n_input=n_input, n_output=n_output)
				X_train, Y_train = split_sequences(data_train, n_input=n_input, n_output=n_output)
				X_val, Y_val = split_sequences(data_val, n_input=n_input, n_output=n_output)
				X_test, Y_test = split_sequences(data_test, n_input=n_input, n_output=n_output)
				
				# Define model
				model = Sequential()
				model.add(LSTM(hidden_dim, activation='relu', input_shape=(n_input, n_features)))
				model.add(Dense(n_output * n_features))
				model.add(Reshape((n_output, n_features)))
				model.compile(optimizer='adam', loss='mse')

				# Define model training parameters
				batch_size = 32
				epochs = 2
				model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)

				# Evaluate model on validation data
				loss = model.evaluate(X_val, Y_val)
				if loss < min_loss:
					min_loss = loss
					best_params = (n_input, n_output, hidden_dim)
				losses[(n_input, n_output, hidden_dim)] = loss

	print(f"min val loss of {min_loss} achieved with n_input {n_input}, n_output {n_output}, hidden_dim {hidden_dim}")
	print("sorted losses:")
	print(sorted(losses.items(), key=lambda x: x[1]))

if __name__ == "__main__":
	main()