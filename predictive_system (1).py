# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:29:40 2024

@author: aaksh
"""

import pickle
import numpy as np

# Load the model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Example input data with 15 features
input_data = (5, 66, 72, 19, 175, 55, 0, 8, 45, 85, 54, 8, 25, 45, 25)

# Convert input data to a NumPy array and reshape it for prediction
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Ensure the input data has the correct number of features
expected_features = loaded_model.n_features_in_

if input_data_reshaped.shape[1] == expected_features:
    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        print('NO Bankruptcy')
    else:
        print('Bankruptcy')
else:
    print(f"Feature shape mismatch, expected: {expected_features}, got {input_data_reshaped.shape[1]}")
