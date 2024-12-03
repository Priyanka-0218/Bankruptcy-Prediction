import streamlit as st
import numpy as np
import pickle

# Load the model from the pickle file
with open('trained_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

def predict_bankruptcy(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Ensure the input data has the correct number of features
    expected_features = loaded_model.n_features_in_
    
    if input_data_reshaped.shape[1] == expected_features:
        prediction = loaded_model.predict(input_data_reshaped)
        return prediction[0]
    else:
        raise ValueError(f"Feature shape mismatch, expected: {expected_features}, got {input_data_reshaped.shape[1]}")

# Streamlit app for bankruptcy prediction
def main():
    st.title('Bankruptcy Prediction Web App')

    # Define the input data fields
    input_data = [st.number_input(f'Feature {i+1}', min_value=0, max_value=1000, value=0) for i in range(15)]

    if st.button('Predict'):
        try:
            result = predict_bankruptcy(input_data)
            if result == 0:
                st.success('NO Bankruptcy')
            else:
                st.error('Bankruptcy')
        except ValueError as e:
            st.error(str(e))

if __name__ == '__main__':
    main()
