import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

def training_plot(X, y, input, output):
    # Plot the training data
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, c='g', label='Training Data')
    plt.xlabel(input)
    plt.ylabel(output)
    plt.title('Data')
    plt.legend()
    hasil = plt.show()
    
    return hasil

def kernel_rbf(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)
    def objective_function(hyperparameters, X_train, y_train, X_val, y_val):
        kernel_length_scale, kernel_noise_level, regularization = hyperparameters

        kernel = ConstantKernel() * RBF(length_scale=kernel_length_scale) + WhiteKernel(noise_level=kernel_noise_level)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=regularization)

        # Fit the GPR model on the training data
        gpr.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = gpr.predict(X_val)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_val, y_pred)

        # Minimize the objective function (return negative MSE for maximization)
        return mse

    # Define the hyperparameter space for Bayesian optimization
    space = [
        (0.1, 10.0),     # Range for kernel_length_scale
        (1e-5, 1e-1),    # Range for kernel_noise_level
        (1e-6, 1e-2)     # Range for regularization
    ]

    # Perform Bayesian Optimization for hyperparameters
    result = gp_minimize(lambda hyperparameters: objective_function(hyperparameters, X_train, y_train, X_test, y_test),
                        space, n_calls=50, random_state=42)

    return result

def gpr_model():
    st.markdown("<h5 style='text-align : center;'>Optimasi Hyperparameter (Bayesian Optimization) dan Modelling dengan Gaussian Process Regression (GPR)</h5>", unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload data dalam format xlsx/xls/csv", type=["xlsx", "xls", "csv"])
    display_button = st.button("Display Dataset")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file,  sep=';')
            else:
                df = pd.read_excel(uploaded_file,  sep=';')
        except Exception as e:
            st.error(f"Error: Unable to read the file. Please make sure it's a valid Excel or CSV file. Exception: {e}")
            st.stop()

    if display_button and 'df' in locals():
        df = df[:20]
        st.write("Dataset:")
        st.write(df)
    
    if uploaded_file is not None:
        df = df.dropna()
        
        opsi = ["Ya", "Tidak"]
        columns = df.columns

        opsi_urutkan = st.radio("Apakah data akan diurutkan?", opsi)
            
        if opsi_urutkan == "Ya":
            # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
            label = st.selectbox("Data akan diurutkan berdasarkan fitur:", columns)
            df = df.sort_values(by=[label])
            
        st.write(df)    
        st.subheader("INPUTKAN DATA YANG AKAN DIPREDIKSI (input/output) :")
        input = st.selectbox("Masukkan data X:", columns)
        X = df[input].values.reshape(-1,1)
        output = st.selectbox("Masukkan data y:", columns)
        y = df[output].values.reshape(-1,1)
        
        #Show plot data 
        plot_train = training_plot(X, y, input, output)
        st.pyplot(plot_train)

        
        
gpr_model()
