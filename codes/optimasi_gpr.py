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

        # memisahkan data untuk train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=None, shuffle=True)
        
        # 1. Define the objective function (MSE) for GPR with covariance tuning
        def objective_function(hyperparameters, covariance_type, X_train, y_train, X_val, y_val):
            if covariance_type == 'RBF':
                kernel_length_scale, noise_level = hyperparameters
                kernel = ConstantKernel() * RBF(length_scale=kernel_length_scale) + WhiteKernel(noise_level=noise_level)
            elif covariance_type == 'Matern':
                kernel_length_scale, noise_level, nu = hyperparameters
                kernel = ConstantKernel() * Matern(length_scale=kernel_length_scale, nu=nu) + WhiteKernel(noise_level=noise_level)
            else:
                raise ValueError("Invalid covariance_type. Supported types: 'RBF', 'Matern'.")

            gpr = GaussianProcessRegressor(kernel=kernel)

            # Split the data into training and validation sets
            # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

            # Fit the GPR model on the training data
            gpr.fit(X_train, y_train)

            # Predict on the validation set
            y_pred = gpr.predict(X_val)

            # Calculate the Mean Squared Error (MSE)
            mse = mean_squared_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            # Minimize the objective function (return negative MSE for maximization)
            return mse

        # 2. Define the hyperparameter space for the covariance functions
        space_rbf = [(0.1, 10.0),   # Range for RBF kernel_length_scale
                    (1e-5, 1e-1)]  # Range for RBF noise_level

        space_matern = [(0.1, 10.0),   # Range for Matern kernel_length_scale
                        (1e-5, 1e-1),  # Range for Matern noise_level
                        (0.5, 2.5)]   # Range for Matern nu (degree of differentiability)

        # 3. Perform Bayesian Optimization for covariance hyperparameters (RBF)
        result_rbf = gp_minimize(lambda hyperparameters: objective_function(hyperparameters, 'RBF', X_train, y_train, X_test, y_test),
                                space_rbf, n_calls=50, random_state=42)

        # 4. Perform Bayesian Optimization for covariance hyperparameters (Matern)
        result_matern = gp_minimize(lambda hyperparameters: objective_function(hyperparameters, 'Matern', X_train, y_train, X_test, y_test),
                                    space_matern, n_calls=50, random_state=42)

        # 5. Get the best hyperparameters and the corresponding MSE value for RBF
        best_kernel_length_scale_rbf, best_noise_level_rbf = result_rbf.x
        best_mse_rbf = result_rbf.fun

        # 6. Get the best hyperparameters and the corresponding MSE value for Matern
        best_kernel_length_scale_matern, best_noise_level_matern, best_nu_matern = result_matern.x
        best_mse_matern = result_matern.fun


        if best_mse_rbf < best_mse_matern:
            best_kernel_rbf = ConstantKernel() * RBF(length_scale=best_kernel_length_scale_rbf) + WhiteKernel(noise_level=best_noise_level_rbf)
            best_gpr_rbf = GaussianProcessRegressor(kernel=best_kernel_rbf)
            best_gpr_rbf.fit(X_train, y_train)

            # Make predictions with the trained Gaussian Process regressor
            y_pred, y_std = best_gpr_rbf.predict(X, return_std=True)
            
            r2 = r2_score(y_test, y_pred)
            
            # st.write("NILAI R2 SCORE YAITU :", r2)
            st.write("Best RBF Covariance Hyperparameters (Length Scale, Noise Level):", best_kernel_length_scale_rbf, best_noise_level_rbf)
            st.write("Best MSE (RBF):", best_mse_rbf)

            # Menyimpan model hasil learning ke dalam file
            model_filename = "model_cofiring_gpr_" + label + ".joblib"
            joblib.dump(best_gpr_rbf, model_filename)

        else:
            best_kernel_matern = ConstantKernel() * Matern(length_scale=best_kernel_length_scale_matern, nu=best_nu_matern) + WhiteKernel(noise_level=best_noise_level_matern)
            best_gpr_matern = GaussianProcessRegressor(kernel=best_kernel_matern)
            best_gpr_matern.fit(X_train, y_train)

            y_pred, y_std = best_gpr_matern.predict(X, return_std=True)
            r2 = r2_score(y_test, y_pred)
            
            st.write("Best Matern Covariance Hyperparameters (Length Scale, Noise Level, nu):", best_kernel_length_scale_matern, best_noise_level_matern, best_nu_matern)
            st.write("Best MSE (Matern):", best_mse_matern)
            # st.write("NILAI R2 SCORE YAITU :", r2)

            # Menyimpan model hasil learning ke dalam file
            model_filename = "model_cofiring_gpr_" + label + ".joblib"
            joblib.dump(best_gpr_matern, model_filename)
        
        # Plot the training data, true function, and predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(X, y, c='r', label='Real data')
        plt.scatter(X, y_pred, c='b', label='Prediction')
        plt.plot(X, y_pred, 'g-', label='Predict Line')
        plt.fill_between(X.ravel(), y_pred - 1.96 * y_std, y_pred + 1.96 * y_std, color='gray', alpha=0.3, label='Posterior Uncertainty')
        plt.xlabel(input)
        plt.ylabel(output)
        plt.title('Optimized Gaussian Process Regression')
        plt.legend()
        hasil_gpr = plt.show()
        
        st.pyplot(hasil_gpr)      
    
gpr_model()
