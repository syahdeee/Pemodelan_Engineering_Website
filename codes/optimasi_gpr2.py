import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tkinter import filedialog
import tkinter as tk
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process.kernels import RationalQuadratic
from skopt import gp_minimize
from sklearn.metrics import r2_score
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error

def kernel_rbf(X_train, X_val, y_train, y_val, space_rbf):
    def objective_function(hyperparameters, X_train, y_train, X_val, y_val):
        kernel_length_scale, kernel_noise_level, regularization = hyperparameters

        kernel = ConstantKernel() * RBF(length_scale=kernel_length_scale) + WhiteKernel(
            noise_level=kernel_noise_level
        )

        gpr = GaussianProcessRegressor(kernel=kernel, alpha=regularization)

        # Fit the GPR model on the training data
        gpr.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = gpr.predict(X_val)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_val, y_pred)

        # Minimize the objective function (return negative MSE for maximization)
        return mse

    # Perform Bayesian Optimization for hyperparameters
    result = gp_minimize(
        lambda hyperparameters: objective_function(
            hyperparameters, X_train, y_train, X_val, y_val
        ),
        space_rbf,
        n_calls=50,
        random_state=42,
    )

    return result


def kernel_matern(X_train, X_val, y_train, y_val, space_matern):
    # Define the objective function for GPR with hyperparameter optimization
    def objective_function(hyperparameters, X_train, y_train, X_val, y_val):
        nu, length_scale, alpha, noise_level = hyperparameters

        kernel = ConstantKernel() * Matern(
            length_scale=length_scale, nu=nu
        ) + WhiteKernel(noise_level=noise_level)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        gpr.fit(X_train, y_train)
        y_pred = gpr.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)

        return mse

    result = gp_minimize(
        lambda hyperparameters: objective_function(
            hyperparameters, X_train, y_train, X_val, y_val
        ),
        space_matern,
        n_calls=50,
        random_state=42,
    )

    return result


def kernel_rational(X_train, X_val, y_train, y_val, space_rational):
    # Define the objective function for GPR with hyperparameter optimization
    def objective_function_rational(hyperparameters, X_train, y_train, X_val, y_val):
        (
            kernel_length_scale,
            kernel_alpha,
            kernel_noise_level,
            regularization,
        ) = hyperparameters

        kernel = ConstantKernel() * RationalQuadratic(
            length_scale=kernel_length_scale, alpha=kernel_alpha
        ) + WhiteKernel(noise_level=kernel_noise_level)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=regularization)

        # Fit the GPR model on the training data
        gpr.fit(X_train, y_train)

        # Predict on the validation set
        y_pred = gpr.predict(X_val)

        # Calculate the Mean Squared Error (MSE)
        mse = mean_squared_error(y_val, y_pred)

        # Minimize the objective function (return negative MSE for maximization)
        return mse

    result = gp_minimize(
        lambda hyperparameters: objective_function_rational(
            hyperparameters, X_train, y_train, X_val, y_val
        ),
        space_rational,
        n_calls=50,
        random_state=42,
    )

    return result


def kernel_expsine(X_train, X_val, y_train, y_val, expsine_space):
    # Define the objective function for GPR with hyperparameter optimization
    def objective_function(hyperparameters, X_train, y_train, X_val, y_val):
        length_scale, periodicity, kernel_noise_level, alpha = hyperparameters

        kernel = ConstantKernel() * ExpSineSquared(
            length_scale=length_scale, periodicity=periodicity
        ) + WhiteKernel(noise_level=kernel_noise_level)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha)

        gpr.fit(X_train, y_train)
        y_pred = gpr.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)

        return mse

    # Perform Bayesian Optimization for hyperparameters
    result = gp_minimize(
        lambda hyperparameters: objective_function(
            hyperparameters, X_train, y_train, X_val, y_val
        ),
        expsine_space,
        n_calls=50,
        random_state=42,
    )

    return result


def gpr_model2():
    st.markdown(
        "<h5 style='text-align : center;'>Optimasi Hyperparameter (Bayesian Optimization) dan Modelling dengan Gaussian Process Regression (GPR)</h5>",
        unsafe_allow_html=True,
    )

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload data dalam format xlsx/xls/csv", type=["xlsx", "xls", "csv"]
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                # Create a list of options for the selectbox
                options_delimiter = [";", ",", ":", "|"]

                # Create a selectbox widget
                delimiter = st.selectbox("Pilih delimiter file yang akan digunakan : ", options_delimiter)
                df = pd.read_csv(uploaded_file, sep=delimiter)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(
                f"Error: Unable to read the file. Please make sure it's a valid Excel or CSV file. Exception: {e}"
            )
            st.stop()
    
    display_button = st.button("Display Dataset")
    
    if display_button and "df" in locals():
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

        input_nfeatures = st.number_input(
            "Masukkan jumlah feature pada dataset yang akan dilakukan learning: ", 0
        )

        # List untuk menyimpan fitur
        features_input = []

        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(input_nfeatures):
            selected_option = st.selectbox(f"Feature Input{i+1}:", columns)
            features_input.append(selected_option)

        output_nfeatures = st.number_input(
            "Masukkan jumlah feature output pada dataset yang akan dilakukan learning :: ",
            0,
        )

        features_output = []

        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(output_nfeatures):
            selected_option = st.selectbox(
                f"Feature Output {i+1}:", columns, key=f"input{i}"
            )
            features_output.append(selected_option)

        X = np.array(df[features_input])
        y = np.array(df[features_output])

        if input_nfeatures != 0 and output_nfeatures != 0:
            y[:, 0] = 0.4 * X[:, 0] + 0.3 * X[:, 1]
            y[:, 1] = 0.6 * X[:, 0] + 0.2 * X[:, 1]
            # Create a checkbox
            st.write("Pilih kernel yang ingin digunakan :")
            checkbox_rbf = st.checkbox("Radial Basis Function (RBF) Kernel")
            checkbox_matern = st.checkbox("Matérn Kernel")
            checkbox_rational = st.checkbox("Rational quadratic kernel")
            checkbox_exp = st.checkbox("Exp-Sine-Squared kernel")

            hasil_data_1 = {}
            hasil_data_2 = {}

            if checkbox_rbf:
                st.subheader("KERNEL TYPE : RBF KERNEL")
                hasil_data_1["rbf"] = 0
                hasil_data_2["rbf"] = 0

                st.markdown(
                    "<h5>RBF Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    min_length_scale_rbf = st.number_input("Nilai minimal length scale:")

                with col2:
                    max_length_scale_rbf = st.number_input("Nilai maksimal length scale:")

                st.markdown(
                    "<h5>White Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    min_noise_rbf = st.number_input("Nilai minimal noise level")

                with col2:
                    max_noise_rbf = st.number_input("Nilai maksimal noise level")

                st.markdown(
                    "<h5>GPR Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1, col2 = st.columns(2)
                with col1:
                    min_alpha_rbf = st.number_input("Nilai minimal regularization:")

                with col2:
                    max_alpha_rbf = st.number_input("Nilai maksimal regularization:")

                # Define the hyperparameter space for Bayesian optimization
                space_rbf = [
                    (
                        min_length_scale_rbf,
                        max_length_scale_rbf,
                    ),  # Range for kernel_length_scale
                    (min_noise_rbf, max_noise_rbf),  # Range for kernel_noise_level
                    (min_alpha_rbf, max_alpha_rbf),  # Range for regularization
                ]

                testSize = st.number_input("Masukkan test size:")

                if testSize != 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=testSize, random_state=None, shuffle=True
                    )

            if checkbox_matern:
                st.subheader("KERNEL TYPE : MATERN KERNEL")
                hasil_data_1["matern"] = 0
                hasil_data_2["matern"] = 0
                st.markdown(
                    "<h5>Matern Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_m_1, col2_m_1 = st.columns(2)
                with col1_m_1:
                    min_length_scale_matern = st.number_input(
                        "Nilai minimal length scale:", key="input01"
                    )

                with col2_m_1:
                    max_length_scale_matern = st.number_input(
                        "Nilai maksimal length scale:", key="input02"
                    )

                col1_m, col2_m = st.columns(2)
                with col1_m:
                    min_nu_matern = st.number_input("Nilai minimal nu:", key="input03")
                with col2_m:
                    max_nu_matern = st.number_input("Nilai maksimal nu:", key="input04")

                st.markdown(
                    "<h5>White Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_m, col2_m = st.columns(2)
                with col1_m:
                    min_noise_matern = st.number_input(
                        "Nilai minimal noise level untuk White Kernel:", key="input05"
                    )
                with col2_m:
                    max_noise_matern = st.number_input(
                        "Nilai maksimal noise level untuk White Kernel:", key="input06"
                    )

                st.markdown(
                    "<h5>GPR Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_m, col2_m = st.columns(2)
                with col1_m:
                    min_alpha_matern = st.number_input(
                        "Nilai minimal regularization:", key="input07"
                    )

                with col2_m:
                    max_alpha_matern = st.number_input(
                        "Nilai maksimal regularization:", key="input08"
                    )

                # Define the hyperparameter space for Bayesian optimization
                space_matern = [
                    (min_nu_matern, max_nu_matern),
                    (min_length_scale_matern, max_length_scale_matern),
                    (
                        min_alpha_matern,
                        max_alpha_matern,
                    ),
                    (min_noise_matern, max_noise_matern),
                ]

                testSize = st.number_input("Masukkan test size:", key="input09")

                if testSize != 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=testSize, random_state=None, shuffle=True
                    )

            if checkbox_rational:
                st.subheader("KERNEL TYPE : RATIONAL QUADRATIC KERNEL")
                hasil_data_1["rational"] = 0
                hasil_data_2["rational"] = 0

                st.markdown(
                    "<h5>Rational Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_r, col2_r = st.columns(2)
                with col1_r:
                    min_length_scale_rational = st.number_input(
                        "Nilai minimal length scale:", key="input21"
                    )

                with col2_r:
                    max_length_scale_rational = st.number_input(
                        "Nilai maksimal length scale:", key="input22"
                    )

                col1_r, col2_r = st.columns(2)
                with col1_r:
                    min_alpha_rational = st.number_input(
                        "Nilai minimal alpha:", key="input23"
                    )
                with col2_r:
                    max_alpha_rational = st.number_input(
                        "Nilai maksimal alpha:", key="input24"
                    )

                st.markdown(
                    "<h5>White Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )

                col1_r, col2_r = st.columns(2)
                with col1_r:
                    min_noise_rational = st.number_input(
                        "Nilai minimal noise level untuk White Kernel:", key="input25"
                    )
                with col2_r:
                    max_noise_rational = st.number_input(
                        "Nilai maksimal noise level untuk White Kernel:", key="input26"
                    )

                st.markdown(
                    "<h5>GPR Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_r, col2_r = st.columns(2)
                with col1_r:
                    min_regular_rational = st.number_input(
                        "Nilai minimal regularization:", key="input27"
                    )

                with col2_r:
                    max_regular_rational = st.number_input(
                        "Nilai maksimal regularization:", key="input28"
                    )

                # Define the hyperparameter space for Bayesian optimization
                space_rational = [
                    (min_length_scale_rational, max_length_scale_rational),
                    (min_alpha_rational, max_alpha_rational),
                    (
                        min_noise_rational,
                        max_noise_rational,
                    ),
                    (min_regular_rational, max_regular_rational),
                ]

                testSize = st.number_input("Masukkan test size:", key="input29")

                if testSize != 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=testSize, random_state=None, shuffle=True
                    )

            if checkbox_exp:
                st.subheader("KERNEL TYPE : EXP-SINE-SQUARED KERNEL")
                hasil_data_1["expsine"] = 0
                hasil_data_2["expsine"] = 0

                st.markdown(
                    "<h5>Expsine Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_e, col2_e = st.columns(2)
                with col1_e:
                    min_length_scale_expsine = st.number_input(
                        "Nilai minimal length scale:", key="input31"
                    )

                with col2_e:
                    max_length_scale_expsine = st.number_input(
                        "Nilai maksimal length scale:", key="input32"
                    )

                col1_e, col2_e = st.columns(2)
                with col1_e:
                    min_periodicity_expsine = st.number_input(
                        "Nilai minimal periodicity:", key="input33"
                    )
                with col2_e:
                    max_periodicity_expsine = st.number_input(
                        "Nilai maksimal periodicity:", key="input34"
                    )

                st.markdown(
                    "<h5>White Kernel Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_e, col2_e = st.columns(2)
                with col1_e:
                    min_noise_expsine = st.number_input(
                        "Nilai minimal noise level untuk white kernel:", key="input35"
                    )
                with col2_e:
                    max_noise_expsine = st.number_input(
                        "Nilai maksimal noise level untuk white kernel:", key="input36"
                    )

                st.markdown(
                    "<h5>GPR Hyperparameter</h5>",
                    unsafe_allow_html=True,
                )
                col1_e, col2_e = st.columns(2)
                with col1_e:
                    min_regular_expsine = st.number_input(
                        "Nilai minimal regularization:", key="input37"
                    )

                with col2_e:
                    max_regular_expsine = st.number_input(
                        "Nilai maksimal regularization:", key="input38"
                    )

                # Define the hyperparameter space for Bayesian optimization
                expsine_space = [
                    (min_length_scale_expsine, max_length_scale_expsine),
                    (min_periodicity_expsine, max_periodicity_expsine),
                    (
                        min_noise_expsine,
                        max_noise_expsine,
                    ),
                    (min_regular_expsine, max_regular_expsine),
                ]

                testSize = st.number_input("Masukkan test size:", key="input39")

                if testSize != 0:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y, test_size=testSize, random_state=None, shuffle=True
                    )

        
        if st.button("Next"):
            for key in hasil_data_1:
                if key == "rbf":
                    st.subheader("HASIL RBF KERNEL")
                    for i in range(len(features_output)):
                        st.write(i + 1, " Hasil Model saat y =", features_output[i])
                        result = kernel_rbf(
                            X_train, X_val, y_train[:, i], y_val[:, i], space_rbf
                        )
                        # Get the best hyperparameters and the corresponding MSE value
                        (
                            best_kernel_length_scale,
                            best_kernel_noise_level,
                            best_regularization,
                        ) = result.x
                        best_mse_rbf = result.fun

                        if i == 0:
                            hasil_data_1["rbf"] = best_mse_rbf
                        else:
                            hasil_data_2["rbf"] = best_mse_rbf

                        st.write("Best Hyperparameters:")
                        st.write("Kernel Length Scale:", best_kernel_length_scale)
                        st.write("Kernel Noise Level:", best_kernel_noise_level)
                        st.write("Regularization:", best_regularization)

                        if i == 0:
                            kernel_1 = ConstantKernel() * RBF(
                                length_scale=best_kernel_length_scale
                            ) + WhiteKernel(noise_level=best_kernel_noise_level)
                            best_gpr_rbf_1 = GaussianProcessRegressor(
                                kernel=kernel_1, alpha=best_regularization
                            )

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_rbf_1.predict(X, return_std=True)
                            r2 = r2_score(y[:, i], y_pred)
                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_rbf)
                        else:
                            kernel_2 = ConstantKernel() * RBF(
                                length_scale=best_kernel_length_scale
                            ) + WhiteKernel(noise_level=best_kernel_noise_level)
                            best_gpr_rbf_2 = GaussianProcessRegressor(
                                kernel=kernel_2, alpha=best_regularization
                            )

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_rbf_2.predict(X, return_std=True)
                            r2 = r2_score(y[:, i], y_pred)
                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_rbf)

                if key == "matern":
                    st.subheader("HASIL MATERN KERNEL")
                    for i in range(len(features_output)):
                        st.write(i + 1, " Hasil Model saat y =", features_output[i])

                        result = kernel_matern(
                            X_train, X_val, y_train[:, i], y_val[:, i], space_matern
                        )

                        #  Get the best hyperparameters and the corresponding MSE value
                        (best_nu, best_length_scale, best_alpha, best_noise) = result.x
                        best_mse_matern = result.fun

                        if i == 0:
                            hasil_data_1["matern"] = best_mse_matern
                            st.write("Best Hyperparameters:")
                            st.write("Nilai length scale terbaik:", best_length_scale)
                            st.write("Nilai Nu terbaik:", best_nu)
                            st.write("Nilai noise level terbaik:", best_noise)
                            st.write("Regularization:", best_alpha)

                            best_kernel_matern_1 = ConstantKernel() * Matern(
                                length_scale=best_length_scale, nu=best_nu
                            ) + WhiteKernel(noise_level=best_noise)
                            best_gpr_matern_1 = GaussianProcessRegressor(
                                kernel=best_kernel_matern_1, alpha=best_alpha
                            )

                            best_gpr_matern_1.fit(X_train, y_train[:, i])

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_matern_1.predict(
                                X, return_std=True
                            )
                            r2 = r2_score(y[:, i], y_pred)

                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_matern)
                        else:
                            hasil_data_2["matern"] = best_mse_matern
                            st.write("Best Hyperparameters:")
                            st.write("Nilai length scale terbaik:", best_length_scale)
                            st.write("Nilai Nu terbaik:", best_nu)
                            st.write("Nilai noise level terbaik:", best_noise)
                            st.write("Regularization:", best_alpha)

                            best_kernel_matern_2 = ConstantKernel() * Matern(
                                length_scale=best_length_scale, nu=best_nu
                            ) + WhiteKernel(noise_level=best_noise)
                            best_gpr_matern_2 = GaussianProcessRegressor(
                                kernel=best_kernel_matern_2, alpha=best_alpha
                            )

                            best_gpr_matern_2.fit(X_train, y_train[:, i])

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_matern_2.predict(
                                X, return_std=True
                            )
                            r2 = r2_score(y[:, i], y_pred)

                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_matern)

                if key == "rational":
                    st.subheader("HASIL RATIONAL KERNEL")
                    for i in range(len(features_output)):
                        st.write(i + 1, " Hasil Model saat y =", features_output[i])
                        result = kernel_rational(
                            X_train, X_val, y_train[:, i], y_val[:, i], space_rational
                        )
                        # Get the best hyperparameters and the corresponding MSE value
                        (
                            best_length_scale,
                            best_alpha,
                            best_noise,
                            best_regular,
                        ) = result.x
                        best_mse_rational = result.fun

                        if i == 0:
                            hasil_data_1["rational"] = best_mse_rational
                            st.write("Best Hyperparameters:")
                            st.write("Nilai length scale terbaik:", best_length_scale)
                            st.write("Nilai alpha terbaik:", best_alpha)
                            st.write("Nilai noise level terbaik:", best_noise)
                            st.write("Regularization:", best_regular)

                            best_kernel_rational_1 = (
                                ConstantKernel()
                                * RationalQuadratic(
                                    length_scale=best_length_scale, alpha=best_alpha
                                )
                                + WhiteKernel(noise_level=best_noise)
                            )
                            best_gpr_rational_1 = GaussianProcessRegressor(
                                kernel=best_kernel_rational_1, alpha=best_regular
                            )

                            best_gpr_rational_1.fit(X_train, y_train[:, i])

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_rational_1.predict(
                                X, return_std=True
                            )
                            r2 = r2_score(y[:, i], y_pred)

                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_rational)
                        else:
                            hasil_data_2["rational"] = best_mse_rational

                            st.write("Best Hyperparameters:")
                            st.write("Nilai length scale terbaik:", best_length_scale)
                            st.write("Nilai alpha terbaik:", best_alpha)
                            st.write("Nilai noise level terbaik:", best_noise)
                            st.write("Regularization:", best_regular)

                            best_kernel_rational_2 = (
                                ConstantKernel()
                                * RationalQuadratic(
                                    length_scale=best_length_scale, alpha=best_alpha
                                )
                                + WhiteKernel(noise_level=best_noise)
                            )
                            best_gpr_rational_2 = GaussianProcessRegressor(
                                kernel=best_kernel_rational_2, alpha=best_regular
                            )

                            best_gpr_rational_2.fit(X_train, y_train[:, i])

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_rational_2.predict(
                                X, return_std=True
                            )
                            r2 = r2_score(y[:, i], y_pred)

                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_rational)

                if key == "expsine":
                    st.subheader("HASIL EXPSINE KERNEL")
                    for i in range(len(features_output)):
                        st.write(i + 1, " Hasil Model saat y =", features_output[i])

                        result = kernel_expsine(
                            X_train, X_val, y_train[:, i], y_val[:, i], expsine_space
                        )
                        # Get the best hyperparameters and the corresponding MSE value
                        (
                            best_length_scale,
                            best_periodicity,
                            best_noise,
                            best_regular,
                        ) = result.x
                        best_mse_expsine = result.fun

                        if i == 0:
                            hasil_data_1["expsine"] = best_mse_expsine

                            st.write("Best Hyperparameters:")
                            st.write("Nilai length scale terbaik:", best_length_scale)
                            st.write("Nilai periodicity terbaik:", best_periodicity)
                            st.write("Nilai noise level terbaik:", best_noise)
                            st.write("Regularization:", best_regular)

                            best_kernel_expsine_1 = ConstantKernel() * ExpSineSquared(
                                length_scale=best_length_scale,
                                periodicity=best_periodicity,
                            ) + WhiteKernel(noise_level=best_noise)
                            best_gpr_expsine_1 = GaussianProcessRegressor(
                                kernel=best_kernel_expsine_1, alpha=best_regular
                            )

                            best_gpr_expsine_1.fit(X_train, y_train[:, i])

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_expsine_1.predict(
                                X, return_std=True
                            )
                            r2 = r2_score(y[:, i], y_pred)

                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_expsine)
                        else:
                            hasil_data_2["expsine"] = best_mse_expsine
                            st.write("Nilai length scale terbaik:", best_length_scale)
                            st.write("Nilai periodicity terbaik:", best_periodicity)
                            st.write("Nilai noise level terbaik:", best_noise)
                            st.write("Regularization:", best_regular)

                            best_kernel_expsine_2 = ConstantKernel() * ExpSineSquared(
                                length_scale=best_length_scale,
                                periodicity=best_periodicity,
                            ) + WhiteKernel(noise_level=best_noise)
                            best_gpr_expsine_2 = GaussianProcessRegressor(
                                kernel=best_kernel_expsine_2, alpha=best_regular
                            )

                            best_gpr_expsine_2.fit(X_train, y_train[:, i])

                            # Predict on the validation set
                            y_pred, y_std = best_gpr_expsine_2.predict(
                                X, return_std=True
                            )
                            r2 = r2_score(y[:, i], y_pred)

                            st.write("R2 Score " + ":", r2)
                            st.write("Best MSE:", best_mse_expsine)

            st.subheader("---------------MODEL TERBAIK-------------")

            keys = hasil_data_1.keys()
            best_mse_all_1 = min(hasil_data_1.values())
            best_kernel_all_1 = [
                key for key, value in hasil_data_1.items() if value == best_mse_all_1
            ][0]

            st.write("HASIL MODEL SAAT Y = ", features_output[0])
            st.write(
                "Berdasarkan kernel yang dipilih : ",
                keys,
                " maka didapatkan model terbaik dari kernel: ",
                best_kernel_all_1,
                " dengan MSE sebesar : ",
                best_mse_all_1,
            )

            best_mse_all_2 = min(hasil_data_2.values())
            best_kernel_all_2 = [
                key for key, value in hasil_data_2.items() if value == best_mse_all_2
            ][0]

            st.write("HASIL MODEL SAAT Y = ", features_output[1])
            st.write(
                "Berdasarkan kernel yang dipilih : ",
                keys,
                " maka didapatkan model terbaik dari kernel: ",
                best_kernel_all_2,
                " dengan MSE sebesar : ",
                best_mse_all_2,
            )
            
            try:
                    root = tk.Tk()
                    root.withdraw()
                    file_dialog = tk.Toplevel(root)
                    file_dialog.attributes("-topmost", True)
                    file_path = filedialog.asksaveasfilename(
                        parent=file_dialog, defaultextension=".joblib"
                    )

                    if file_path:
                        if best_kernel_all_1 == "rbf":
                            joblib.dump(
                                best_gpr_rbf_1, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
                        if best_kernel_all_1 == "matern":
                            joblib.dump(
                                best_gpr_matern_1, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
                        if best_kernel_all_1 == "rational":
                            joblib.dump(
                                best_gpr_rational_1, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
                        if best_kernel_all_1 == "expsine":
                            joblib.dump(
                                best_gpr_expsine_1, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
            except Exception as e:
                    st.error("An error occurred while saving the model: " + str(e))
                    
            try:
                    root = tk.Tk()
                    root.withdraw()
                    file_dialog = tk.Toplevel(root)
                    file_dialog.attributes("-topmost", True)
                    file_path = filedialog.asksaveasfilename(
                        parent=file_dialog, defaultextension=".joblib"
                    )

                    if file_path:
                        if best_kernel_all_2 == "rbf":
                            joblib.dump(
                                best_gpr_rbf_2, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
                        if best_kernel_all_2 == "matern":
                            joblib.dump(
                                best_gpr_matern_2, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
                        if best_kernel_all_2 == "rational":
                            joblib.dump(
                                best_gpr_rational_2, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
                        if best_kernel_all_2 == "expsine":
                            joblib.dump(
                                best_gpr_expsine_2, file_path
                            )  # Save the model using joblib
                            st.success(f"Model saved successfully at: {file_path}")
                            file_dialog.destroy()
            except Exception as e:
                    st.error("An error occurred while saving the model: " + str(e))




gpr_model2()
