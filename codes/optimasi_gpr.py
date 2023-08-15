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


def training_plot(X, y, input, output):
    # Plot the training data
    hasil = plt.figure(figsize=(8, 6))
    hasil = plt.scatter(X, y, c="g", label="Training Data")
    hasil = plt.xlabel(input)
    hasil = plt.ylabel(output)
    hasil = plt.title("Data")
    hasil = plt.legend()
    hasil = plt.show()

    return hasil


def hasil_plot(X, y, y_pred, y_std, input, output):
    # Plot the training data, true function, and predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, c="r", label="Real data")
    plt.scatter(X, y_pred, c="b", label="Prediction")
    plt.plot(X, y_pred, "g-", label="Predict Line")
    plt.fill_between(
        X.ravel(),
        y_pred - 1.96 * y_std,
        y_pred + 1.96 * y_std,
        color="gray",
        alpha=0.3,
        label="Posterior Uncertainty",
    )
    plt.xlabel(input)
    plt.ylabel(output)
    plt.title("Optimized Gaussian Process Regression")
    plt.legend()
    hasil = plt.show()

    return hasil


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


def gpr_model():
    st.markdown(
        "<h5 style='text-align : center;'>Optimasi Hyperparameter (Bayesian Optimization) dan Modelling dengan Gaussian Process Regression (GPR)</h5>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload data dalam format xlsx/xls/csv", type=["xlsx", "xls", "csv"]
    )
    display_button = st.button("Display Dataset")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=";")
            else:
                df = pd.read_excel(uploaded_file, sep=";")
        except Exception as e:
            st.error(
                f"Error: Unable to read the file. Please make sure it's a valid Excel or CSV file. Exception: {e}"
            )
            st.stop()

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
        input = st.selectbox("Masukkan data X:", columns)
        X = df[input].values.reshape(-1, 1)
        output = st.selectbox("Masukkan data y:", columns)
        y = df[output].values.reshape(-1, 1)

        # Show plot data
        # plot_train = training_plot(X, y, input, output)
        # st.pyplot(plot_train)

        # Create a checkbox
        checkbox_rbf = st.checkbox("Radial Basis Function (RBF) Kernel")
        checkbox_matern = st.checkbox("Matérn Kernel")
        checkbox_rational = st.checkbox("Rational quadratic kernel")
        checkbox_exp = st.checkbox("Exp-Sine-Squared kernel")

        hasil_data = {}

        if checkbox_rbf:
            st.subheader("KERNEL TYPE : RBF KERNEL")
            hasil_data["rbf"] = 0

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
            hasil_data["matern"] = 0
            st.markdown(
                "<h5>Matern Kernel Hyperparameter</h5>",
                unsafe_allow_html=True,
            )
            col1_m_1, col2_m_1 = st.columns(2)
            with col1_m_1:
                min_length_scale_matern = st.number_input(
                    "Nilai minimal length scale:", key="input1"
                )

            with col2_m_1:
                max_length_scale_matern = st.number_input(
                    "Nilai maksimal length scale:", key="input2"
                )

            col1_m, col2_m = st.columns(2)
            with col1_m:
                min_nu_matern = st.number_input("Nilai minimal nu:", key="input3")
            with col2_m:
                max_nu_matern = st.number_input("Nilai maksimal nu:", key="input4")

            st.markdown(
                "<h5>White Kernel Hyperparameter</h5>",
                unsafe_allow_html=True,
            )
            col1_m, col2_m = st.columns(2)
            with col1_m:
                min_noise_matern = st.number_input(
                    "Nilai minimal noise level untuk White Kernel:", key="input5"
                )
            with col2_m:
                max_noise_matern = st.number_input(
                    "Nilai maksimal noise level untuk White Kernel:", key="input6"
                )

            st.markdown(
                "<h5>GPR Hyperparameter</h5>",
                unsafe_allow_html=True,
            )
            col1_m, col2_m = st.columns(2)
            with col1_m:
                min_alpha_matern = st.number_input(
                    "Nilai minimal regularization:", key="input7"
                )

            with col2_m:
                max_alpha_matern = st.number_input(
                    "Nilai maksimal regularization:", key="input8"
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

            testSize = st.number_input("Masukkan test size:", key="input9")

            if testSize != 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=testSize, random_state=None, shuffle=True
                )

        if checkbox_rational:
            st.subheader("KERNEL TYPE : RATIONAL QUADRATIC KERNEL")
            hasil_data["rational"] = 0

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
            hasil_data["expsine"] = 0

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
            for key in hasil_data:
                if key == "rbf":
                    st.subheader("HASIL RBF KERNEL")
                    result = kernel_rbf(X_train, X_val, y_train, y_val, space_rbf)
                    # Get the best hyperparameters and the corresponding MSE value
                    (
                        best_kernel_length_scale,
                        best_kernel_noise_level,
                        best_regularization,
                    ) = result.x
                    best_mse_rbf = result.fun

                    hasil_data["rbf"] = best_mse_rbf

                    st.write("Best Hyperparameters:")
                    st.write("Kernel Length Scale:", best_kernel_length_scale)
                    st.write("Kernel Noise Level:", best_kernel_noise_level)
                    st.write("Regularization:", best_regularization)

                    kernel = ConstantKernel() * RBF(
                        length_scale=best_kernel_length_scale
                    ) + WhiteKernel(noise_level=best_kernel_noise_level)
                    best_gpr_rbf = GaussianProcessRegressor(
                        kernel=kernel, alpha=best_regularization
                    )

                    # Fit the GPR model on the training data
                    best_gpr_rbf.fit(X_train, y_train)

                    # Predict on the validation set
                    y_pred, y_std = best_gpr_rbf.predict(X, return_std=True)
                    r2 = r2_score(y, y_pred)
                    hasil_rbf = hasil_plot(X, y, y_pred, y_std, input, output)

                    st.pyplot(hasil_rbf)
                    st.write("R2 Score " + label + ":", r2)
                    st.write("Best MSE:", best_mse_rbf)

                if key == "matern":
                    st.subheader("HASIL MATERN KERNEL")
                    result = kernel_matern(X_train, X_val, y_train, y_val, space_matern)
                    # Get the best hyperparameters and the corresponding MSE value
                    (best_nu, best_length_scale, best_alpha, best_noise) = result.x
                    best_mse_matern = result.fun

                    hasil_data["matern"] = best_mse_matern

                    st.write(hasil_data)

                    st.write("Best Hyperparameters:")
                    st.write("Nilai length scale terbaik:", best_length_scale)
                    st.write("Nilai Nu terbaik:", best_nu)
                    st.write("Nilai noise level terbaik:", best_noise)
                    st.write("Regularization:", best_alpha)

                    best_kernel_matern = ConstantKernel() * Matern(
                        length_scale=best_length_scale, nu=best_nu
                    ) + WhiteKernel(noise_level=best_noise)
                    best_gpr_matern = GaussianProcessRegressor(
                        kernel=best_kernel_matern, alpha=best_alpha
                    )

                    best_gpr_matern.fit(X_train, y_train)

                    # Predict on the validation set
                    y_pred, y_std = best_gpr_matern.predict(X, return_std=True)
                    r2 = r2_score(y, y_pred)
                    hasil_matern = hasil_plot(X, y, y_pred, y_std, input, output)

                    st.pyplot(hasil_matern)
                    st.write("R2 Score " + label + ":", r2)
                    st.write("Best MSE:", best_mse_matern)

                if key == "rational":
                    st.subheader("HASIL RATIONAL KERNEL")
                    result = kernel_rational(
                        X_train, X_val, y_train, y_val, space_rational
                    )
                    # Get the best hyperparameters and the corresponding MSE value
                    (best_length_scale, best_alpha, best_noise, best_regular) = result.x
                    best_mse_rational = result.fun

                    hasil_data["rational"] = best_mse_rational

                    st.write(hasil_data)

                    st.write("Best Hyperparameters:")
                    st.write("Nilai length scale terbaik:", best_length_scale)
                    st.write("Nilai alpha terbaik:", best_alpha)
                    st.write("Nilai noise level terbaik:", best_noise)
                    st.write("Regularization:", best_regular)

                    best_kernel_rational = ConstantKernel() * RationalQuadratic(
                        length_scale=best_length_scale, alpha=best_alpha
                    ) + WhiteKernel(noise_level=best_noise)
                    best_gpr_rational = GaussianProcessRegressor(
                        kernel=best_kernel_rational, alpha=best_regular
                    )

                    best_gpr_rational.fit(X_train, y_train)

                    # Predict on the validation set
                    y_pred, y_std = best_gpr_rational.predict(X, return_std=True)
                    r2 = r2_score(y, y_pred)
                    hasil_rational = hasil_plot(X, y, y_pred, y_std, input, output)

                    st.pyplot(hasil_rational)
                    st.write("R2 Score " + label + ":", r2)
                    st.write("Best MSE:", best_mse_rational)

                if key == "expsine":
                    st.subheader("HASIL EXPSINE KERNEL")
                    result = kernel_expsine(
                        X_train, X_val, y_train, y_val, expsine_space
                    )
                    # Get the best hyperparameters and the corresponding MSE value
                    (
                        best_length_scale,
                        best_periodicity,
                        best_noise,
                        best_regular,
                    ) = result.x
                    best_mse_expsine = result.fun

                    hasil_data["expsine"] = best_mse_expsine

                    st.write(hasil_data)

                    st.write("Best Hyperparameters:")
                    st.write("Nilai length scale terbaik:", best_length_scale)
                    st.write("Nilai periodicity terbaik:", best_periodicity)
                    st.write("Nilai noise level terbaik:", best_noise)
                    st.write("Regularization:", best_regular)

                    best_kernel_expsine = ConstantKernel() * ExpSineSquared(
                        length_scale=best_length_scale, periodicity=best_periodicity
                    ) + WhiteKernel(noise_level=best_noise)
                    best_gpr_expsine = GaussianProcessRegressor(
                        kernel=best_kernel_expsine, alpha=best_regular
                    )

                    best_gpr_expsine.fit(X_train, y_train)

                    # Predict on the validation set
                    y_pred, y_std = best_gpr_expsine.predict(X, return_std=True)
                    r2 = r2_score(y, y_pred)
                    hasil_expsine = hasil_plot(X, y, y_pred, y_std, input, output)

                    st.pyplot(hasil_expsine)
                    st.write("R2 Score " + label + ":", r2)
                    st.write("Best MSE:", best_mse_expsine)

            st.subheader("---------------MODEL TERBAIK-------------")

            keys = hasil_data.keys()
            best_mse_all = min(hasil_data.values())
            best_kernel_all = [
                key for key, value in hasil_data.items() if value == best_mse_all
            ][0]

            st.write(
                "Berdasarkan kernel yang dipilih : ",
                keys,
                " maka didapatkan model terbaik dari kernel: ",
                best_kernel_all,
                " dengan MSE sebesar : ",
                best_mse_all,
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
                    if best_kernel_all == "rbf":
                        joblib.dump(
                            best_gpr_rbf, file_path
                        )  # Save the model using joblib
                        st.success(f"Model saved successfully at: {file_path}")
                        file_dialog.destroy()
                    if best_kernel_all == "matern":
                        joblib.dump(
                            best_gpr_matern, file_path
                        )  # Save the model using joblib
                        st.success(f"Model saved successfully at: {file_path}")
                        file_dialog.destroy()
                    if best_kernel_all == "rational":
                        joblib.dump(
                            best_gpr_rational, file_path
                        )  # Save the model using joblib
                        st.success(f"Model saved successfully at: {file_path}")
                        file_dialog.destroy()
                    if best_kernel_all == "expsine":
                        joblib.dump(
                            best_gpr_expsine, file_path
                        )  # Save the model using joblib
                        st.success(f"Model saved successfully at: {file_path}")
                        file_dialog.destroy()
            except Exception as e:
                st.error("An error occurred while saving the model: " + str(e))


gpr_model()
