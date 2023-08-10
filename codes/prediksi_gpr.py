import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

def plot_gpr(new_X_pred, new_y_pred, y_std, input, output):
    # Plot the training data, true function, and predictions
    plt.figure(figsize=(8, 6))
    #plt.scatter(X, y, c='r', label='Real data')
    plt.scatter(new_X_pred, new_y_pred, c='b', label='Prediction')
    plt.plot(new_X_pred, new_y_pred, 'g-', label='Predict Line')
    plt.fill_between(new_X_pred.ravel(), new_y_pred - 1.96 * y_std, new_y_pred + 1.96 * y_std, color='gray', alpha=0.3, label='Posterior Uncertainty')
    plt.xlabel(input)
    plt.ylabel(output)
    plt.title('Optimized Gaussian Process Regression')
    plt.legend()
    hasil = plt.show()
    
    return hasil

def predict_gpr():
    # File uploader
    nama_data = st.text_input("Masukkan nama data : ")
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
        
    uploaded_file = st.file_uploader("Upload a Joblib file", type=["joblib"])

    if uploaded_file is not None:
        try:
            model = joblib.load(uploaded_file)
            st.success("File uploaded successfully!")
            # st.write("Model details:", model)  # Display model details
        except Exception as e:
            st.e
            
    if uploaded_file is not None:
        
        # List untuk menyimpan input X
        new_X_pred = []
        
        columns = df.columns
        input = st.selectbox("Masukkan data X:", columns)
        X = df[input].values.reshape(-1,1)
        output = st.selectbox("Masukkan data y:", columns)
        y = df[output].values.reshape(-1,1)
            
        # Menggunakan model yang telah dimuat untuk prediksi baru
        input_nX = st.number_input("Masukkan jumlah data pada dataset yang akan diprediksi dengan GPR: ", 0)

        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(input_nX):
            x = st.number_input(f"X-{i+1} : ")
            new_X_pred.append(x)
            
        if input_nX != 0:
            new_X_pred = np.array(new_X_pred).reshape(-1, 1)
            #new_y_pred, new_sigma = loaded_gpr.predict(new_X_pred, return_std=True)
            new_y_pred, y_std = model.predict(new_X_pred, return_std=True)
            
            new_X_pred = np.array(new_X_pred).reshape(-1, 1)
            #new_y_pred, new_sigma = loaded_gpr.predict(new_X_pred, return_std=True)
            new_y_pred, y_std = model.predict(new_X_pred, return_std=True)
        
            hasil_plot = plot_gpr(new_X_pred, new_y_pred, y_std, input, output)
            
            if st.button("PREDICT"):
                st.pyplot(hasil_plot)

    
predict_gpr()


