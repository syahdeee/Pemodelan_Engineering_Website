import streamlit as st
import tensorflow as tf
import os
import tempfile
from sklearn.preprocessing import MinMaxScaler
import tensorflow_addons as tfa
import pandas as pd
import io

def predict_mlp():
    # Define the custom metric function
    @tf.function
    def r_square(y_true, y_pred):
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

    # Register the custom metric function
    tf.keras.utils.get_custom_objects()["r_square"] = r_square

    nama_data = st.text_input("Masukkan nama data : ")
    #File uploader
    uploaded_file = st.file_uploader("Upload data dalam format xlsx/xls", type=["xlsx", "xls"])
    display = st.button("Display Dataset")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error: Unable to read the file. Please make sure it's a valid Excel file. Exception: {e}")
            st.stop()

    if display and 'df' in locals():
        # df = df[:100]
        st.write("Dataset:")
        st.write(df)

    # File uploader widget
    uploaded_model = st.file_uploader("Upload model dalam format H5 file", type=["h5"])

    if uploaded_model is not None:
        st.write("File uploaded successfully!")

        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(uploaded_model.read())

        # Load the model from the temporary file using Keras and custom metric
        with tf.keras.utils.custom_object_scope({"r_square": r_square}):
            model = tf.keras.models.load_model(temp_path)

        # Remove the temporary file
        os.remove(temp_path)

        if model is not None:
            st.subheader("Model Summary")
            # Use io.StringIO to capture the model summary as a string
            with io.StringIO() as stream:
                # Save the model summary to the string buffer
                model.summary(print_fn=lambda x: stream.write(x + "\n"))
                # Display the model summary from the string buffer
                st.text(stream.getvalue())
        
        st.subheader(f"TENTUKAN FEATURE INPUT DAN OUTPUT DATA {nama_data}")
        
        options = df.columns
        # Elemen input ditempatkan di kolom pertama ()
        input_nfeatures = st.number_input("Masukkan jumlah feature pada dataset yang akan dilakukan prediksi: ", 0)

        # List untuk menyimpan fitur
        features_input = []

        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(input_nfeatures):
            selected_option = st.selectbox(f"Feature {i+1}:", options)
            features_input.append(selected_option)
            
        feature_output = st.selectbox("Masukkan feature output pada dataset yang akan dilakukan prediksi :", options)
        
        data = {}
        
        st.write()
        
        for i in range(len(features_input)):
            min = df[features_input[i]].min()
            max = df[features_input[i]].max()
            data[features_input[i]] = (min, max)
        
        min_output = df[feature_output].min()
        max_output = df[feature_output].max()
            
        if input_nfeatures != 0 :
            st.subheader("INPUTKAN DATA YANG AKAN DIPREDIKSI :")
            
            dfx = pd.DataFrame(data)
            new = {}
            
            for i in range(len(features_input)):
                X = st.number_input(f"Masukkan nilai {features_input[i]} : ")
                
                new[features_input[i]] = X
            
            dfx = dfx.append(new, ignore_index=True)
            
            scaler = MinMaxScaler()
            scaler.fit(dfx)
            x_scaled = scaler.transform(dfx)
            
            dfy = pd.DataFrame({'output': [min_output, max_output]})
                 
            # Fit the scaler to the data
            scaler.fit(dfy)
            
            #prediksi data
            y_model_test = model.predict(x_scaled)
                    
            #inverse ke nilai normal
            y_test_denorm = scaler.inverse_transform(y_model_test)
            hasil = y_test_denorm[-1][0]
                    
            predict = st.button("PREDIKSI")
            if predict:
                st.write(f"Hasil prediksi nilai {feature_output} pada data {nama_data} : ", hasil)
    
predict_mlp()