import streamlit as st
import tensorflow as tf
import os
import tempfile
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import io


def reverse_dictionary(original_dict):
    reversed_dict = {value: key for key, value in original_dict.items()}
    return reversed_dict


def prediksi_klasifikasi():
    nama_data = st.text_input("Masukkan nama data : ")
    # File uploader
    # File uploader
    file_pro = st.file_uploader(
        "Upload data dalam format xlsx/xls/csv", type=["xlsx", "xls", "csv"]
    )
    display = st.button("Display Dataset")

    if file_pro is not None:
        try:
            if file_pro.name.endswith(".csv"):
                df = pd.read_csv(file_pro, sep=",")
            else:
                df = pd.read_excel(file_pro)
        except Exception as e:
            st.error(
                f"Error: Unable to read the file. Please make sure it's a valid Excel or CSV file. Exception: {e}"
            )
            st.stop()

    if display and "df" in locals():
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

        # Load the model from the temporary file using Keras
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

    if file_pro is not None:
        options = df.columns
        # Elemen input ditempatkan di kolom pertama ()
        input_nfeatures = st.number_input(
            "Masukkan jumlah feature pada dataset yang akan dilakukan prediksi: ", 0
        )

        # List untuk menyimpan fitur
        features_input = []

        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(input_nfeatures):
            selected_option = st.selectbox(f"Feature {i+1}:", options)
            features_input.append(selected_option)

        feature_output = st.selectbox(
            "Masukkan feature output pada dataset yang akan dilakukan prediksi :",
            options,
        )

        y = df[feature_output]
        le = LabelEncoder()
        y_encode = le.fit_transform(y)

        label_encoding_dict = {label: y_encode for label, y_encode in zip(y, y_encode)}

        # Reverse the dictionary to create a new dictionary with keys and values swapped
        reversed_dict = {int(value): key for key, value in label_encoding_dict.items()}

        data = {}

        for i in range(len(features_input)):
            min = df[features_input[i]].min()
            max = df[features_input[i]].max()
            data[features_input[i]] = (min, max)

        if input_nfeatures != 0:
            st.subheader("INPUTKAN DATA YANG AKAN DIPREDIKSI :")

            dfx = pd.DataFrame(data)
            new = {}

            for i in range(len(features_input)):
                X = st.number_input(
                    f"Masukkan nilai {features_input[i]} : ", key=f"input{i}"
                )

                new[features_input[i]] = X

            dfx = dfx.append(new, ignore_index=True)

            scaler = MinMaxScaler()
            scaler.fit(dfx)
            x_scaled = scaler.transform(dfx)

            y_pred = model.predict(x_scaled)
            y_pred_classes = np.argmax(y_pred, axis=1)
            hasil = y_pred_classes[-1]

            predict = st.button("PREDIKSI")
            if predict:
                # Look up the key using the reversed dictionary
                if hasil in reversed_dict:
                    value = reversed_dict[hasil]
                    st.write(
                        f"Hasil prediksi nilai {feature_output} pada data {nama_data} : ",
                        value,
                    )


prediksi_klasifikasi()
