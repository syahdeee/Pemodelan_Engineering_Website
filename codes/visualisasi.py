import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def showHeatmap():
    uploaded_file = st.file_uploader("Upload data dalam format xlsx/xls", type=["xlsx", "xls"])
    display_button = st.button("Display Dataset")

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error: Unable to read the file. Please make sure it's a valid Excel file. Exception: {e}")
            st.stop()

        if display_button and 'df' in locals():
            # df = df[:100]
            st.write("Dataset:")
            st.write(df)
            
            
        opsi = ["Semua Feature", "Feature Pilihan"]

        opsi_feature = st.radio("Pilih fitur yang akan divisualisasikan", opsi)
            
        options = df.columns
        
        if opsi_feature == "Semua Feature":
            features = options
            df = df[features]
            
            if st.button("SHOW HEATMAP"):
                # Compute the correlation matrix
                corr_matrix = df.corr()

                # Generate the heatmap figure
                fig, ax = plt.subplots(figsize=(10, 8))
                heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)

                # Display the heatmap using Streamlit
                st.pyplot(fig)
        else:
            # Elemen input ditempatkan di kolom pertama ()
            nfeatures = st.number_input("Masukkan jumlah feature yang akan divisualisasikan dengan heatmap ", 0)

            # List untuk menyimpan fitur
            features = []

            # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
            for i in range(nfeatures):
                selected_option = st.selectbox(f"Feature {i+1}:", options)
                features.append(selected_option)
                
            df = df[features]
            
            if st.button("SHOW HEATMAP"):
                # Compute the correlation matrix
                corr_matrix = df.corr()

                # Generate the heatmap figure
                fig, ax = plt.subplots(figsize=(10, 8))
                heatmap = sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)

                # Display the heatmap using Streamlit
                st.pyplot(fig)
        
                