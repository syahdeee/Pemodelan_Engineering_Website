import streamlit as st 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sklearn.ensemble as ske
from io import BytesIO
import numpy as np

def preprocessing():
    # File uploader
    file_pro = st.file_uploader("Upload data dalam format xlsx/xls/csv", type=["xlsx", "xls", "csv"])
    display_button = st.button("Display Dataset")

    if file_pro is not None:
        try:
            if file_pro.name.endswith('.csv'):
                df = pd.read_csv(file_pro,  sep=';')
            else:
                df = pd.read_excel(file_pro)
        except Exception as e:
            st.error(f"Error: Unable to read the file. Please make sure it's a valid Excel or CSV file. Exception: {e}")
            st.stop()

    if display_button and 'df' in locals():
        df = df[:100]
        st.write("Dataset:")
        st.write(df)
        
    st.write("Pilih data preprocessing yang akan dilakukan :")    
    checkbox_heatmap = st.checkbox("Visualisasi Heatmap")
    checkbox_fs = st.checkbox("Feature Selection dengan Random Forest")
    checkbox_normalisasi = st.checkbox("Normalisasi")
    checkbox_denormalisasi = st.checkbox("Denormalisasi")
    
    if file_pro is not None:
        if checkbox_heatmap:
            st.subheader("VISUALISASI HEATMAP")
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
                    st.pyplot(fig)  # Display the entire figure

                    # Save the figure if needed
                    fig.savefig("heatmap.png")  # Save the figure to a file
                    
        if checkbox_fs:
        # st.write(df)
            st.subheader("FEATURE SELECTION DENGAN RANDOM FOREST")
                    
            options = df.columns
            output = st.selectbox("Masukkan feature y (target) : ", options)

            y = df[output]
            X = df.drop(columns=[output])  # Drop the 'output' column from X
            
            #Random Forest for Feature Selection
            reg = ske.RandomForestRegressor()
            reg.fit(X, y)
            fet_ind = np.argsort(reg.feature_importances_)[::-1]
            fet_imp = reg.feature_importances_[fet_ind]
            
            # Create a DataFrame to store the feature selection results
            feature_names = X.columns.values
            feature_selection_results = pd.DataFrame({'Feature': feature_names[fet_ind], 'Importance': fet_imp})
            
            st.write("Hasil feature selection : ")
            st.write(feature_selection_results)
                
            show_Nfeature = st.number_input("Masukkan jumlah feature teratas yang ingin didapatkan : ", 0)
                
            # Mendapatrkan nFeature terbaik
            labels = feature_names[fet_ind][:show_Nfeature]
            if show_Nfeature != 0:
                # Print each value in the array
                top_values = ', '.join(str(value) for value in labels[:show_Nfeature])
                st.write(f"{show_Nfeature} feature terbaik yaitu : ", top_values)

            opsi_save = ["Ya", "Tidak"]
            save_dataset = st.selectbox("Apakah feature hasil seleksi akan disimpan ke dalam dataset sebelumnya?", opsi_save)
                
            if save_dataset == "Ya":
                df = df[labels]
                st.write("Dataset hasil feature selection : ", df)
            else:
                df = df
                           
        if checkbox_normalisasi:
            
            st.subheader("NORMALISASI DATA")
            
            opsi_norm = ["Semua feature", "Feature pilihan"]
            norm_opsi = st.radio("Normalisasi akan dilakukan pada : ", opsi_norm)
            
            options = df.columns
            
            if norm_opsi == "Feature pilihan":
                nfeatures = st.number_input("Masukkan banyak feature pada dataset yang akan dilakukan normalisasi : ", 0)
                # List untuk menyimpan fitur
                features_normalized = []

                # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
                for i in range(nfeatures):
                    selected_option = st.selectbox(f"Feature {i+1}:", options)
                    features_normalized.append(selected_option)
                
                if nfeatures != 0:
                    X = df[features_normalized]
                    # Create an instance of the StandardScaler
                    scaler = MinMaxScaler()
                    # Fit the scaler to the data
                    scaler.fit(X)
                    # Transform the data
                    data_scaled = scaler.transform(X)
                    
                    # Create a DataFrame
                    df = pd.DataFrame(data_scaled, columns=features_normalized)
                    
                    st.write("DATA HASIL NORMALISASI : ", df)
            else:
                features_normalized = options
                X = df[features_normalized]
                # Create an instance of the StandardScaler
                scaler = MinMaxScaler()
                # Fit the scaler to the data
                scaler.fit(X)
                # Transform the data
                data_scaled = scaler.transform(X)
                
                # Create a DataFrame
                df = pd.DataFrame(data_scaled, columns=features_normalized)
                
                st.write("DATA HASIL NORMALISASI : ", df)
        
        if checkbox_denormalisasi:
            st.subheader("DENORMALISASI DATA")
            koloms = df.columns
            # Initialize MinMaxScaler
            scaler_denorm = MinMaxScaler()

            # Fit the scaler on the data to capture the scaling parameters
            scaler_denorm.fit(df)

            # Denormalize the data
            df_denormalized = scaler.inverse_transform(df)

            # Create a DataFrame from the denormalized data
            df = pd.DataFrame(df_denormalized, columns=koloms)
            
            st.write(df)
        
        if  checkbox_fs or checkbox_normalisasi or checkbox_denormalisasi:
            opsi_convert = ["csv", "excel"]
            convert_to = st.radio("Dataset akan didownload dalam format", opsi_convert)
        
            if convert_to == "csv":
                # Download Button for CSV
                csv_button = st.download_button(
                    label="DOWNLOAD DATASET",
                    data=df.to_csv(index=False).encode(),
                    file_name='output.csv',
                    mime='text/csv'
                )
                
                st.write(csv_button)
            else:
                
                # Download Button for Excel
                excel_buffer = BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_data = excel_buffer.getvalue()
                excel_button = st.download_button(
                    label="Download Excel",
                    data=excel_data,
                    file_name='output.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
                
                st.write(excel_button)


preprocessing()