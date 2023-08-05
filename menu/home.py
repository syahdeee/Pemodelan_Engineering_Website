import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from model.menu.optimasi import optimasi_func
from model.menu.optimasi import normalisasi_input
from model.menu.optimasi import normalisasi_output
from model.menu.optimasi import get_best_hyperparameter
from model.menu.optimasi import get_model
from model.menu.optimasi import print_model_summary

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
import pandas as pd

# Provide the path to your H5 model file
model_path_diesel = 'D:\KP\Pelaksanaan\KEGIATAN KP\WEBSITE\Koefisien-Pajak-Website\model\model_diesel.h5'
model_path_gasoline = 'D:\KP\Pelaksanaan\KEGIATAN KP\WEBSITE\Koefisien-Pajak-Website\model\model_gasoline.h5'

# Define the custom R-squared metric function using tfa.metrics.RSquare()
def custom_r_square(y_true, y_pred):
    # Create an instance of RSquare metric
    rsquare_metric = tfa.metrics.RSquare()

    # Call update_state() to update the metric's state with the current y_true and y_pred
    rsquare_metric.update_state(y_true, y_pred)

    # Return the result using result() method of the RSquare metric
    return rsquare_metric.result()

# Register the custom R-squared metric function
tf.keras.utils.get_custom_objects().update({'custom_r_square': custom_r_square})

# Load the model
model_diesel = load_model(model_path_diesel, custom_objects={"custom_r_square": custom_r_square})
# Load the model
model_gasoline = load_model(model_path_gasoline, custom_objects={"custom_r_square": custom_r_square})

def home():
    # st.sidebar.title(f"Hello {nama}")
    with st.sidebar:
        selected = option_menu(None, ["Prediksi", "Optimasi dan Learning", "Visualisasi Heatmap", "Feature Selection", "Ensemble Learning"],
        icons=['house', 'graph-up', 'gear-fill'], menu_icon="cast", default_index=0)
        
    if (selected == 'Prediksi'):
        selected_option = st.sidebar.radio("Pilih metode prediksi :", ("By script", "By ML"))
        with st.container():
            st.markdown("<h5 style='text-align : center;'>Prediksi Koefisien Pajak Bahan Bakar Diesel/Gasoline</h5>", unsafe_allow_html=True)
            ## horizontal Menu
            selected2 = option_menu(None, ["Diesel", "Gasoline"], 
            icons=['fuel-pump-diesel', 'fuel-pump'], 
            menu_icon="cast", default_index=  0, orientation="horizontal")
                
            if (selected2 == "Diesel" and selected_option == "By ML"):
                tahun = st.number_input("Tahun Pembuatan kendaraan (s/d 2023):")
                opasitas = st.number_input("Opasitas kendaraan (angka 0 - 100):")
                usia = st.number_input("Usia Kendaraan:")
                
                # Membuat dictionary dengan data awal
                data = {
                    'tahun' : [0, 2023],
                    'opasitas': [0, 100],
                    'usia': [0, 73],
                }

                # Membuat data frame
                dfx = pd.DataFrame(data)
                new = {'tahun': tahun, 'opasitas': opasitas, 'usia': usia}
                # Append the new row to the data frame
                dfx = dfx.append(new, ignore_index=True)
                
                #Normalisasi data uji
                # Create an instance of the StandardScaler
                scaler = MinMaxScaler()

                # Fit the scaler to the data
                scaler.fit(dfx)

                # Transform the data
                x_scaled = scaler.transform(dfx)

                dfy = pd.DataFrame({'regrating': [0, 6]})
                # Fit the scaler to the data
                scaler.fit(dfy)
                
                #prediksi data
                y_model_test = model_diesel.predict(x_scaled)
                
                #inverse ke nilai normal
                y_test_denorm = scaler.inverse_transform(y_model_test)
                hasilD = y_test_denorm[-1][0]
                
                hitungD = st.button("PREDIKSI")
                if hitungD:
                    st.write("Hasil prediksi nilai rating bahan bakar diesel : ", hasilD)
                
            if (selected2 == "Gasoline" and selected_option == "By ML"):
                co = st.number_input("Nilai CO kendaraan (0 - 10):")
                hc = st.number_input("Nilai HC kendaraan (0 - 10000):")
                usia = st.number_input("Usia Kendaraan:")
                
                # Membuat dictionary dengan data awal
                data = {
                    'CO': [0, 10],
                    'HC': [0, 10000],
                    'Usia': [0, 73]
                }

                # Membuat data frame
                dfx = pd.DataFrame(data)
                new = {'CO': co, 'HC': hc, 'Usia': usia}
                # Append the new row to the data frame
                dfx = dfx.append(new, ignore_index=True)
                
                # print(dfx)
                #Normalisasi data uji
                # Create an instance of the StandardScaler
                scaler = MinMaxScaler()

                # Fit the scaler to the data
                scaler.fit(dfx)

                # Transform the data
                x_scaled_gasoline = scaler.transform(dfx)
                
                # print(x_scaled_gasoline)

                dfy = pd.DataFrame({'regrating': [0, 11]})
                # Fit the scaler to the data
                scaler.fit(dfy)
                
                #prediksi data
                y_model_test = model_gasoline.predict(x_scaled_gasoline)
                
                # #inverse ke nilai normal
                y_test_denorm = scaler.inverse_transform(y_model_test)
                hasilG = y_test_denorm[-1][0]
                
                hitungG = st.button("PREDIKSI")
                if hitungG:
                    st.write("Hasil prediksi nilai rating bahan bakar gasoline : ", hasilG)
    
    if (selected == 'Optimasi dan Learning'):
        with st.container():
            st.markdown("<h5 style='text-align : center;'>Optimasi Hyperparameter dengan Multilayer Perceptron (MLP)</h5>", unsafe_allow_html=True)
            
            optimasi_func()
             

def landing_page():
    st.image("logo.png", None, use_column_width=True)
    st.markdown("<h1 style='font-size:40px;font-family: 'Courier New';'>SELAMAT DATANG,</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style=''>Web Aplikasi Pemodelan untuk Data Engineering</h2>", unsafe_allow_html=True)
    st.markdown("<h6 style=''>By : PSTB - BRIN team</h6>", unsafe_allow_html=True)
    # click_login = st.button("Login Disini")
    # if click_login:
    #     login()
    login()
                 
def login():
    names = ["Audrina Angela", "Arlo Amstrong"]
    usernames = ["user1", "user2"]
    passwords = ["user1", "user2"]

    # print([names[0]])

    # st.title('Login Page')
    # Add a login form

    username = st.text_input('Username')
    password = st.text_input('Password', type='password')

    if st.button('Login'):
        if username in usernames and password in passwords:
            user_index = usernames.index(username)
            if passwords[user_index] == password:
                st.session_state.is_logged_in = True
            else:
                st.error('Invalid password')
        else:
            st.error('Invalid username')

def main():
    # Initialize the 'is_logged_in' session state variable
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False

    if not st.session_state.is_logged_in:
        landing_page()
    else:
        home()
   
    # landing_page()

if __name__ == '__main__':
    main()
