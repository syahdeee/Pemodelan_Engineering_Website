import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from optimasi_mlp import optimasi_func
from visualisasi import showHeatmap
from prediksi_mlp import predict_mlp
from optimasi_gpr import gpr_model
from preprocessing import preprocessing
import tensorflow as tf
import tensorflow_addons as tfa
import streamlit as st
import time
from prediksi_gpr import predict_gpr
from tensorflow.keras.models import load_model
import pandas as pd

def home():

    with st.sidebar:
        selected = option_menu(None, ["Data Pre-Processing", "Optimasi dan Learning", "Prediksi Model"],
        icons=['graph-up', 'gear-fill', 'bar-chart-fill'], menu_icon="cast", default_index=0)
      
    if (selected == "Data Pre-Processing"):
        preprocessing()   
    if (selected == "Optimasi dan Learning"):
        selected_modelling = option_menu(None, ["MLP", "GPR"], 
            icons=[], 
            menu_icon="cast", default_index=0, orientation="horizontal")
        if selected_modelling == "MLP":
            st.markdown("<h5 style='text-align : center;'>Optimasi Hyperparameter (Bayesian Optimization) dan Modelling dengan Multilayer Perceptron (MLP)</h5>", unsafe_allow_html=True)
            optimasi_func()
        if selected_modelling == "GPR":
            gpr_model()
            
    if (selected == "Prediksi Model"):
        # horizontal Menu
        selected_pred = option_menu(None, ["MLP", "GPR"], 
            icons=[], 
            menu_icon="cast", default_index=0, orientation="horizontal")
        
        if selected_pred == "MLP":
            predict_mlp()
        if selected_pred == "GPR":
            predict_gpr()

def landing_page():
    st.image("logo.png", None, use_column_width=True)
    st.markdown("<h1 style='font-size:40px;font-family: z'Courier New';'>SELAMAT DATANG,</h1>", unsafe_allow_html=True)
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
    # # Initialize the 'is_logged_in' session state variable
    # if 'is_logged_in' not in st.session_state:
    #     st.session_state.is_logged_in = False

    # if not st.session_state.is_logged_in:
    #     landing_page()
    # else:
    #     home()
   
    # landing_page()
    
    home()

if __name__ == '__main__':
    main()
