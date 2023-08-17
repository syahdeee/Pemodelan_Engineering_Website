import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_squared_error
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tkinter import filedialog
import tkinter as tk
import streamlit as st


def normalisasi_input(input):
    # Create an instance of the StandardScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the data
    scaler.fit(input)
    # Transform the data
    x_scaled = scaler.transform(input)

    return x_scaled


def normalisasi_output(output):
    scaler = MinMaxScaler()

    # Fit the scaler to the data
    scaler.fit(output)

    # Fit the scaler to the data
    scaler.fit(output)
    # Transform the data
    y_scaled = scaler.transform(output)

    return y_scaled


def get_best_hyperparameter(
    params_nn2, inputs_opt_act, X_train, X_test, y_train, y_test, num_classes
):
    activations = inputs_opt_act[0]
    optimizers = inputs_opt_act[1]

    # Create function
    def nn_cl_bo2(
        neurons,
        activation,
        optimizer,
        learning_rate,
        batch_size,
        epochs,
        num_layers,
        dropout,
    ):
        optimizerL = optimizers
        activationL = activations
        neurons = round(neurons)
        dropout = round(dropout)
        activation = activationL[round(activation)]
        batch_size = round(batch_size)
        epochs = round(epochs)
        num_layers = round(num_layers)

        optimizerD = {}
        # Loop through the optimizers list and create the optimizer instances in the optimizerD dictionary
        for optimizer_name in optimizers:
            if optimizer_name == "Adam":
                optimizerD[optimizer_name] = Adam(learning_rate=learning_rate)
            elif optimizer_name == "SGD":
                optimizerD[optimizer_name] = SGD(learning_rate=learning_rate)
            elif optimizer_name == "RMSprop":
                optimizerD[optimizer_name] = RMSprop(learning_rate=learning_rate)
            elif optimizer_name == "Adadelta":
                optimizerD[optimizer_name] = Adadelta(learning_rate=learning_rate)
            elif optimizer_name == "Adagrad":
                optimizerD[optimizer_name] = Adagrad(learning_rate=learning_rate)
            elif optimizer_name == "Adamax":
                optimizerD[optimizer_name] = Adamax(learning_rate=learning_rate)
            elif optimizer_name == "Nadam":
                optimizerD[optimizer_name] = Nadam(learning_rate=learning_rate)
            elif optimizer_name == "Ftrl":
                optimizerD[optimizer_name] = Ftrl(learning_rate=learning_rate)

        optimizer = optimizerD[optimizerL[round(optimizer)]]

        def nn_cl_fun():
            nn = Sequential()
            nn.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))

            for i in range(num_layers):
                if i == 0:
                    nn.add(
                        Dense(
                            neurons, input_dim=X_train.shape[1], activation=activation
                        )
                    )
                    nn.add(
                        Dropout(dropout)
                    )  # Create the Dropout layer instance and add it to the model
                else:
                    nn.add(Dense(neurons, activation=activation))
                    nn.add(
                        Dropout(dropout)
                    )  # Create the Dropout layer instance and add it to the model

            nn.add(Dense(num_classes, activation="sigmoid"))
            nn.compile(
                loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"],
            )
            return nn

        es = EarlyStopping(monitor="loss", patience=20, verbose=0)
        nn = nn_cl_fun()
        nn.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[es],
        )
        y_pred = nn.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        accuracy = accuracy_score(y_test_classes, y_pred_classes)

        return accuracy

    tuner = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)

    tuner.maximize(init_points=5, n_iter=2)
    # Get the best parameters
    params_nn_ = tuner.max["params"]

    return params_nn_


def get_model(params_nn_, X_train, num_classes):
    nn = Sequential()
    nn.add(
        Dense(
            params_nn_["neurons"],
            input_dim=X_train.shape[1],
            activation=params_nn_["activation"],
        )
    )

    for i in range(params_nn_["num_layers"]):
        if i == 0:
            nn.add(
                Dense(
                    params_nn_["neurons"],
                    input_dim=X_train.shape[1],
                    activation=params_nn_["activation"],
                )
            )
            if params_nn_["dropout"] > 0.5:
                nn.add(
                    Dropout(0.1, seed=123)
                )  # Add Dropout layer with 0.1 dropout rate and seed
            else:
                nn.add(
                    Dense(params_nn_["neurons"], activation=params_nn_["activation"])
                )
                nn.add(
                    Dropout(0.1, seed=123)
                )  # Add Dropout layer with 0.1 dropout rate and seed

    nn.add(Dense(num_classes, activation="sigmoid"))

    nn.compile(
        loss="categorical_crossentropy",
        optimizer=params_nn_["optimizer"],
        metrics=["accuracy"],
    )

    return nn


def print_model_summary(model):
    # Capture the model summary in a string variable
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)

    # Display the model summary in Streamlit using st.code()
    st.code(model_summary)


def optimasi_klasifikasi():
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload data dalam format xlsx/xls/csv", type=["xlsx", "xls", "csv"]
    )
    display_button = st.button("Display Dataset")

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file, sep=",")
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(
                f"Error: Unable to read the file. Please make sure it's a valid Excel or CSV file. Exception: {e}"
            )
            st.stop()

    if display_button and "df" in locals():
        # df = df[:100]
        st.write("Dataset:")
        st.write(df)

    if uploaded_file is not None:
        options = df.columns
        # Elemen input ditempatkan di kolom pertama ()
        input_nfeatures = st.number_input(
            "Masukkan jumlah feature pada dataset yang akan dilakukan learning: ", 0
        )

        # List untuk menyimpan fitur
        features_input = []

        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(input_nfeatures):
            selected_option = st.selectbox(f"Feature {i+1}:", options)
            features_input.append(selected_option)

        feature_output = st.selectbox(
            "Masukkan feature output pada dataset yang akan dilakukan learning :",
            options,
        )

        st.write("Masukkan nilai range jumlah neurons yang akan dilakukan optimasi:")

        # Input nilai range min dan max n_neuron
        col1, col2 = st.columns(2)

        # Elemen input min n_neuron
        with col1:
            min_Nneurons = st.number_input("Range minimal jumlah neurons:", 0)

        # Elemen input max n_neuron
        with col2:
            max_Nneurons = st.number_input("Range maksimal jumlah neurons:", 0)

        # Input nilai aktivasi yang akan digunakan
        activations = []
        option_activations = [
            "relu",
            "sigmoid",
            "softplus",
            "softsign",
            "tanh",
            "selu",
            "elu",
            "exponential",
        ]
        input_activations = st.number_input(
            "Masukkan jumlah fungsi aktivasi yang akan dilakukan optimasi: ", 0
        )
        # Tambahkan teks input ke dalam list sesuai dengan input_nfeatures
        for i in range(input_activations):
            selected_activation = st.selectbox(
                f"Fungsi aktivasi {i+1}:", option_activations
            )
            activations.append(selected_activation)

        # Input nilai optimizers yang akan digunakan
        optimizers = []
        option_optimizers = [
            "SGD",
            "Adam",
            "RMSprop",
            "Adadelta",
            "Adagrad",
            "Adamax",
            "Nadam",
            "Ftrl",
        ]
        input_optimizers = st.number_input(
            "Masukkan jumlah optimizer yang akan dilakukan optimasi: ", 0
        )
        for i in range(input_optimizers):
            selected_optimizers = st.selectbox(f"Optimizer {i+1}:", option_optimizers)
            optimizers.append(selected_optimizers)

        # Input range learning rate
        st.write("Masukkan nilai range learning rate yang akan dilakukan optimasi:")

        # Input nilai range min dan max learningRate
        col1, col2 = st.columns(2)

        # Elemen input min_learningRate
        with col1:
            min_learningRate = st.number_input("Range minimal learning rate:")

        # Elemen input max n_neuron
        with col2:
            max_learningRate = st.number_input("Range maksimal learning rate:")

        # Input range batch size
        st.write("Masukkan nilai range batch size yang akan dilakukan optimasi:")

        # Input nilai range min dan max batch size
        col1, col2 = st.columns(2)

        # Elemen input min_batchSize
        with col1:
            min_batchSize = st.number_input("Range minimal batch size:", 0)

        # Elemen input max_batchSize
        with col2:
            max_batchSize = st.number_input("Range maksimal batch size:", 0)

        # Input range jumlah epochs
        st.write("Masukkan nilai range jumlah epochs yang akan dilakukan optimasi:")

        # Input nilai range min dan max jumlah epochs
        col1, col2 = st.columns(2)

        # Elemen input min_nEpochs
        with col1:
            min_nEpochs = st.number_input("Range minimal jumlah epochs:", 0)

        # Elemen input max_nEpochs
        with col2:
            max_nEpochs = st.number_input("Range maksimal jumlah epochs:", 0)

        # Input range jumlah epochs
        st.write("Masukkan nilai range jumlah layers yang akan dilakukan optimasi:")

        # Input nilai range min dan max jumlah layers
        col1, col2 = st.columns(2)

        # Elemen input min_nLayers
        with col1:
            min_nLayers = st.number_input("Range minimal jumlah layers:", 0)

        # Elemen input max_nLayers
        with col2:
            max_nLayers = st.number_input("Range maksimal jumlah layers:", 0)

        # Input range dropout rate
        st.write("Masukkan nilai range dropout rate yang akan dilakukan optimasi:")

        # Input nilai range min dan max dropout rate
        col1, col2 = st.columns(2)

        # Elemen input min_dropout
        with col1:
            min_dropout = st.number_input("Range minimal dropout rate:")

        # Elemen input max_dropout
        with col2:
            max_dropout = st.number_input("Range maksimal dropout rate:")

        testSize = st.number_input("Masukkan test size : ")

        # Dict input hyperparameter
        hyperparams = {
            "neurons": (min_Nneurons, max_Nneurons),
            "activation": (0, len(activations) - 1),
            "optimizer": (0, len(optimizers) - 1),
            "learning_rate": (min_learningRate, max_learningRate),
            "batch_size": (min_batchSize, max_batchSize),
            "epochs": (min_nEpochs, max_nEpochs),
            "num_layers": (min_nLayers, max_nLayers),
            "dropout": (min_dropout, max_dropout),
        }

        inputs_opt_act = [activations, optimizers]

        if st.button("Next"):
            X = df[features_input]
            y = np.array(df[feature_output])
            num_classes = df[feature_output].nunique()

            # Normalisasi input dan output
            x_scaled = normalisasi_input(X)
            le = LabelEncoder()
            y_encode = le.fit_transform(y)

            # One-hot encode the target classes
            onehot_encoder = OneHotEncoder(sparse=False)
            y_onehot = onehot_encoder.fit_transform(y_encode.reshape(-1, 1))

            X_train, X_test, y_train, y_test = train_test_split(
                x_scaled, y_onehot, test_size=testSize, random_state=42
            )

            params_nn_ = get_best_hyperparameter(
                hyperparams,
                inputs_opt_act,
                X_train,
                X_test,
                y_train,
                y_test,
                num_classes,
            )

            learning_rate = params_nn_["learning_rate"]
            activationL = activations
            params_nn_["activation"] = activationL[round(params_nn_["activation"])]
            params_nn_["batch_size"] = round(params_nn_["batch_size"])
            params_nn_["epochs"] = round(params_nn_["epochs"])
            params_nn_["num_layers"] = round(params_nn_["num_layers"])
            params_nn_["neurons"] = round(params_nn_["neurons"])
            params_nn_["dropout"] = round(params_nn_["dropout"])
            optimizerL = optimizers
            optimizerD = {}
            # Loop through the optimizers list and create the optimizer instances in the optimizerD dictionary
            for optimizer_name in optimizers:
                if optimizer_name == "Adam":
                    optimizerD[optimizer_name] = Adam(learning_rate=learning_rate)
                elif optimizer_name == "SGD":
                    optimizerD[optimizer_name] = SGD(learning_rate=learning_rate)
                elif optimizer_name == "RMSprop":
                    optimizerD[optimizer_name] = RMSprop(learning_rate=learning_rate)
                elif optimizer_name == "Adadelta":
                    optimizerD[optimizer_name] = Adadelta(learning_rate=learning_rate)
                elif optimizer_name == "Adagrad":
                    optimizerD[optimizer_name] = Adagrad(learning_rate=learning_rate)
                elif optimizer_name == "Adamax":
                    optimizerD[optimizer_name] = Adamax(learning_rate=learning_rate)
                elif optimizer_name == "Nadam":
                    optimizerD[optimizer_name] = Nadam(learning_rate=learning_rate)
                elif optimizer_name == "Ftrl":
                    optimizerD[optimizer_name] = Ftrl(learning_rate=learning_rate)

            params_nn_["optimizer"] = optimizerD[
                optimizerL[round(params_nn_["optimizer"])]
            ]

            st.write("Berikut adalah hyperparameter terbaik hasil optimasi : ")
            st.write("Activation : ", params_nn_["activation"])
            st.write("Batch Size : ", params_nn_["batch_size"])
            st.write("Jumlah epochs : ", params_nn_["epochs"])
            st.write("Neurons : ", params_nn_["neurons"])
            st.write("Jumlah Layers : ", params_nn_["num_layers"])
            st.write("Optimizer : ", params_nn_["optimizer"])
            st.write("Dropout : ", params_nn_["dropout"])

            # # Get model
            model = get_model(params_nn_, X_train, num_classes)

            print_model_summary(model)

            es = EarlyStopping(monitor="loss", verbose=0, patience=20)

            history = model.fit(
                X_train,
                y_train,
                epochs=params_nn_["epochs"],
                callbacks=[es],
                validation_split=0.2,
                batch_size=params_nn_["batch_size"],
            )

            def make_plot(train, validation, title):
                graph = plt.plot(history.history[train])
                graph = plt.plot(history.history[validation])
                graph = plt.title(title)
                graph = plt.legend(["training", "validation"])
                graph = plt.show()
                return graph

            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            mse = mean_squared_error(y_test_classes, y_pred_classes)

            # Display the R2 score
            st.write("Nilai MSE:", mse)

            # Calculate and display additional metrics
            micro_f1 = f1_score(y_test_classes, y_pred_classes, average="micro")
            macro_f1 = f1_score(y_test_classes, y_pred_classes, average="macro")
            weighted_f1 = f1_score(y_test_classes, y_pred_classes, average="weighted")
            precision = precision_score(
                y_test_classes, y_pred_classes, average="weighted"
            )
            recall = recall_score(y_test_classes, y_pred_classes, average="weighted")

            st.subheader("Additional Metrics")
            st.write("Accuracy:", accuracy)
            st.write("Micro F1 Score:", micro_f1)
            st.write("Macro F1 Score:", macro_f1)
            st.write("Weighted F1 Score:", weighted_f1)
            st.write("Precision:", precision)
            st.write("Recall:", recall)

            # st.write(classification_report(y_test_classes, y_pred_classes))

            plot_accuracy = make_plot(
                "accuracy",
                "val_accuracy",
                "Perubahan Accuracy pada tiap Epoch",
            )
            st.pyplot(plot_accuracy)

            plot_loss = make_plot("loss", "val_loss", "Perubahan Loss pada tiap Epoch")
            st.pyplot(plot_loss)

            try:
                root = tk.Tk()
                root.withdraw()
                file_dialog = tk.Toplevel(root)
                file_dialog.attributes("-topmost", True)
                file_path = filedialog.asksaveasfilename(
                    parent=file_dialog, defaultextension=".h5"
                )
                if file_path:
                    model.save(file_path)
                    st.success(f"Model saved successfully at: {file_path}")
                    file_dialog.destroy()
            except Exception as e:
                st.error("An error occurred while saving the model: " + str(e))


optimasi_klasifikasi()
