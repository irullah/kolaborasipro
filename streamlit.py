import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib


def peramalan(n_pred, dataset_test, tanggal):
    tanggal = tanggal[0].split("-")
    tahun = tanggal[0]
    bulan = tanggal[1]
    tahun = int(tahun)
    bulan = int(bulan)

    last = dataset_test.tail(1)
    fitur = last.values
    n_fit = len(fitur[0])
    fiturs = np.zeros((n_pred, n_fit))
    tanggals = []
    preds = np.zeros(n_pred)
    for i in range(n_pred):
        if i == 0:
            fitur = fitur[:, 1:n_fit]
            y_pred = lr.predict(fitur)
            new_fit = np.array(fitur[0])
            new_fit = np.append(new_fit, y_pred)
        else:
            fitur = fiturs[i - 1][1:n_fit]
            y_pred = lr.predict([fitur])
            new_fit = np.array(fitur)
            new_fit = np.append(new_fit, y_pred)
        preds[i] = y_pred
        fiturs[i, :] = new_fit
        bulan += 1
        if bulan > 12:
            tahun += 1
            bulan = 1
        tanggal = str(tahun) + "-" + f"{bulan:02d}"
        tanggals.append(tanggal)
    #     print(preds)
    #     print(fiturs)
    return preds, tanggals


# Mengimpor model dan text dari file yang telah diekspor sebelumnya
lr = joblib.load("modelLR.pkl")
df = pd.read_csv(
    "Monthly_Gold_Price_on_World.csv", usecols=["Date", "Indonesian rupiah (IDR)"]
)
scaler = joblib.load("ModelScaler.pkl")
dataset_test = pd.read_csv("data-test.csv")

# Judul aplikasi
st.markdown(
    "<h1 style='text-align: center;'>Forecasting Gold Price Data Time Series</h1>",
    unsafe_allow_html=True,
)
# st.title('Forecasting Temperature Anomalies Data Time Series')

# Define a range for the slider
min_value = 0
max_value = 60

# Create a slider widget
selected_value = st.slider("Select a value", min_value, max_value)

# Display the selected value
st.write("You selected:", selected_value)

# Tombol untuk memprediksi
if st.button("Run"):
    if selected_value:
        col1, col2 = st.columns(2)
        tanggal_terakhir = df["Date"].tail(1).values
        pred, tanggal = peramalan(selected_value, dataset_test, tanggal_terakhir)
        # print(pred)
        # print(tahun)
        reshaped_data = pred.reshape(-1, 1)
        original_data = scaler.inverse_transform(reshaped_data)
        print(len(original_data))
        print(pred)
        pred = original_data.flatten()
        df_pred = pd.DataFrame({"Date": tanggal, "Indonesian rupiah (IDR)": pred})
        # st.dataframe(df_pred)
        # st.write('Hasil Prediksi:', prediction)

        # Plot data df
        fig, ax = plt.subplots()
        plt.plot(df["Date"], df["Indonesian rupiah (IDR)"], label="Data Awal")

        # Plot data df_pred
        plt.plot(df_pred["Date"], df_pred["Indonesian rupiah (IDR)"], label="Prediksi")

        # Menghubungkan plot terakhir data awal dengan plot awal data prediksi
        last_Date = df["Date"].iloc[-1]
        plt.plot(
            [last_Date, df_pred["Date"].iloc[0]],
            [
                df["Indonesian rupiah (IDR)"].iloc[-1],
                df_pred["Indonesian rupiah (IDR)"].iloc[0],
            ],
            "k--",
        )

        # Konfigurasi plot
        plt.xlabel("Tanggal")
        plt.ylabel("Indonesian Rupiah (IDR)")
        plt.title("Perbandingan Data Awal dan Prediksi")
        plt.legend()

        # Tampilkan plot
        # st.pyplot(fig)
        with col1:
            st.dataframe(df_pred)
        with col2:
            st.pyplot(fig)
    else:
        st.write("Masukkan teks terlebih dahulu")