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
