import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import interface.methods_testing.BiScope as BiScope



# pip install streamlit stqdm
import streamlit as st, time
from stqdm import stqdm  # barra compatibile Streamlit
from pathlib import Path







UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

if "last_csv_path" not in st.session_state:
    st.session_state.last_csv_path = None

uploaded = st.file_uploader("Carica un CSV", type=["csv"])
df_train = "./temp/BiScope/train/"
os.makedirs(df_train,exist_ok=True)

df_train = os.path.join(df_train, "dataset.csv")

if uploaded is not None:
    # leggi subito
    df = pd.read_csv(uploaded)
    df = df.dropna(subset=["code"])
    df = df.dropna(subset=["cleared_code"])
    st.dataframe(df.head())
    df.to_csv(df_train, index=False, encoding="utf-8")





#############################










if "busy" not in st.session_state: 
    st.session_state.busy = False

clicked = st.button("Train", disabled=st.session_state.busy)
if clicked:
    st.session_state.busy = True
    st.write("Elaboration...")
    
    
    BiScope.train("CodeLlama", dataset_path=df_train ,clear_code = True, use_prompt= True)
    
    st.success("Completato")
    st.session_state.busy = False





st.divider()



uploaded2 = st.file_uploader("Carica un CSV2", type=["csv"])
df_test = "./temp/BiScope/test/"
os.makedirs(df_test,exist_ok=True)

df_test = os.path.join(df_test, "dataset.csv")

if uploaded2 is not None:
    # leggi subito
    df = pd.read_csv(uploaded2)
    df = df.dropna(subset=["code"])
    df = df.dropna(subset=["cleared_code"])
    st.dataframe(df.head())
    df.to_csv(df_test, index=False, encoding="utf-8")




clicked = st.button("Test", disabled=st.session_state.busy)
if clicked:
    st.session_state.busy = True
    st.write("Elaboration...")

    ###
    y_true, y_score = BiScope.test("CodeLlama", dataset_path=df_test ,clear_code = True, use_prompt= True)

    fpr, tpr, _ = roc_curve(y_true, y_score)        # calcolo
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC")
    ax.legend(loc="lower right")

    st.pyplot(fig)  # mostra in Streamlit
    ###


    st.success("Completato")
    st.session_state.busy = False




