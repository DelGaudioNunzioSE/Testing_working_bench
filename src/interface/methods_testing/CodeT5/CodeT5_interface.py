import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from interface.methods_testing.CodeT5.CodeT5_learning import  CodeT5_learning
from interface.methods_testing.CodeT5.CodeT5_test import  test_model
from datasets import load_dataset
from interface.methods_testing.compute_metrics import auto_compute


from stqdm import stqdm 
from pathlib import Path


st.title('CodeT5')

debug = False  ###############################  <---------------------
st.set_page_config(layout="wide")


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

if "last_csv_path_train" not in st.session_state:
    st.session_state.last_csv_path_train = None
if "last_csv_path_val" not in st.session_state:
    st.session_state.last_csv_path_val = None
if "model" not in st.session_state:
    st.session_state.model = None
if "clicked_train" not in st.session_state:
    st.session_state.clicked_train = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None
if "busy" not in st.session_state: 
    st.session_state.busy = False



col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Upload dataset to train the model", type=["csv"])
    if uploaded:
        df_train = "./temp/CodeT5/train/"
        os.makedirs(df_train,exist_ok=True)

        st.session_state.last_csv_path_train = os.path.join(df_train, "train_dataset.csv")

        # leggi subito
        df = pd.read_csv(uploaded)
        df = df.dropna(subset=["code"])
        df = df.dropna(subset=["cleared_code"])
        st.dataframe(df.head())
        df.to_csv(st.session_state.last_csv_path_train, index=False, encoding="utf-8")


    uploaded2 = st.file_uploader("Upload dataset to train the model", type=["csv"])
    if uploaded2:
        df_train = "./temp/CodeT5/train/"
        os.makedirs(df_train,exist_ok=True)

        st.session_state.last_csv_path_val = os.path.join(df_train, "val_dataset.csv")

        # leggi subito
        df = pd.read_csv(uploaded2)
        df = df.dropna(subset=["code"])
        df = df.dropna(subset=["cleared_code"])
        st.dataframe(df.head())
        df.to_csv(st.session_state.last_csv_path_val, index=False, encoding="utf-8")

    if uploaded and uploaded2:
        st.session_state.clicked_train = st.button("Train", disabled=st.session_state.busy)

with col2:
    if st.session_state.last_csv_path is None:
        uploaded = st.file_uploader("Upload model .bin", type=["bin"])
        if uploaded:
            model = "./temp/CodeT5/train/"
            os.makedirs(model, exist_ok=True)

            model_path = os.path.join(model, "CodeT5.bin")
            i = 0
            while os.path.isfile(model_path):
                i += 1
                model_path = os.path.join(model, "CodeT5" + str(i)+ ".bin")

            with open(model_path, "wb") as f:
                 f.write(uploaded.getbuffer())

            button = st.button("ok", disabled=st.session_state.busy)
            if button:
                st.session_state.model_path = model_path



st.divider()
#############################





if st.session_state.clicked_train :
    st.session_state.busy = True
    st.write("Elaboration...")
    
    
    st.session_state.model_path = CodeT5_learning(train_path=st.session_state.last_csv_path_train, val_path=st.session_state.last_csv_path_val)
    
    st.success("completed")
    st.session_state.busy = False





###################


if st.session_state.model_path:


    with open(st.session_state.model_path, "rb") as f:
        st.download_button(
            label="Download model",
            data=f,
            file_name="CodeT5.bin",
            mime="application/octet-stream"
        )


    uploaded3 = st.file_uploader("Upload test set", type=["csv"])



    if uploaded3 is not None:
        df_test = "./temp/CodeT5/test/"
        os.makedirs(df_test,exist_ok=True)

        df_test = os.path.join(df_test, "dataset_to_test.csv")
        # leggi subito
        df = pd.read_csv(uploaded3)
        df = df.dropna(subset=["code"])
        st.dataframe(df.head())
        df.to_csv(df_test, index=False, encoding="utf-8")




        clicked = st.button("Test", disabled=st.session_state.busy)
        if clicked:
            st.session_state.busy = True
            st.write("Elaboration...")

            ###
            if debug:
                st.text('debug mode enabled')
            y_true, y_score = test_model(test_path=df_test, model_path = st.session_state.model_path)
            ds3 = load_dataset("csv", data_files=df_test, split="train")
            ds3 = ds3.add_column("score", y_score)
            auto_compute(ds= ds3, method = 'CodeT5')


            st.success("completed")
            st.session_state.busy = False



