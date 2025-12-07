import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from interface.methods_testing.methods.CodeT5.CodeT5_learning import  CodeT5_learning
from interface.methods_testing.methods.CodeT5.CodeT5_test import  test_model
from datasets import load_dataset
from interface.methods_testing.utils.compute_metrics import auto_compute
from sklearn.model_selection import train_test_split
import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm
from pathlib import Path


st.title('CodeT5')

st.set_page_config(layout="wide")


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



# st.set_option("server.maxUploadSize", 1024)  streamlit run script.py --server.maxUploadSize 1024
code_column = st.selectbox("code_column", ["cleared_code", "code" ])
LoRa = st.selectbox("LoRa", ["no", "yes" ])
LoRa : bool = (LoRa == "yes" )
col1, col2 = st.columns([1,3])
quantize = st.selectbox('quantization affect test only', ["not", "8-bit" ])
quantize : bool = (quantize == "8-bit")

col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Upload dataset to train the model", type=["csv"])
    if uploaded:
        df_train = "./temp/CodeT5/train/"
        os.makedirs(df_train,exist_ok=True)

        st.session_state.last_csv_path_train = os.path.join(df_train, "train_dataset.csv")
        st.session_state.last_csv_path_val = os.path.join(df_train, "val_dataset.csv")

        # leggi subito
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        belance_conlum = df["LLM"]
        train_df, val_df  = train_test_split(df, test_size=0.1, random_state=42)

        train_df.to_csv(st.session_state.last_csv_path_train, index=False, encoding="utf-8")
        val_df.to_csv(st.session_state.last_csv_path_val, index=False, encoding="utf-8")

        st.session_state.clicked_train = st.button("Train")

with col2:
    if st.session_state.last_csv_path_train  is None and st.session_state.last_csv_path_val  is None:
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

            st.session_state.model_path = model_path



st.divider()
#############################





if st.session_state.clicked_train :
    st.write("Elaboration...")
    
    
    st.session_state.model_path = CodeT5_learning(train_path=st.session_state.last_csv_path_train, val_path=st.session_state.last_csv_path_val, use_lora = LoRa)
    
    st.success("completed")





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
        st.dataframe(df.head())
        df.to_csv(df_test, index=False, encoding="utf-8")




        clicked = st.button("Test")
        if clicked:
            st.write("Elaboration...")
            print('hi')

            ###
                
            y_true, y_score = test_model(test_path=df_test, model_path = st.session_state.model_path, quantize = quantize, use_lora = LoRa)
            ds3 = load_dataset("csv", data_files=df_test, split="train")
            ds3 = ds3.add_column("score", y_score)
            auto_compute(ds= ds3, method = 'CodeT5')


            st.success("completed")



