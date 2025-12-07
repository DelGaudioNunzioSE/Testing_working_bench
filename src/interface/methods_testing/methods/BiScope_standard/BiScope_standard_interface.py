import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import interface.methods_testing.methods.BiScope_standard.BiScope as BiScope
from datasets import load_dataset
from interface.methods_testing.utils.compute_metrics import auto_compute

# pip install streamlit stqdm
import streamlit as st, time
import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm


st.title('BiScope_standard')

debug = False  ###############################  <---------------------
st.set_page_config(layout="wide")

if "last_csv_path" not in st.session_state:
    st.session_state.last_csv_path = None
if "clicked_train" not in st.session_state:
    st.session_state.clicked_train = None
if "model_path" not in st.session_state:
    st.session_state.model_path = None
if "busy" not in st.session_state: 
    st.session_state.busy = False


code_column = st.selectbox("code_column", ["cleared_code", "code" ])
use_prompt = st.selectbox("use prompt", ["no", "yes"])
use_prompt = (use_prompt == "yes")

col1, col2 = st.columns([1,1])
with col1:
    uploaded = st.file_uploader("Upload dataset to train the model", type=["csv"])
    if uploaded:
        df_train = "./temp/BiScope_standard/train/"
        os.makedirs(df_train,exist_ok=True)

        st.session_state.last_csv_path = os.path.join(df_train, "dataset.csv")

        # read immediately
        df = pd.read_csv(uploaded)
        df = df.dropna(subset=[code_column])
        st.dataframe(df.head())
        df.to_csv(st.session_state.last_csv_path, index=False, encoding="utf-8")

        st.session_state.clicked_train = st.button("Train", disabled=st.session_state.busy)

with col2:
    if st.session_state.last_csv_path is None:
        uploaded = st.file_uploader("Upload model .joblib", type=["joblib"])
        if uploaded:
            model = "./temp/BiScope_standard/train/"
            os.makedirs(model, exist_ok=True)

            model_path = os.path.join(model, "biscope_standard_rf.joblib")
            i = 0
            while os.path.isfile(model_path):
                i += 1
                model_path = os.path.join(model, "biscope_standard_rf" + str(i)+ ".joblib")

            with open(model_path, "wb") as f:
                 f.write(uploaded.getbuffer())
            st.session_state.model_path = model_path



st.divider()
#############################





if st.session_state.clicked_train :
    st.session_state.busy = True
    st.write("Elaboration...")
    
    
    st.session_state.model_path = BiScope.train("llama2-7b", dataset_path=st.session_state.last_csv_path ,code = code_column, use_prompt= use_prompt, quantization = None, out_dir="./temp/BiScope_standard/train")
    
    st.success("completed")
    st.session_state.busy = False





###################


if st.session_state.model_path:


    with open(st.session_state.model_path, "rb") as f:
        st.download_button(
            label="Download model",
            data=f,
            file_name="biscope_standard_rf.joblib",
            mime="application/octet-stream"
        )


    uploaded2 = st.file_uploader("Upload test set", type=["csv"])



    if uploaded2 is not None:
        df_test = "./temp/BiScope_standard/test/"
        os.makedirs(df_test,exist_ok=True)

        df_test = os.path.join(df_test, "dataset_to_test.csv")
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
            if debug:
                st.text('debug mode enabled')
            y_true, y_score = BiScope.test("llama2-7b", dataset_path=df_test ,code = code_column, use_prompt= use_prompt, model_path= st.session_state.model_path, debug=debug, out_dir="./temp/BiScope_standard/train")
            ds3 = load_dataset("csv", data_files=df_test, split="train")
            ds3 = ds3.add_column("score", y_score)
            auto_compute(ds= ds3, method = 'Biscope_standard', opposite=True)


            st.success("completed")
            st.session_state.busy = False



