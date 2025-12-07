import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import interface.methods_testing.methods.BiScope.BiScope as BiScope
from datasets import load_dataset
from interface.methods_testing.utils.compute_metrics import auto_compute, thr_for_fpr

# pip install streamlit stqdm
import streamlit as st, time
import tqdm
from stqdm import stqdm  
tqdm.tqdm = stqdm
from pathlib import Path
from interface.methods_testing.methods.llmppl.mamba import MambaPPL, compute_ppl

# 8.7






st.title('LLMPPL')
mamba = MambaPPL(model_name="state-spaces/mamba-370m-hf")

Setted_threshould2 = 8.7

st.set_page_config(layout="wide")




if "busy" not in st.session_state: 
    st.session_state.busy = False


code_column = st.selectbox("code_column", ["cleared_code", "code" ])


col1, col2 = st.columns([1,1])
with col1:
    Setted_threshould2 = st.number_input(
        "insert threshold",
        min_value=-10.0,
        max_value=10.0,
        value=Setted_threshould2,
        step=0.01,
        format="%.2f"
    )
with col2:
    st.markdown("You can use a dataset to set a threshold")
    uploaded_thresh = st.file_uploader("Upload threshold set", type=["csv"])
    if uploaded_thresh is not None:
        df_train = "./temp/llmppl/train/"
        os.makedirs(df_train,exist_ok=True)
        df = pd.read_csv(uploaded_thresh)
        df_train = os.path.join(df_train, "dataset_to_threshold.csv")
        df.to_csv(df_train, index=False, encoding="utf-8")

        ds = load_dataset("csv", data_files=df_train, split="train")

        ds = ds.map(compute_ppl, fn_kwargs={"mamba":mamba, "k":None, "code_column":code_column})

        y_true = np.asarray(ds["label"]).astype(int)
        ppl    = np.asarray(ds["score"], dtype=float)

        Setted_threshould2 = thr_for_fpr(y_true, ppl, target_fpr=0.10, positive_is_higher=True)
        
        st.success(f"Threshold set to {Setted_threshould2}")


uploaded2 = st.file_uploader("Upload test set", type=["csv"])



if uploaded2 is not None:
    df_test = "./temp/llmppl/test/"
    os.makedirs(df_test,exist_ok=True)

    df_test = os.path.join(df_test, "dataset_to_test.csv")
    # read immediately
    df = pd.read_csv(uploaded2)
    df = df.dropna(subset=[code_column])
    st.dataframe(df.head())
    df.to_csv(df_test, index=False, encoding="utf-8")




    clicked = st.button("Test", disabled=st.session_state.busy)
    if clicked:
        st.session_state.busy = True
        st.write("Elaboration...")

        ###
        dataset = load_dataset("csv", data_files=df_test, split="train")


        dataset_with_ppl = dataset.map(compute_ppl, fn_kwargs={"mamba":mamba, "k":Setted_threshould2, "code_column":code_column})


        auto_compute(ds= dataset_with_ppl, method = 'LLMPPL', opposite=True)


        st.success("completed")
        st.session_state.busy = False



