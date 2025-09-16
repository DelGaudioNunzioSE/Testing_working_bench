import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from interface.methods_testing.UncoveringLLM.UncoveringLLM import Analyzer, map_functionAnalyzer
from interface.methods_testing.compute_metrics import auto_compute
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

# pip install streamlit stqdm
import streamlit as st, time
from stqdm import stqdm  # barra compatibile Streamlit
from pathlib import Path



st.title('UncoveringLLM')


if "last_csv_path" not in st.session_state:
    st.session_state.last_csv_path = None
if "busy" not in st.session_state:
    st.session_state.busy = False




uploaded2 = st.file_uploader("Upload dataset", type=["csv"])
df_test = "./temp/UncoveringLLM/test/"
os.makedirs(df_test,exist_ok=True)

df_test = os.path.join(df_test, "dataset.csv")

if uploaded2 is not None:

    st.markdown('Your dataset must contain at least the following columns: **code**, **code_rewrited**, **code_rewrited2**.')

    df = pd.read_csv(uploaded2)
    df = df.dropna(subset=["code"])
    df = df.dropna(subset=["code_rewrited"])
    df = df.dropna(subset=["code_rewrited2"])
    st.dataframe(df.head())
    df.to_csv(df_test, index=False, encoding="utf-8")
    st.session_state.last_csv_path = df_test
    st.session_state.busy = False




    clicked = st.button("Test", disabled=st.session_state.busy)
    if st.session_state.last_csv_path is not None and clicked:
        st.session_state.busy = True
        st.write("Elaboration...")

        ###
        ds = load_dataset("csv", data_files=df_test, split='train')
        Analyzer_obj = Analyzer()
        ds3 = ds.map(map_functionAnalyzer, fn_kwargs={"analyzer": Analyzer_obj})

        auto_compute(ds= ds3, method = 'UncoveringLLM')
        

        ###


        st.success("finisched")
        st.session_state.busy = False

    elif clicked and st.session_state.last_csv_path is None:
        st.text('Please before upload the csv')




