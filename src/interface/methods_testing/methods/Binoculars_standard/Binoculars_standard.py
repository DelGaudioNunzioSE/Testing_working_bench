import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from binoculars import Binoculars
from datasets import load_dataset
from interface.methods_testing.utils.compute_metrics import auto_compute

# pip install streamlit stqdm
import streamlit as st, time
import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm
from pathlib import Path
from llmppl import MambaPPL

st.title('Binoculars standard')



st.set_page_config(layout="wide")





if "busy" not in st.session_state: 
    st.session_state.busy = False


code_column = st.selectbox("code_column", ["cleared_code", "code" ])


uploaded2 = st.file_uploader("Upload test set", type=["csv"])



if uploaded2 is not None:
    df_test = "./temp/Binoculars_standard/test/"
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


        bino = Binoculars()
        def Binoculars_map(batch, code_column, bino ):
             return {"score": [bino.compute_score(x) for x in batch[code_column]]}
        
        dataset = dataset.map(Binoculars_map, batched=True, fn_kwargs={"code_column": "code", "bino": bino})

        
        auto_compute(ds= dataset, method = 'Binoculars')


        st.success("completed")
        st.session_state.busy = False



