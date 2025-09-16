import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import interface.methods_testing.BiScope.BiScope as BiScope
from datasets import load_dataset
from interface.methods_testing.compute_metrics import auto_compute

# pip install streamlit stqdm
import streamlit as st, time
from stqdm import stqdm  # barra compatibile Streamlit
from pathlib import Path
from llmppl import MambaPPL

st.title('LLMPPL')



st.set_page_config(layout="wide")





if "busy" not in st.session_state: 
    st.session_state.busy = False


code_column = st.selectbox("code_column", ["code", "cleared_code"])

k = st.number_input(
    "insert threshold",
    min_value=-10.0,
    max_value=10.0,
    value=-0.26,
    step=0.01,
    format="%.2f"
)

uploaded2 = st.file_uploader("Upload test set", type=["csv"])


mamba = MambaPPL(model_name="state-spaces/mamba-370m-hf")



if uploaded2 is not None:
    df_test = "./temp/llmppl/test/"
    os.makedirs(df_test,exist_ok=True)

    df_test = os.path.join(df_test, "dataset_to_test.csv")
    # leggi subito
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

        def compute_ppl(ex):
            code = ex[code] 
            ppl = mamba.calculate_ppl(code)
            if ppl < k:
                score = 0
            else:
                score = 1
            return {"score": score}
        dataset_with_ppl = dataset.map(compute_ppl, num_proc=4)


        auto_compute(ds= dataset_with_ppl, method = 'LLMPPL')


        st.success("completed")
        st.session_state.busy = False



