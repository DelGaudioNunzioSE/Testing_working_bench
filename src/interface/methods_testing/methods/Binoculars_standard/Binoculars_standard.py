import os
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import sys
import io
import warnings

from interface.methods_testing.methods.Binoculars_standard.binoculars import *
from datasets import load_dataset
from interface.methods_testing.utils.compute_metrics import auto_compute

# pip install streamlit stqdm
import streamlit as st, time
import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm
from pathlib import Path


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
        
        # Load dataset
        with st.spinner("Loading dataset..."):
            dataset = load_dataset("csv", data_files=df_test, split="train")
        
        # Initialize Binoculars with loading message and capture output
        # Create two containers: one for status messages, one for detailed output
        status_container = st.empty()  # Dynamic container that can be updated
        output_container = st.expander("Model Loading Details", expanded=True)  # Collapsible expander for details
        
        with output_container:
            # === REDIRECT STDOUT AND STDERR ===
            # Save original references to stdout/stderr to restore them later
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Create in-memory buffers to capture output
            stdout_buffer = io.StringIO()  # Buffer for stdout (normal prints)
            stderr_buffer = io.StringIO()  # Buffer for stderr (warnings, progress bars)
            
            try:
                # REDIRECT: all print() and warnings will go to buffers
                sys.stdout = stdout_buffer
                sys.stderr = stderr_buffer
                
                # Show status message (this DOES go to screen because it uses st.info)
                status_container.info("🔄 Loading Binoculars models (Falcon-7B)... This may take a few minutes...")
                
                # MODEL LOADING: all output (warnings, progress bars) is captured in buffers
                bino = Binoculars()
                
                # Update status message to success
                status_container.success("✅ Binoculars models loaded successfully!")
                
            finally:
                # === RESTORE AND DISPLAY ===
                # IMPORTANT: always restore stdout/stderr, even if there are errors
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                
                # Retrieve all text captured from buffers
                stdout_text = stdout_buffer.getvalue()  # Get content from stdout
                stderr_text = stderr_buffer.getvalue()  # Get content from stderr (warnings are here)
                
                # Display captured output in the expander
                if stderr_text:
                    st.text("Loading Messages:")
                    st.text(stderr_text)  # Display warnings, progress bars, etc.
                if stdout_text:
                    st.text(stdout_text)  # Display any print() statements
        
        st.info("Processing dataset with Binoculars...")
        
        def Binoculars_map(batch, code_column, bino):
            return {"score": [bino.compute_score(x) for x in batch[code_column]]}
        
        # Process with progress bar
        dataset = dataset.map(Binoculars_map, batched=True, fn_kwargs={"code_column": code_column, "bino": bino})
        
        with st.spinner("Computing metrics..."):
            auto_compute(ds=dataset, method='Binoculars')

        st.success("completed")
        del bino # erase model from memory
        st.session_state.busy = False



