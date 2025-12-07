import os
import sys


import streamlit as st
import numpy as np
import pandas as pd
from code_editor import code_editor
import plotly.graph_objects as go # pip install plotly
import time
from interface.methods_testing.methods.CodeT5.CodeT5_test import test_model
from interface.methods_testing.methods.CodeT5.sniffer.code_cleaner import comment_remover, newline_remover, import_remover
from interface.methods_testing.methods.CodeT5.sniffer.gptsniffer import CodeT5pClassifier
from interface.methods_testing.methods.CodeT5.sniffer.dataset import CodeDataset
import torch
import torch.nn.functional as F



st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title('LLM Code Detector') #set title's app




@st.cache_resource(show_spinner=True)
def load_model(path: str, device: str):
    m = CodeT5pClassifier(quantize=False, use_lora=False)
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    m.load_state_dict(sd)
    m.to(device)
    m.eval()
    return m

# Build absolute path: /user/ndelgaudio/Testing_working_bench/src/model/CodeT5.bin
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # /user/.../src/interface/
MODEL_PATH = os.path.abspath(os.path.join(CURRENT_DIR, '..', 'model', 'CodeT5.bin'))
print(f"🔍 Looking for model at: {MODEL_PATH}")  # Debug: mostra dove cerca

with st.spinner('🔄 Loading CodeT5 model... Please wait'):
    model = load_model(MODEL_PATH, 'cuda')  # <-- does not reload on every detection

def test_single_string(code: str, language: str, 
                       model = model):

    # Preprocess single example
    code = comment_remover(code, language)
    code = import_remover(code, language)
    code = newline_remover(code)

    # Prepare input tensor (depends on tokenizer used in CodeT5pClassifier)
    inputs = model.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        print(f"code:{code}")
        inputs = inputs
        logits = model(**inputs)["logits"]
        probs = F.softmax(logits, dim=1)
        score = probs[0, 1].item() 

    return score






# HOOK function
def compute_probability(code: str, lang: str) -> float:
    time.sleep(2)
    score = test_single_string(code, language=lang, model=model)  # score in [0,1]
    prob_pct = float(np.clip(score * 100.0, 0.0, 100.0)) 
    return prob_pct







value="""#insert code here"""




if "lang" not in st.session_state:
    st.session_state.lang = "python"
if "do_compute" not in st.session_state:
    st.session_state.do_compute = False


##


## --- Code section --- 
c1, c2 = st.columns([7,2], vertical_alignment="top")
with c1:

    code = code_editor(code = value, 
                            lang=st.session_state.lang, 
                            key="editor", 
                            theme="monokai", 
                            height = [17.5, 17.5],
                            allow_reset=True,
                            response_mode=["blur", "debounce"])  # <--- essential


    col1, col2 = st.columns([8,1], vertical_alignment="bottom")


    with col1:
            st.selectbox("_", ["python", "java", 'C++', 'C'],  label_visibility="collapsed",  key="lang" )

    with col2:
        if st.button("Run"):
            st.session_state.run_code = code["text"] if isinstance(code, dict) else code
            st.session_state.do_compute = True








# --- UI colonna destra ---
with c2:
    placeholder = st.empty() 

    
    if st.session_state.get("do_compute"):

        with placeholder.container():

            with st.spinner("PROCESSING..."):

                code_txt = st.session_state.get("run_code", "")

                prob = compute_probability(code_txt, st.session_state.lang) # <--------------- HOOK
                prob = round(float(prob), 0)   # force scalar and round to integer

                ########### ############# ############# ############# ############# ############# #############

                
                label = "LLM" if prob > 50 else "human"
                color1 = "#E62020" #if prob > 50 else "#4ae01d"
                color2 = "#4ae01d" #if prob > 50 else "#E62020"
                color_label = "#E62020" if prob > 50 else "#4ae01d"
                bg = "#1a1c24"


                fig = go.Figure( # general container for the plot
                    data=[
                        go.Pie(  # graph type
                            values=[prob, 100 - prob],
                            labels=["LLM", "Human"],
                            hole=0.78, # hole size in the center
                            sort=False,
                            direction="clockwise",
                            marker=dict(colors=[color1, color2], 
                                        line=dict(color=bg, width=2)),
                            textinfo="none",
                            hovertemplate="%{label}<extra></extra>"
                        )],

                    layout=dict(
                        showlegend=False, # no legend
                        margin=dict(l=0, r=0, t=0, b=60),
                        paper_bgcolor=bg,
                        plot_bgcolor=bg,
                        annotations=[
                            dict(
                                text=label,
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(size=28, color=color_label)
                                ),
                            dict(
                                text=f" Human: {100-prob}% \tLLM: {prob}% ",  # text below
                                x=0.5, y=-0.1, showarrow=False,
                                xref="paper", yref="paper",    # coordinates relative to canvas
                                font=dict(size=16, color="white")
                            )
                        ]

                    )
                )


                st.plotly_chart(fig, use_container_width=True)

                html = f'FPR@10%: <span style="color:{color_label}; font-weight:bold">{label}</span>'
                st.markdown(html, unsafe_allow_html=True)
                ############# ############# ############# ############# ############# ############# #############

                

        # reset flag to avoid recomputing on every rerun
        st.session_state.do_compute = False






















