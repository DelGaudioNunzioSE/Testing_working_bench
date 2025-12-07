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



st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title('Why a LLM Code Detector') #set title's app




##########
st.subheader(" 1° Academic and Professional integrity")
st.text('The undeclared use of LLMs in evaluation contexts makes the assessment process highly problematic.  \
Students, for example, may complete assignments or even exams using LLMs\
without contributing meaningfully to the generated code, potentially impairing their\
learning of fundamental programming concepts (getting the most out with the least\
effort). \n\
Similarly, during a technical interview, a candidate might rely on a tool like\
AlphaCode 2 to generate solutions to proposed problems.')
st.markdown("Reference: [The Impact of Large Language Models on Programming Education and Student Learning Outcomes ](https://www.mdpi.com/2076-3417/14/10/4115)")





st.subheader(" 2° Vulnerable or Inefficient")

st.text("Another critical issue lies in the way LLMs generate code. Since these models are\
trained on massive corpora, evaluating the security and efficiency of the generated code\
is nearly impossible.\n\
The output may be vulnerable or inefficient due to subtle\
flaws that are hard to detect, especially when the code appears well-formatted and\
logically structured. This happens to overreliance on commonly seen patterns in the\
training corpus rather than more appropriate niche solutions.")
st.markdown("Reference: [Do Users Write More Insecure Code with AI Assistants? ](https://arxiv.org/abs/2211.03622)")



st.subheader("3° Intellectual property")
st.text("Intellectual property is yet another motivation for detecting LLM-generated\
code, a general problem in generative AI. \n\
This is not limited to the origin of training\
data but also concerns the risk that an LLM may reproduce copyrighted code,\
posing significant legal risks to software companies that could unknowingly integrate\
such code into their products.")
st.markdown("Reference: [Doe v. GitHub – Order on Motion to Dismiss ](https://law.justia.com/cases/federal/district-courts/california/candce/4:2022cv06823/403220/195/)")



st.subheader("4° LLM training")
st.text("The least important reason, which directly concerns the development of codeoriented\
LLMs themselves, is the need to distinguish machine-generated code in training \
datasets. \n\
If a model is trained on LLMs’ code we might run into a general LLM code\
oriented Model Collapse. Indeed, training LLM on LLMs’ code serves to deteriorate\
the output diversity and adaptability to real-world scenarios.")
st.markdown("Reference: [The Curse of Recursion: Training on Generated Data Makes Models Forget ](https://arxiv.org/abs/2305.17493)")




















