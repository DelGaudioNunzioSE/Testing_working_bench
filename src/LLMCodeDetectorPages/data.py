import streamlit as st
import numpy as np
import pandas as pd
from LLMCodeDetectorPages.utils.tabel import render_summary_table


st.header("Dataset purpose:")
c1, c2 = st.columns(2)
with c1:
    st.markdown("""**CodeMirage:** is the test to evaluate:
- Language generalization
- LLMs Generalization
- pharafrazed
""")
with c2:
    st.markdown("""**AIGCodeSet:** and **Sun et al:** are the test to evaluate:
- competitive code evaluation
- correct code vs wrong code evaluation
""")
st.divider()



c1, c2 = st.columns([1, 2], vertical_alignment="bottom")
with c1:
    st.header("CodeMirage")
with c2 :
    st.link_button(
        "Go on hugging face",
        "https://www.huggingface.co/datasets/HanxiGuo/CodeMirage",
        type="secondary"
    )
st.text('The work and dataset presented in the CodeMirage publication aim to evaluate ten\
LLM code detection methods. The dataset was also used to train the detectors (at\
least those requiring training). They employed as many as ten different LLMs for code\
generation.')




######





with st.expander("Dataset Tabel", expanded=False):
    html = render_summary_table(
    human_code="10,000",
    llms_code="199,988",
    num_llms_desc="10  LLMs: <em>(GPT-4o-mini, GPT-o3-mini, Qwen-2.5-Coder-32B, Claude-3.5-Haiku, DeepSeek-R1, DeepSeek-V3, Gemini-2.0-Flash, Gemini-2.0-Flash-Thinking, Gemini-2.0-Pro, Llama-3.3-70B)</em>",
    diversity_desc="6 different source models: <em>(GPT, CodeQwen, Claude, DeepSeek, Gemini, Llama)</em>",
    use_period="2024–2025",
    languages_desc="<em>(HTML, JavaScript, Java, C, Python, Ruby, Go, C#, C++, PHP)</em>",
    code_types="unspecified",
    code_size="1<sup>st</sup> percentile: 639 words, 3<sup>rd</sup> percentile: 804 words",
    code_context="open-source",
    prompts="Provided (one)",
    source_human="GitHub",
    code_quality="Statistical alignment",
    reliability="Medium, <em>(no precise references are provided regarding the data source)</em>",
    note="Evaluation Summary: CodeMirage Dataset",
    left_width="50%", right_width="50%"
    )
    st.html(html)