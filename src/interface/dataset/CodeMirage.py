import streamlit as st
from utils.tabel import render_summary_table
from tests.Dataset.data import CodeMirage

import altair as alt


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h2 style='color:#e53935;margin:0'>CodeMirage</h2>", unsafe_allow_html=True)
with c2 :
    st.link_button(
        "Go on hugging face 🤗",
        "https://www.huggingface.co/datasets/HanxiGuo/CodeMirage",
        type="secondary"
    )
c1, c2 = st.columns(2)
with c1:
    st.markdown("""**CodeMirage:** is the test to evaluate:
- Language generalization
- LLMs Generalization
""")
with c2:
    st.markdown('The work and dataset presented in the CodeMirage publication aim to evaluate ten\
LLM code detection methods. The dataset was also used to train the detectors (at\
least those requiring training). They employed as many as ten different LLMs for code\
generation.')




######





with st.expander("Original dataset Tabel", expanded=False):
    html = render_summary_table(
    human_code="10,000",
    llms_code="199,988",
    num_llms_desc="10  LLMs: <em>(GPT-4o-mini, GPT-o3-mini, Qwen-2.5-Coder-32B, Claude-3.5-Haiku,\
          DeepSeek-R1, DeepSeek-V3, Gemini-2.0-Flash, Gemini-2.0-Flash-Thinking, Gemini-2.0-Pro, Llama-3.3-70B)</em>",
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



st.divider()

number = 1000
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h3 style='color:#FF4500;margin:0'>CodeMirage use for test</h3>", unsafe_allow_html=True)
with c2 :
    number = st.number_input(
        "Insert the dimension of the dataset that you prefer",  # etichetta
        min_value=1000,            # valore minimo (opzionale)
        max_value=60000,          # max value
        step=1000                  # incremento predefinito
    )
st.text('To perform quick but indicative tests of code quality, a split of only 1000 code samples\
(500 human and 500 LLM) was taken, while maintaining language balance as much as possible.\n\
you can take any split you want.')











with st.spinner("Creating Subset ..."):
    CM = CodeMirage(seed = 30, len_dataset=number)
    st.write(CM.head())





    csv_data = CM.convert_df_to_csv()
    csv_train, csv_val, csv_test = CM.convert_df_to_csv_split()


    st.download_button(
        label="Download like CSV",
        data=csv_data,
        file_name="CodeMirage.csv",
        mime="text/csv",
        icon="📁",
        width="stretch"
    )



    col1, col2, col3 = st.columns([5, 1, 1], border=False)
    with col1:
        st.download_button(
            label="train_split",
            data=csv_train,
            file_name="CodeMirage_train.csv",
            mime="text/csv",
            width="stretch"
        )
    with col2:
        st.download_button(
            label="val_split",
            data=csv_val,
            file_name="CodeMirage_val.csv",
            mime="text/csv",
            width="stretch"
        )
    with col3:
        st.download_button(
            label="test_split",
            data=csv_test,
            file_name="CodeMirage_test.csv",
            mime="text/csv",
            width="stretch"
        )

    col1, col2 = st.columns(2)

    with col1:
        chart = CM.LLM_count()
        st.altair_chart(chart, use_container_width=True)

    with col2:
        chart= CM.language_count()
        st.altair_chart(chart, use_container_width=True)



    col3, col4 = st.columns(2)

    with col3:
        chart = CM.status_in_folder()
        st.altair_chart(chart, use_container_width=True)

    with col4:
        chart = CM.cleared_code()
        st.altair_chart(chart, use_container_width=True)



