import streamlit as st
from interface.dataset_analyzer.data import CodeMirage

import altair as alt

@st.cache_data(show_spinner="Creating CodeMirage subset...")
def create_codemirage_dataset(len_dataset, seed):
    """Create and preprocess CodeMirage dataset - cached to avoid recomputation"""
    return CodeMirage(seed=seed, len_dataset=len_dataset)


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




st.divider()

number = 1000
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h3 style='color:#FF4500;margin:0'>CodeMirage use for test</h3>", unsafe_allow_html=True)
with c2 :
    number = st.number_input(
        "Insert the dimension of the dataset that you prefer",  # etichetta
        min_value=1000,            # valore minimo (opzionale)
        max_value=100000,          # max value
        step=1000                  # incremento predefinito
    )
st.text('This dataset, while large, only includes GitHub code that was originally heavily commented.')

# Use cached dataset creation - only recomputes if len_dataset or seed changes
CM = create_codemirage_dataset(len_dataset=number, seed=30)

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
    chart = CM.LLM_count(color='#E02F4E')
    st.altair_chart(chart, use_container_width=True)

with col2:
    chart= CM.language_count(color='#E05014')
    st.altair_chart(chart, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    chart = CM.status_in_folder(color='#E0452F')
    st.altair_chart(chart, use_container_width=True)

with col4:
    chart = CM.cleared_code(color='#E08779')
    st.altair_chart(chart, use_container_width=True)



