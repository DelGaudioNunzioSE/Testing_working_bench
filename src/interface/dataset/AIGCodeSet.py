import streamlit as st
from utils.tabel import render_summary_table
from tests.Dataset.data import AIG

import altair as alt


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h2 style='color:#0000CD;margin:0'>AIGCodeSet</h2>", unsafe_allow_html=True)
with c2 :
    st.link_button(
        "Go on hugging face 🤗",
        "https://huggingface.co/datasets/basakdemirok/AIGCodeSet",
        type="secondary"
    )
c1, c2 = st.columns(2)
with c1:
    st.markdown("""**AIGCodeSet:** is the test to evaluate:
- competitive code evaluation
- correct code vs wrong code evaluation
- difference between difficulty 
""")
with c2:
    st.markdown('Unlike other works, this dataset includes both functional and non-\
functional human-written code, as it evaluates LLM code generation starting both from\
a plain prompt and from existing code.')




######





with st.expander("Original dataset Table", expanded=False):
    html = render_summary_table(
        human_code="9,510",
        llms_code="5,650",
        num_llms_desc="3 <em>Gemini 1.5 Flash, Codestral-22B, CodeLlama-34B</em>",
        diversity_desc="3 different source models",
        use_period="Average LLMs release date: 2024 (now exist Gemini-2.0)",
        languages_desc="Python",
        code_types="unspecified",
        code_size="1<sup>st</sup> percentile: 30 words, 3<sup>rd</sup> percentile: 50 words",
        code_context="competitive",
        prompts="provided",
        source_human='CodeNet by IBM <a href="https://github.com/IBM/Project_CodeNet" target="_blank">[64]</a>',
        code_quality="Executable check",
        reliability="Hight",
        note="Evaluation Summary: AIGCodeSet Dataset",
        left_width="50%", right_width="50%"
    )
    st.html(html)



st.divider()

number = 1000
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h3 style='color:#4169E1;margin:0'>AIGCodeSet use for test</h3>", unsafe_allow_html=True)
with c2 :
    number = st.number_input(
        "Insert the dimension of the dataset that you prefer",  # etichetta
        min_value=100,            # valore minimo (opzionale)
        max_value=1000,          # max value
        step=100,                  # incremento predefinito
        value=1000
    )
st.text('To perform quick but indicative tests of code quality, a split of only 1000 code samples\
(500 human and 500 LLM) was taken, while maintaining language balance as much as possible.\n\
you can take any split you want.')











with st.spinner("Creating Subset ..."):
    A = AIG(seed = 30, len_dataset=number)
    st.write(A.head())




    csv_data = A.convert_df_to_csv()
    csv_train, csv_val, csv_test = A.convert_df_to_csv_split()


    st.download_button(
        label="Download like CSV",
        data=csv_data,
        file_name="AIGCodeSet.csv",
        mime="text/csv",
        icon="📁",
        width="stretch"
    )

    col1, col2, col3 = st.columns([5, 1, 1], border=False)
    with col1:
        st.download_button(
            label="train_split",
            data=csv_train,
            file_name="AIGCodeSet_train.csv",
            mime="text/csv",
            width="stretch"
        )
    with col2:
        st.download_button(
            label="val_split",
            data=csv_val,
            file_name="AIGCodeSet_val.csv",
            mime="text/csv",
            width="stretch"
        )
    with col3:
        st.download_button(
            label="test_split",
            data=csv_test,
            file_name="AIGCodeSet_test.csv",
            mime="text/csv",
            width="stretch"
        )



    col1, col2 = st.columns(2)

    with col1:
        chart = A.LLM_count('steelblue')
        st.altair_chart(chart, use_container_width=True)

    with col2:
        chart= A.language_count('royalblue')
        st.altair_chart(chart, use_container_width=True)



    col3, col4 = st.columns(2)

    with col3:
        chart = A.status_in_folder('cornflowerblue')
        st.altair_chart(chart, use_container_width=True)

    with col4:
        chart = A.cleared_code('skyblue')
        st.altair_chart(chart, use_container_width=True)



