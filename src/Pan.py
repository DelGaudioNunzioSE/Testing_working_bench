import streamlit as st
from utils.tabel import render_summary_table
from tests.Dataset.data import Pan

import altair as alt


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h2 style='color:#32CD32;margin:0'>Pan</h2>", unsafe_allow_html=True)
with c2 :
    None
c1, c2 = st.columns(2)
with c1:
    st.markdown("""**Pan:** is the test to evaluate:
- correct code vs wrong code evaluation
""")
with c2:
    st.markdown('The goal of the original paper is is to demonstrate, beyond any doubt, that detection\
methods commonly used for natural language are not reliable when applied to code.\
To achieve this, the authors construct a dedicated dataset.')




######





with st.expander("Original dataset Table", expanded=False):
    html = render_summary_table(
        human_code="5,069",
        llms_code="65,897",
        num_llms_desc="1 <em>GPT</em>",
        diversity_desc="1",
        use_period="unspecified",
        languages_desc="Python",
        code_types="unspecified",
        code_size="1<sup>st</sup> percentile: 54 words, 3<sup>rd</sup> percentile: 83 words",
        code_context="competitive",
        prompts="provided",
        source_human="Quescol, Kaggle",
        code_quality="filtered by human",
        reliability="Peer-review paper",
        note="Table 3.4: Evaluation Summary: Pan-et-al Dataset",
        left_width="60%", right_width="40%"
    )
    st.html(html)



st.divider()

number = 1000
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h3 style='color:#006400;margin:0'>Pan use for test</h3>", unsafe_allow_html=True)
with c2 :
    number = st.number_input(
        "Insert the dimension of the dataset that you prefer",  # etichetta
        min_value=100,            # valore minimo (opzionale)
        max_value=272,          # max value
        step=50                  # incremento predefinito
    )

link_url = "https://github.com/DelGaudioNunzioSE/LLM-CodeTester"  # cambia qui

st.markdown(
    f'A subset of code from the original dataset was extracted and tested using the <a href="{link_url}" target="_blank">framework</a>.',
    unsafe_allow_html=True,
)












P = Pan(seed = 30, len_dataset=number)
st.write(P.head())




csv_data = P.convert_df_to_csv()


st.download_button(
    label="Download like CSV",
    data=csv_data,
    file_name="Sun.csv",
    mime="text/csv",
    icon="📁",
    width="stretch"
)



col1, col2 = st.columns(2)

with col1:
    chart = P.LLM_count('seagreen')
    st.altair_chart(chart, use_container_width=True)

with col2:
    chart= P.language_count('mediumaquamarine')
    st.altair_chart(chart, use_container_width=True)



col3, col4 = st.columns(2)

with col3:
    chart = P.status_in_folder('springgreen')
    st.altair_chart(chart, use_container_width=True)

with col4:
    chart = P.cleared_code('darkolivegreen')
    st.altair_chart(chart, use_container_width=True)



