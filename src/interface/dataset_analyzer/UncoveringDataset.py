import streamlit as st
from interface.dataset_analyzer.data import Uncovering

import altair as alt


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h2 style='color:#D800FF;margin:0'>Uncovering</h2>", unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown("""**Uncovering dataset:** is to:
- use for training
""")
with c2:
    st.markdown('This is the dataset used in the UncoveringLLM work and employed to train some models during the tests.')




######





st.divider()

number = 1000
c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h3 style='color:#FF00FF;margin:0'>Uncovering use for test</h3>", unsafe_allow_html=True)
with c2 :
    number = st.number_input(
        "Insert the dimension of the dataset that you prefer",  # etichetta
        min_value=1000,            # valore minimo (opzionale)
        max_value=3000,          # max value
        step=100,                  # incremento predefinito
        value= 1000
    )

link_url = "https://github.com/DelGaudioNunzioSE/LLM-CodeTester"  # cambia qui













with st.spinner("Creating Subset ..."):
    P = Uncovering(seed = 30, len_dataset=number)
    st.write(P.head())




    csv_data = P.convert_df_to_csv()
    


    st.download_button(
        label="Download like CSV",
        data=csv_data,
        file_name="Uncovering.csv",
        mime="text/csv",
        icon="📁",
        width="stretch"
    )



    col1, col2 = st.columns(2)

    with col1:
        chart = P.LLM_count('#AC00FF')
        st.altair_chart(chart, use_container_width=True)

    with col2:
        chart= P.language_count('#DA00FF')
        st.altair_chart(chart, use_container_width=True)



    col3, col4 = st.columns(2)

    with col3:
        chart = P.status_in_folder('#8500FF')
        st.altair_chart(chart, use_container_width=True)

    with col4:
        chart = P.cleared_code('#FF00AC')
        st.altair_chart(chart, use_container_width=True)



