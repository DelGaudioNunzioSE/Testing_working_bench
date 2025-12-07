import streamlit as st
from interface.dataset_analyzer.data import CoDETM4

import altair as alt


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h2 style='color:#E6BA20;margin:0'>CoDET-M4</h2>", unsafe_allow_html=True)
with c2 :
    st.link_button(
        "Go on hugging face 🤗",
        "https://huggingface.co/datasets/DaniilOr/CoDET-M4",
        type="secondary"
    )
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
""")
with c2:
    st.markdown('')




######









st.divider()


c1, c2 = st.columns([1, 1], vertical_alignment="bottom")
with c1:
    st.markdown("<h3 style='color:#E6BA20;margin:0'>CoDET use for test</h3>", unsafe_allow_html=True)
with c2 :
    number = st.number_input(
        "Insert the dimension of the dataset that you prefer",  
        min_value=1000,            #  min
        max_value=100000,          # max 
        step=1000                  
    )
st.text('This dataset should be use only for learning and not for testing.')











with st.spinner("Creating Subset ..."):
    CD = CoDETM4(seed = 30, len_dataset=number)
    st.write(CD.head())





    csv_data = CD.convert_df_to_csv()
    csv_train, csv_val, csv_test = CD.convert_df_to_csv_split()


    st.download_button(
        label="Download like CSV",
        data=csv_data,
        file_name="CoDET.csv",
        mime="text/csv",
        icon="📁",
        width="stretch"
    )



    col1, col2, col3 = st.columns([5, 1, 1], border=False)
    with col1:
        st.download_button(
            label="train_split",
            data=csv_train,
            file_name="CoDET_train.csv",
            mime="text/csv",
            width="stretch"
        )
    with col2:
        st.download_button(
            label="val_split",
            data=csv_val,
            file_name="CoDET_val.csv",
            mime="text/csv",
            width="stretch"
        )
    with col3:
        st.download_button(
            label="test_split",
            data=csv_test,
            file_name="CoDET_test.csv",
            mime="text/csv",
            width="stretch"
        )

    col1, col2 = st.columns(2)

    with col1:
        chart = CD.LLM_count()
        st.altair_chart(chart, use_container_width=True)

    with col2:
        chart= CD.language_count()
        st.altair_chart(chart, use_container_width=True)



    col3, col4 = st.columns(2)

    with col3:
        chart = CD.status_in_folder()
        st.altair_chart(chart, use_container_width=True)

    with col4:
        chart = CD.cleared_code()
        st.altair_chart(chart, use_container_width=True)



