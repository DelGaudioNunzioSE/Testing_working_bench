import streamlit as st
import numpy as np
import pandas as pd

random_number = np.random.randint(low=0,high=100)
st.text(f'Here is your random number: {random_number}')


if st.button("Refresh"):
    st.write("you clicked me")