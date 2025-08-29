import streamlit as st
import numpy as np
import pandas as pd



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






