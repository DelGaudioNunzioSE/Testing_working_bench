import os
import streamlit as st
import numpy as np
import pandas as pd
import io
import contextlib
import time

st.title('LLM Code Detector') #set title's app




# Init stato
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False
    st.session_state["code"] = ""
    st.session_state["lang"] = "python"





if not st.session_state["submitted"]:


    code = st.text_area(
    "Write your code here:", 
    value="""# ============================================
# Generated automatically by a large language model
# Purpose: Demonstrate an extremely verbose
#          implementation of Hello World in Python
# ============================================

def main():
    \"\"\"
    Main entry point of the program.
    Prints a friendly greeting to the console.
    \"\"\"
    # Define the message (hard-coded, of course)
    greeting_message = "Hello, World!"
    
    # Extra unnecessary abstraction
    output_message(greeting_message)

def output_message(msg: str):
    \"\"\"
    Output a given message to standard output.
    Args:
        msg (str): The message to display
    \"\"\"
    # Superfluous try/except for a single print
    try:
        print(msg)
    except Exception as e:
        print(f"Unexpected error while printing message: {e}")

# Standard LLM boilerplate to ensure execution
if __name__ == "__main__":
    main()
""",   # testo predefinito
    height=800
)

    st.caption("Otherwise upload the file")
    uploaded_file = st.file_uploader("Upload file (.py o .c)", type=["py","c"], label_visibility="collapsed")
    


    

    lang = "python"
    if uploaded_file is not None:
        code = uploaded_file.read().decode("utf-8")
        ext = os.path.splitext(uploaded_file.name)[1].lower() # aware the code type from the file
        if ext == ".c":
            lang = "c"

    c1, c2, c3 = st.columns([2,1,2])
    with c2:
        if st.button("Test"):
            st.session_state["submitted"] = True
            st.session_state["code"] = code
            st.session_state["lang"] = lang
            st.rerun()





else:
    # --- output ---
    with st.spinner("Elaboration..."):
        result = st.session_state["code"]  # HOOK <---------------






    def compute_probability(code: str, lang: str) -> float:
        # TODO: rimpiazza con la tua logica
        return min(1.0, max(0.0, len(code) % 101 / 100))  # demo

    prob = compute_probability(st.session_state["code"], st.session_state["lang"])

    st.metric("LLM Probability", f"{prob:.2%}")
    st.progress(min(max(prob, 0.0), 1.0))






    if st.button("Go back"):
        st.session_state["submitted"] = False
        st.session_state["code"] = ""
        st.session_state["lang"] = "python"
        st.rerun()