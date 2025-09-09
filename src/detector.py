import os
import streamlit as st
import numpy as np
import pandas as pd
from code_editor import code_editor
import plotly.graph_objects as go


st.title('LLM Code Detector') #set title's app










# HOOK function
def compute_probability(code: str, lang: str) -> float:
    # TODO: rimpiazza con la tua logica
    return 100







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
"""



c1, c2 = st.columns([4,1], vertical_alignment="center")
with c1:

    lang = st.selectbox("Lang", ["python", "java"])

    editor_btns = [{
    "name": "Run",
    "feather": "Play",
    "primary": True,
    "hasText": True,
    "showWithIcon": True,
    "commands": ["submit"],
    "style": {"bottom": "0.44rem", "right": "0.4rem"}
    }]

    resp = code_editor(code = value, 
                        lang=lang, 
                        key="editor", 
                        theme="monokai", 
                        height = [10, 15],
                        allow_reset=True,
                        buttons=editor_btns)

with c2:
    # --- output ---
    with st.spinner("Elaboration..."):

        prob = compute_probability(resp, lang) # <------- HOOK

        label = "LLM" if prob > 50 else "human"
        color = "tomato" if prob > 50 else "mediumturquoise"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': f" {label}"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': 'lightgray'},
                    {'range': [50, 100], 'color': 'lightgray'}
                ],
                'threshold': None
            },
            domain={'x': [0, 1], 'y': [0, 1]},
        ))
        st.plotly_chart(fig, use_container_width=True)





























