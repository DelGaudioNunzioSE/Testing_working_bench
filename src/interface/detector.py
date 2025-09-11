import os
import streamlit as st
import numpy as np
import pandas as pd
from code_editor import code_editor
import plotly.graph_objects as go # pip install plotly
import time


st.title('LLM Code Detector') #set title's app
st.set_page_config(layout="wide", initial_sidebar_state="expanded")









# HOOK function
def compute_probability(code: str, lang: str) -> float:
    time.sleep(2)
    # TODO: rimpiazza con la tua logica
    return 10







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


##


if "lang" not in st.session_state:
    st.session_state.lang = "python"


##


## --- Code section --- 
c1, c2 = st.columns([7,2], vertical_alignment="top")
with c1:

    code = code_editor(code = value, 
                            lang=st.session_state.lang, 
                            key="editor", 
                            theme="monokai", 
                            height = [17.5, 17.5],
                            allow_reset=True)


    col1, col2 = st.columns([15,1], vertical_alignment="bottom")


    with col1:
            st.selectbox("_", ["python", "java"],  label_visibility="collapsed",  key="lang" )

    with col2:
        if st.button("Run"):

            st.session_state.do_compute = True







# --- UI colonna destra ---
with c2:
    placeholder = st.empty() 

    
    if st.session_state.get("do_compute"):

        with placeholder.container():

            with st.spinner("ELABORATION..."):

                code_txt = code["text"] if isinstance(code, dict) else code

                prob = compute_probability(code_txt, st.session_state.lang) # <--------------- HOOK

                ########### ############# ############# ############# ############# ############# #############


                label = "LLM" if prob > 50 else "human"
                color1 = "#E62020" if prob > 50 else "#4ae01d"
                color2 = "#4ae01d" if prob > 50 else "#E62020"
                bg = "#1a1c24"

                fig = go.Figure( # general container for the plot
                    data=[
                        go.Pie(  # type of graph
                            values=[prob, 100 - prob],
                            labels=["LLM", "Human"],
                            hole=0.78, # how big must bhe the hole in the center
                            sort=False,
                            direction="clockwise",
                            marker=dict(colors=[color2, color1], 
                                        line=dict(color=bg, width=2)),
                            textinfo="none",
                            hovertemplate="%{label}<extra></extra>"
                        )],

                    layout=dict(
                        showlegend=False, # no legend
                        margin=dict(l=0, r=0, t=0, b=60),
                        paper_bgcolor=bg,
                        plot_bgcolor=bg,
                        annotations=[
                            dict(
                                text=label,
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(size=28, color=color1)
                                ),
                            dict(
                                text=f" Human: {100-prob}% \tLLM: {prob}% ",  # testo sotto
                                x=0.5, y=-0.1, showarrow=False,
                                xref="paper", yref="paper",    # coordinate rispetto al canvas
                                font=dict(size=16, color="white")
                            )
                        ]

                    )
                )


                st.plotly_chart(fig, use_container_width=True)
                ############# ############# ############# ############# ############# ############# #############

                st.write(f"{st.session_state.lang}")

        # reset del flag per evitare recompute ad ogni rerun
        st.session_state.do_compute = False





st.divider()

st.header('Why a LLM Code Detector')
##########
st.subheader(" 1° Academic and Professional integrity")
st.text('The undeclared use of LLMs in evaluation contexts makes the assessment process highly problematic.  \
Students, for example, may complete assignments or even exams using LLMs\
without contributing meaningfully to the generated code, potentially impairing their\
learning of fundamental programming concepts (getting the most out with the least\
effort). \n\
Similarly, during a technical interview, a candidate might rely on a tool like\
AlphaCode 2 to generate solutions to proposed problems.')
st.markdown("Reference: [The Impact of Large Language Models on Programming Education and Student Learning Outcomes ](https://www.mdpi.com/2076-3417/14/10/4115)")





st.subheader(" 2° Vulnerable or Inefficient")

st.text("Another critical issue lies in the way LLMs generate code. Since these models are\
trained on massive corpora, evaluating the security and efficiency of the generated code\
is nearly impossible.\n\
The output may be vulnerable or inefficient due to subtle\
flaws that are hard to detect, especially when the code appears well-formatted and\
logically structured. This happens to overreliance on commonly seen patterns in the\
training corpus rather than more appropriate niche solutions.")
st.markdown("Reference: [Do Users Write More Insecure Code with AI Assistants? ](https://arxiv.org/abs/2211.03622)")



st.subheader("3° Intellectual property")
st.text("Intellectual property is yet another motivation for detecting LLM-generated\
code, a general problem in generative AI. \n\
This is not limited to the origin of training\
data but also concerns the risk that an LLM may reproduce copyrighted code,\
posing significant legal risks to software companies that could unknowingly integrate\
such code into their products.")
st.markdown("Reference: [Doe v. GitHub – Order on Motion to Dismiss ](https://law.justia.com/cases/federal/district-courts/california/candce/4:2022cv06823/403220/195/)")



st.subheader("4° LLM training")
st.text("The least important reason, which directly concerns the development of codeoriented\
LLMs themselves, is the need to distinguish machine-generated code in training \
datasets. \n\
If a model is trained on LLMs’ code we might run into a general LLM code\
oriented Model Collapse. Indeed, training LLM on LLMs’ code serves to deteriorate\
the output diversity and adaptability to real-world scenarios.")
st.markdown("Reference: [The Curse of Recursion: Training on Generated Data Makes Models Forget ](https://arxiv.org/abs/2305.17493)")




















