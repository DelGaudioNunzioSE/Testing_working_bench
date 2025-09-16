import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
from interface import *


import streamlit as st

#
#
# what you write here will apear in all pages
#
#


# pages
BASE = Path(__file__).parent        
# PAGES_DIR = BASE / "LLMCodeDetectorPages"  




#why_rel    = os.path.relpath(BASE / "interface/why.py", start=BASE)
det_rel     = os.path.relpath(BASE / "interface/detector.py", start=BASE)
codemirage_rel = os.path.relpath(BASE / "interface/dataset/CodeMirage.py", start=BASE)
coDET_rel = os.path.relpath(BASE / "interface/dataset/CoDETM4.py", start=BASE)
AIG_rel = os.path.relpath(BASE / "interface/dataset/AIGCodeSet.py", start=BASE)
Pan_rel = os.path.relpath(BASE / "interface/dataset/Pan.py", start=BASE)

BiScope = os.path.relpath(BASE / "interface/methods_testing/BiScope/BiScope_interface.py", start=BASE)
UncoveringLLM = os.path.relpath(BASE / "interface/methods_testing/UncoveringLLM/UncoveringLLM_interface.py", start=BASE)
CodeT5 = os.path.relpath(BASE / "interface/methods_testing/CodeT5/CodeT5_interface.py", start=BASE)
llmppl = os.path.relpath(BASE / "interface/methods_testing/llmppl/llmppl_interface.py", start=BASE)

# Definisci le pagine usando st.Page

#why     = st.Page(why_rel, title="Why a LLM Code Detector", icon="❔")
detector_page = st.Page(det_rel, title="Detector", icon="🕵️‍♂️")
codemirage_page = st.Page(codemirage_rel, title="CodeMirage", icon="📊")
coDET_page = st.Page(coDET_rel, title="CoDET-M4", icon="📊")
AIG_rel_page = st.Page(AIG_rel, title="AIGCodeSet", icon="📊")
Pan_rel_page = st.Page(Pan_rel, title="Pan", icon="📊")
BiScope_page = st.Page(BiScope, title="BiScope", icon="👀")
UncoveringLLM_page = st.Page(UncoveringLLM, title="UncoveringLLM", icon="👀")
CodeT5_page = st.Page(CodeT5, title="CodeT5", icon="👀")
llmppl_page = st.Page(llmppl, title="llmppl", icon="👀")

# Costruisci il menu con sezioni
pages = {
    "Home": [detector_page],
    "Datasets": [codemirage_page, AIG_rel_page, Pan_rel_page, coDET_page],
    "methods": [BiScope_page, UncoveringLLM_page, CodeT5_page, llmppl_page]
}
pg = st.navigation(pages, position='top')

# Esegui solo la pagina selezionata
pg.run()



# streamlit run src/house.py