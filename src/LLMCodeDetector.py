import os
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd


import streamlit as st

#
#
# what you write here will apear in all pages
#
#


# pages
BASE = Path(__file__).parent        
# PAGES_DIR = BASE / "LLMCodeDetectorPages"  



home_rel    = os.path.relpath(BASE / "home.py", start=BASE)
why_rel    = os.path.relpath(BASE / "why.py", start=BASE)
det_rel     = os.path.relpath(BASE / "detector.py", start=BASE)
data_rel    = os.path.relpath(BASE / "data.py", start=BASE)
codemirage_rel = os.path.relpath(BASE / "CodeMirage.py", start=BASE)
AIG_rel = os.path.relpath(BASE / "AIGCodeSet.py", start=BASE)

# Definisci le pagine usando st.Page
home     = st.Page(home_rel, title="Hi", icon="👋", default=True)
why     = st.Page(why_rel, title="Why a LLM Code Detector", icon="❔")
detector_page = st.Page(det_rel, title="Detector", icon="🕵️‍♂️")
data_page     = st.Page(data_rel, title="Data overview", icon="🗃️")
codemirage_page = st.Page(codemirage_rel, title="CodeMirage", icon="📊")
AIG_rel_page = st.Page(AIG_rel, title="AIGCodeSet", icon="📊")

# Costruisci il menu con sezioni
pages = {
    "Home": [home, why],
    "Detector": [detector_page],
    "Datasets": [data_page,codemirage_page,AIG_rel_page] # <-- CodeMirage come sottopagina
}
pg = st.navigation(pages, position='top')

# Esegui solo la pagina selezionata
pg.run()
