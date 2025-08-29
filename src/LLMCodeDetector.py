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
PAGES_DIR = BASE / "LLMCodeDetectorPages"  




home_rel = os.path.relpath(PAGES_DIR / "home.py", start=BASE)
det_rel  = os.path.relpath(PAGES_DIR / "detector.py", start=BASE)
data_rel = os.path.relpath(PAGES_DIR / "data.py", start=BASE)

home_page     = st.Page(home_rel, title="Home", icon="🏚️", default=True)
detector_page = st.Page(det_rel,  title="Detector", icon="🕵️‍♂️")
settings_page = st.Page(data_rel, title="Data", icon="🗃️")

# Crea il menu di navigazione (in sidebar per default)
pages = [home_page, detector_page, settings_page]
pg = st.navigation(pages, position='top')

# Esegui solo la pagina selezionata
pg.run()
