import streamlit as st
import numpy as np
import pandas as pd


import streamlit as st

#
#
# what you write here will apear in all pages
#
#


# Definisci le pagine
home_page = st.Page("./LLMCodeDetectorPages/home.py", title="Home", icon="🏚️", default=True)
detector_page = st.Page("./LLMCodeDetectorPages/detector.py", title="Detector", icon="🕵️‍♂️")
settings_page = st.Page("./LLMCodeDetectorPages/data.py", title="Data", icon="🗃️")

# Crea il menu di navigazione (in sidebar per default)
pages = [home_page, detector_page, settings_page]
pg = st.navigation(pages, position='top')

# Esegui solo la pagina selezionata
pg.run()
