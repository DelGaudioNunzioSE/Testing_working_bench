import os
import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import shutil

#from interface import *
import streamlit as st

#
# Sidebar: Temp files cleanup
#
with st.sidebar:
    st.divider()
    temp_dir = Path(__file__).parent.parent / "temp"
    if temp_dir.exists():
        # Calculate temp directory size
        total_size = sum(f.stat().st_size for f in temp_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        st.caption(f"📁 Temp files: {size_mb:.1f} MB")
        
        if st.button("🗑️ Clean temp files"):
            try:
                shutil.rmtree(temp_dir)
                temp_dir.mkdir(exist_ok=True)
                st.success("Temp files cleaned!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
#


# pages
BASE = Path(__file__).parent        
# PAGES_DIR = BASE / "LLMCodeDetectorPages"  




#why_rel    = os.path.relpath(BASE / "interface/why.py", start=BASE)
det_rel     = os.path.relpath(BASE / "interface/detector.py", start=BASE)
motivations_rel = os.path.relpath(BASE / "interface/motivations.py", start=BASE)
codemirage_rel = os.path.relpath(BASE / "interface/dataset_analyzer/CodeMirage.py", start=BASE)
coDET_rel = os.path.relpath(BASE / "interface/dataset_analyzer/CoDETM4.py", start=BASE)
AIG_rel = os.path.relpath(BASE / "interface/dataset_analyzer/AIGCodeSet.py", start=BASE)
Pan_rel = os.path.relpath(BASE / "interface/dataset_analyzer/Pan.py", start=BASE)
Uncovering_rel = os.path.relpath(BASE / "interface/dataset_analyzer/UncoveringDataset.py", start=BASE)

BiScope = os.path.relpath(BASE / "interface/methods_testing/methods/BiScope/BiScope_interface.py", start=BASE)
BiScope_standard = os.path.relpath(BASE / "interface/methods_testing/methods/BiScope_standard/BiScope_standard_interface.py", start=BASE)
UncoveringLLM = os.path.relpath(BASE / "interface/methods_testing/methods/UncoveringLLM/UncoveringLLM_interface.py", start=BASE)
CodeT5 = os.path.relpath(BASE / "interface/methods_testing/methods/CodeT5/CodeT5_interface.py", start=BASE)
llmppl = os.path.relpath(BASE / "interface/methods_testing/methods/llmppl/llmppl_interface.py", start=BASE)
Binoculars_standard = os.path.relpath(BASE / "interface/methods_testing/methods/Binoculars_standard/Binoculars_standard.py", start=BASE)
# Definisci le pagine usando st.Page

#why     = st.Page(why_rel, title="Why a LLM Code Detector", icon="❔")
detector_page = st.Page(det_rel, title="Detector", icon="🕵️‍♂️")
motivations_page = st.Page(motivations_rel, title="Motivations", icon="❔")
codemirage_page = st.Page(codemirage_rel, title="CodeMirage", icon="📊")
coDET_page = st.Page(coDET_rel, title="CoDET-M4", icon="📊")
AIG_rel_page = st.Page(AIG_rel, title="AIGCodeSet", icon="📊")
Pan_rel_page = st.Page(Pan_rel, title="Pan", icon="📊")
Uncovering_rel_page = st.Page(Uncovering_rel, title="Uncovering", icon="📊")

BiScope_page = st.Page(BiScope, title="BiScope", icon="🏹")
BiScope_standard_page = st.Page(BiScope_standard, title="BiScope standard", icon="🏹")
UncoveringLLM_page = st.Page(UncoveringLLM, title="UncoveringLLM", icon="✍")
CodeT5_page = st.Page(CodeT5, title="CodeT5", icon="🚨")
llmppl_page = st.Page(llmppl, title="llmppl", icon="🕵️‍♀️")
Binoculars_standard_page = st.Page(Binoculars_standard, title="Binoculars", icon="👀")

# Costruisci il menu con sezioni
pages = {
    "Home": [motivations_page, detector_page],
    "Datasets": [codemirage_page, AIG_rel_page, Pan_rel_page, coDET_page, Uncovering_rel_page],
    "methods": [BiScope_page, BiScope_standard_page, UncoveringLLM_page, CodeT5_page, llmppl_page, Binoculars_standard_page]
}
pg = st.navigation(pages, position='top')

# Esegui solo la pagina selezionata
pg.run()



# streamlit run src/house.py