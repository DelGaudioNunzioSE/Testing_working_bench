import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from interface.methods_testing.methods.UncoveringLLM.UncoveringLLM import Analyzer, map_functionAnalyzer
from interface.methods_testing.utils.compute_metrics import auto_compute
from datasets import load_dataset
from datasets import Dataset

# pip install streamlit stqdm
import streamlit as st, time
import tqdm
from stqdm import stqdm
tqdm.tqdm = stqdm
import re

def comment_remover(code, language="python"):
    if code is None:
        print('Input code is None')
        return

    language = language.lower()

    if language == "python":
        pattern = re.compile(     
            r'""".*?"""'                  # triple double-quoted string
            r"|'''.*?'''"                 # triple single-quoted string
            r'|#.*?$',                    # single-line comment starting with #
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language in ("c", "cpp", "c++", "go", "csharp"):  #cpp is an other way to say c++
        pattern = re.compile(
            r'//.*?$'                     # single-line comment //
            r'|/\*.*?\*/'                 # multi-line comment /* ... */
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string (char literal or escaped)
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|`[^`]*`',                  # raw string in Go using backticks
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language in ("csharp", "c#"):
        pattern = re.compile(
            r'//.*?$'                     # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|@\"(?:[^\"]|\"\")*\"',     # verbatim string @"..." ("" is escape)
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "java":
        pattern = re.compile(
            r'/\*\*.*?\*/'                # Javadoc comment /** ... */
            r'|//.*?$'                    # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|"""(?:.|\n)*?"""',         # text block (Java 15+)
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "javascript":
        pattern = re.compile(
            r'/\*\*.*?\*/'                # JSDoc comment /** ... */
            r'|//.*?$'                    # single-line comment
            r'|/\*.*?\*/'                 # multi-line comment
            r"|'(?:\\.|[^\\'])*'"         # single-quoted string
            r'|"(?:\\.|[^\\"])*"'         # double-quoted string
            r'|`(?:\\.|[^\\`])*`',        # template literal
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, lambda m: '' if m.group(0).startswith('/') else m.group(0), code)

    elif language == "ruby":
        # =begin / =end must be at the beginning of the line (no spaces before)
        pattern = re.compile(
            r'^=begin.*?^=end$'           # multi-line block comment
            r'|#.*?$',                    # single-line comment
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language == "php":
        pattern = re.compile(
            r'//.*?$'                     # single-line comment //
            r'|/\*.*?\*/'                 # multi-line comment /* ... */
            r'|#.*?$',                    # single-line comment starting with #
            re.DOTALL | re.MULTILINE
        )
        code = re.sub(pattern, '', code)

    elif language == "html":
        pattern = re.compile(
            r'<!--.*?-->', re.DOTALL # HTML comment <!-- ... -->
        )  
        code = re.sub(pattern, '', code)

    else:
        print(f"{language} is not supported")

    if code is None:
        raise ValueError('Output code is None')
    
    
    return code




def newline_remover(code):

    code = re.sub(r'^\s*\n', '', code, flags=re.MULTILINE)

    return code




def import_remover(code, language="python"):
    if code is None:
        return
    language = language.lower()

    if language == "python":
        # 
        code = re.sub(r'(?ms)^\s*(?:from\s+[A-Za-z_][\w\.]*\s+import\s*\([\s\S]*?\)\s*'
                      r'|from\s+[A-Za-z_][\w\.]*\s+import\s+[^\n#]+'
                      r'|import\s+[^\n#]+)\s*$', '', code)

    elif language in ("c", "cpp", "c++"):
        code = re.sub(r'(?m)^\s*#\s*include\s*[<"].*[>"].*$', '', code)

    elif language in ("csharp", "c#"):
        code = re.sub(r'(?m)^\s*using\s+[\w\.]+(?:\s*=\s*[\w\.]+)?\s*;\s*$', '', code)

    elif language == "java":
        code = re.sub(r'(?m)^\s*import\s+(?:static\s+)?[\w\.]+(?:\.\*)?\s*;\s*$', '', code)

    elif language in ("javascript", "typescript", "ts", "jsx", "tsx"):
        # import ...;  + require(...)
        code = re.sub(r'(?m)^\s*import\s+[^;]*;?\s*$', '', code)
        code = re.sub(r'(?m)^\s*(?:const|let|var)\s+[\w$]+\s*=\s*require\([^)]*\)\s*;?\s*$', '', code)
        code = re.sub(r'(?m)^\s*require\([^)]*\)\s*;?\s*$', '', code)

    elif language == "ruby":
        code = re.sub(r'(?m)^\s*require(?:_relative)?\s+.+$', '', code)

    elif language == "php":
        code = re.sub(r'(?m)^\s*use\s+[^;]+;\s*$', '', code)
        code = re.sub(r'(?m)^\s*(?:require|require_once|include|include_once)\s*\([^)]*\)\s*;\s*$', '', code)

    elif language == "go":
        # import (...)
        code = re.sub(r'(?ms)^\s*import\s*\(\s*[\s\S]*?\)\s*', '', code)
        code = re.sub(r'(?m)^\s*import\s+["\'][^"\']+["\']\s*$', '', code)

    elif language == "html":
        pass  # no import

    else:
        print(f"{language} is not supported")

    return code

st.title('UncoveringLLM')


if "last_csv_path" not in st.session_state:
    st.session_state.last_csv_path = None
if "busy" not in st.session_state:
    st.session_state.busy = False


#code_column = st.selectbox("code_column", ["cleared_code", "code" ])


code_column = st.selectbox("do you want to clear the code?", ["yes", "no"])
code_column : bool = (code_column=='yes')

uploaded2 = st.file_uploader("Upload dataset", type=["csv"])
df_test = "./temp/UncoveringLLM/test/"
os.makedirs(df_test,exist_ok=True)

df_test = os.path.join(df_test, "dataset.csv")

if uploaded2 is not None:

    st.markdown('Your dataset must contain at least the following columns: **code**, **code_rewrited**, **code_rewrited2**.')

    df = pd.read_csv(uploaded2)
    if code_column:
        ds = Dataset.from_pandas(df)     
        if 'language' in ds.column_names:
            all_codes = [import_remover(comment_remover(x, language), language)  for x, language in zip(ds["code"], ds["language"])]
        else:
             all_codes = [import_remover(comment_remover(x, 'python'), 'python')  for x in ds["code"]]
        ds = ds.remove_columns(['code'])
        ds = ds.add_column("code", all_codes)
        ds = ds.filter(lambda ex: isinstance(ex.get("code"), str) and ex["code"] != "")
        df = ds.to_pandas()
    df = df.dropna(subset=['code'])
    st.dataframe(df.head())
    df.to_csv(df_test, index=False, encoding="utf-8")
    st.session_state.last_csv_path = df_test
    st.session_state.busy = False




    clicked = st.button("Test", disabled=st.session_state.busy)
    if st.session_state.last_csv_path is not None and clicked:
        st.session_state.busy = True
        st.write("Elaboration...")

        ###
        ds = load_dataset("csv", data_files=df_test, split='train')
        Analyzer_obj = Analyzer(code_conlum = 'code')
        ds3 = ds.map(map_functionAnalyzer, fn_kwargs={"analyzer": Analyzer_obj})

        auto_compute(ds= ds3, method = 'UncoveringLLM')
        

        ###


        st.success("finisched")
        st.session_state.busy = False

    elif clicked and st.session_state.last_csv_path is None:
        st.text('Please before upload the csv')




