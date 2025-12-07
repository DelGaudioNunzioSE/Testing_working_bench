from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from matplotlib import pyplot as plt
import numpy as np
#from tests.Dataset.Code_preprocessing.code_cleaner import comment_remover, newline_remover, import_remover
#from tests.Dataset.Code_preprocessing.balance_dataset import balanced_sample_multi_cols
import pandas as pd

import altair as alt

from sklearn.model_selection import train_test_split

from datasets import Dataset
from collections import defaultdict
import random

import re

def balanced_sample_multi_cols(
    ds: Dataset,
    cols=("language", "source", "variant"),
    desired_n: int = 1000,
    seed: int = 30,
):
    """
    Returns a balanced sub-dataset across ALL combinations of `cols`,
    with the same number of examples for each combination.  
    If 1000 cannot be reached while keeping perfect balance, it returns 
    the maximum balanced size <= desired_n.

    Note: only the combinations that ACTUALLY exist in the dataset are considered.
    """
    rng = random.Random(seed)

    # 1) index dataset examples by group (tuple of column values)
    idx_by_group = defaultdict(list)
    for i, ex in enumerate(ds):
        key = tuple(ex[c] for c in cols)
        idx_by_group[key].append(i)

    groups = list(idx_by_group.keys())
    G = len(groups)
    if G == 0:
        raise ValueError("Nessun gruppo trovato: controlla i nomi delle colonne.")

    # 2) how many can we take per group while keeping equality?
    #    (a) theoretical target is floor(desired_n / G)
    #    (b) cannot exceed the smallest group size
    target_per_group = desired_n // G
    cap_min = min(len(idxs) for idxs in idx_by_group.values())
    take = min(target_per_group, cap_min)

    if take == 0:
        raise ValueError(
            f"Cannot create a balanced sample: desired_n={desired_n} < number of groups={G} "
            f"or some groups have 0 elements. Increase desired_n or reduce group dimensions."
        )

    # 3) campiona esattamente 'take' per ogni gruppo
    chosen = []
    for g in groups:
        idxs = idx_by_group[g]
        chosen.extend(rng.sample(idxs, k=take))

    chosen.sort()
    sub = ds.select(chosen)

    return sub




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

    elif language in ("c", "cpp", "c++", "go", 'csharp'):  #cpp is an other way to say c++
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




class dataset():
    def __init__(self, seed=30):
        self.seed = seed
        self.DF :pd.DataFrame = None

    def __len__(self):
        return len(self.DF)
    
    def head(self):
        return self.DF.head()
    
    def LLM_count(self, color = '#EBC94C'):
        count = self.DF['LLM'].dropna().value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = ['LLM', 'count']
        chart = alt.Chart(df, title='different code souce', height=400).mark_bar(color = color).encode(
            x=alt.X('LLM:N', sort='-y', title='LLM'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['LLM', 'count']
        )
        return chart
    
    def language_count(self, color = '#E9C235'):
        name = 'language'
        count = self.DF[name].dropna().value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = [name, 'count']
        chart = alt.Chart(df, title='language', height=400).mark_bar(color = color).encode(
            x=alt.X('language:N', sort='-y', title='language'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['language', 'count']
        )
        return chart
    
    def status_in_folder(self, color = '#D3AA17'):
        name = 'status_in_folder'
        count = self.DF[name].dropna().value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = [name, 'count']
        chart = alt.Chart(df, title='status_in_folder', height=400).mark_bar(color = color).encode(
            x=alt.X('status_in_folder:N', sort='-y', title='status in folder'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['status_in_folder', 'count']
        )
        return chart
    


    def cleared_code(self, color= '#BC9815'):
        lenx = self.DF['cleared_code'].dropna().str.len()
        df = pd.DataFrame({"len": lenx})
        chart = alt.Chart(df, title='clean code length', height=400).mark_bar(color = color).encode(
            x=alt.X("len:Q",
                    bin=alt.Bin(step=20),          # o maxbins=30
                    axis=alt.Axis(title="len", tickCount=10, labelAngle=0)),
            y=alt.Y("count()", title="count")
        )
        return chart
        #st.pyplot(fig)
    



    def convert_df_to_csv(self):
        return self.DF.to_csv(index=False).encode('utf-8')
    
    def convert_df_to_csv_split(self):
        train, test = train_test_split(self.DF, test_size=0.10, random_state=42, stratify=self.DF["label"])
        train, val  = train_test_split(train, test_size=0.1111, random_state=42, stratify=train["label"])
        return train.to_csv(index=False).encode('utf-8'), val.to_csv(index=False).encode('utf-8'), test.to_csv(index=False).encode('utf-8')









class CodeMirage(dataset):

    def __init__(self, len_dataset = 1000, seed = 30):
        super().__init__(seed)
        dataset_original : Dataset  = load_dataset("HanxiGuo/CodeMirage")
        #dataset = concatenate_datasets([dataset["train"], dataset["test"]])
        dataset = dataset_original["train"]
        dataset_test = dataset_original["test"]

        dataset = dataset.shuffle(seed=self.seed)

        dataset = dataset.rename_column(original_column_name= "source", 
                                        new_column_name="LLM") # old name -> new name
        dataset_test = dataset_test.rename_column(original_column_name= "source", 
                                        new_column_name="LLM")
        
        dataset = dataset.filter(lambda example: example['variant'] != 'Paraphrased')
        dataset_test = dataset.filter(lambda example: example['variant'] != 'Paraphrased')

        dataset = dataset.remove_columns('variant')
        dataset_test = dataset_test.remove_columns('variant')


        not_human = dataset.filter(lambda x: x["LLM"] != "Human" )
        not_human = balanced_sample_multi_cols(not_human, 
                                               cols=("language","LLM"), 
                                               desired_n=len_dataset//2, 
                                               seed=self.seed)
        

        human = dataset.filter(lambda x: x["LLM"] == "Human" )
        human = balanced_sample_multi_cols(human, 
                                           cols=("language","LLM"), 
                                           desired_n=len_dataset//2, 
                                           seed=self.seed)
        

        if int(len_dataset*0.1) < 200:
            desired_n = 200
        elif int(len_dataset) > 60000:
            desired_n = 60000
        else:
            desired_n = len_dataset

        dataset_test_not_human = dataset_test.filter(lambda x: x["LLM"] != "Human" )
        dataset_test_not_human = balanced_sample_multi_cols(dataset_test_not_human, 
                                               cols=("language","LLM"), 
                                               desired_n=desired_n,
                                               seed=self.seed)
        

        dataset_test_human = dataset_test.filter(lambda x: x["LLM"] == "Human" )
        dataset_test_human = balanced_sample_multi_cols(dataset_test_human, 
                                           cols=("language","LLM"), 
                                           desired_n=desired_n,
                                           seed=self.seed)
        


        dataset =concatenate_datasets([human, not_human])
        dataset_test =concatenate_datasets([dataset_test_human, dataset_test_not_human])
        
        all_codes = [import_remover(comment_remover(x, language), language)  for x, language in zip(dataset["code"], dataset["language"])]
        dataset = dataset.add_column("cleared_code", all_codes)

        all_codes_test = [import_remover(comment_remover(x, language), language)  for x, language in zip(dataset_test["code"], dataset_test["language"])]
        dataset_test = dataset_test.add_column("cleared_code", all_codes_test)

        dataset = dataset.filter(lambda x: x["code"] is not None and len(x["code"]) >= 10)
        dataset_test = dataset_test.filter(lambda x: x["code"] is not None and len(x["code"]) >= 10)


        dataset = dataset.add_column('status_in_folder', [None] * len(dataset))
        dataset = dataset.add_column('prompt', [None]*len(dataset))
        dataset_test = dataset_test.add_column('status_in_folder', [None] * len(dataset_test))
        dataset_test = dataset_test.add_column('prompt', [None]*len(dataset_test))



        def add_label(x):
            if x['LLM'] == 'Human':
                y = 0
            else :
                y = 1

            return {'label': y}
        
        dataset = dataset.map(add_label, num_proc=4)
        dataset_test = dataset_test.map(add_label, num_proc=4)


        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language', 'status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        
        self.DF_test : pd.DataFrame = dataset_test.to_pandas()
        self.DF_test ['index'] = self.DF_test.index
        self.DF_test = self.DF_test.reindex(columns=['label', 'LLM', 'language', 'status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        return
    
    #@override
    def convert_df_to_csv_split(self):
        train, val  = train_test_split(self.DF, test_size=0.1, random_state=42, stratify=self.DF["label"])
        return train.to_csv(index=False).encode('utf-8'), val.to_csv(index=False).encode('utf-8'), self.DF_test.to_csv(index=False).encode('utf-8')






class AIG(dataset):

    def __init__(self, len_dataset = 1000, seed = 30):
        super().__init__(seed)
        dataset : Dataset  = load_dataset("basakdemirok/AIGCodeSet")
        dataset = concatenate_datasets([dataset["train"], dataset["test"]])

        dataset = dataset.shuffle(seed=self.seed)
        

        dataset = dataset.remove_columns(['problem_id', 'submission_id', 'ada_embedding', 'label', 'lines', 'code_lines', 'comments', 'functions', 'blank_lines'])


        not_human = dataset.filter(lambda x: x["LLM"] != "Human" )
        not_human = balanced_sample_multi_cols(not_human, 
                                               cols=("status_in_folder","LLM"), 
                                               desired_n=len_dataset//2, 
                                               seed=self.seed)
        

        human = dataset.filter(lambda x: x["LLM"] == "Human" )
        human = balanced_sample_multi_cols(human, 
                                           cols=("status_in_folder","LLM"), 
                                           desired_n=len_dataset//2, 
                                           seed=self.seed)
        


        dataset =concatenate_datasets([human, not_human])
        
        all_codes = [import_remover(comment_remover(x, 'python'),'python') for x in dataset["code"]]
        dataset = dataset.add_column("cleared_code", all_codes)

        dataset = dataset.filter(lambda x: x["code"] is not None and len(x["code"]) >= 10)


        dataset = dataset.add_column('language', ['python']*len(dataset))



        def add_label(x):
            if x['LLM'] == 'Human':
                y = 0
            else :
                y = 1

            return {'label': y}
        
        dataset = dataset.map(add_label)



        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language','status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        return












class Pan(dataset):

    def __init__(self, len_dataset = 1000, seed = 30):
        super().__init__(seed)
        ROOT = Path(__file__).resolve().parent
        CSV  = ROOT  / "local_dataset" / "Pan" / "pan.csv"
        CSV_final  = ROOT / "local_dataset" / "Pan" / "pan_final.csv"


        original = pd.read_csv(CSV, index_col="local index")
        processed   = pd.read_csv(CSV_final,  index_col="metadata.local index")
        
        if len(original) == 0 or len(processed) == 0:
            raise FileNotFoundError()
        human_conl = "Python Code"
        out = processed.join(original[[human_conl]], how="inner")  # aggiunge la colonna a 'short' per indice
        print(len(out))

        dataset : Dataset  = Dataset.from_pandas(out)


        dataset = dataset.shuffle(seed=self.seed)
        

        dataset = dataset.remove_columns(['file_source', 'metadata.index', 'metadata.Source Name', 'metadata.GPT Answer', 'test_result.errors' ,'test_result.failed', 'test_result.test_reliability',\
                                          'metadata.test_folder_name', 'metadata.variant', "test_code"])


        dataset = dataset.filter(lambda x: x["solution_code"] is not None and len(x["solution_code"]) >= 10)


        dataset = dataset.add_column('language', ['python']*len(dataset))


        ds_a = dataset.remove_columns(["solution_code"]).rename_column("Python Code",'code')
        print(ds_a)
        ds_a = ds_a.map(lambda batch: {'test_result.passed': [-1] * len(batch['test_result.passed'])}, batched=True, num_proc=4)
        ds_b = dataset.remove_columns(["Python Code"]).rename_column("solution_code",'code')
        ds_a = ds_a.add_column('LLM', ['Human']*len(ds_a))
        ds_b = ds_b.add_column('LLM', ['GPT']*len(ds_a))


        dataset = concatenate_datasets([ds_a, ds_b])


        all_codes = [comment_remover(x, 'python') for x in dataset["code"]]
        all_codes = [import_remover(x, 'python') for x in dataset["code"]]
        dataset = dataset.add_column("cleared_code", all_codes)



        dataset = dataset.rename_column('metadata.local index', 'index' )
        dataset = dataset.rename_column('instruction' ,'prompt')
        dataset = dataset.rename_column('test_result.passed' ,'status_in_folder')

        def status(x):
            if x['status_in_folder'] == -1:
                y = 'Not tested'
            elif x['status_in_folder'] == 3:
                y = 'Accepted'
            else :
                y = 'wrong'

            return {'status_in_folder': y}
        
        dataset = dataset.map(status, num_proc=4)


        dataset = balanced_sample_multi_cols(dataset, 
                                    cols=("LLM", "status_in_folder"), 
                                    desired_n=len_dataset, 
                                    seed=self.seed)
        


        def add_label(x):
            if x['LLM'] == 'Human':
                y = 0
            else :
                y = 1

            return {'label': y}
        
        dataset = dataset.map(add_label, num_proc=4)


        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        #print(self.DF.head())
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language','status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        
        return
    
    def return_statu_in_folder(self):
        '''return accepted and wrong'''
        return self.DF[self.DF["status_in_folder"] != "wrong"].to_csv(index=False).encode('utf-8') , self.DF[self.DF["status_in_folder"] != "Accepted"].to_csv(index=False).encode('utf-8')




    
class CoDETM4(dataset):

    def __init__(self, len_dataset = 10000, seed = 30):
        super().__init__(seed)
        dataset : Dataset  = load_dataset("DaniilOr/CoDET-M4", split=f"train")

        dataset = dataset.rename_column(original_column_name= "model", 
                                        new_column_name="LLM") # old name -> new name
        

        dataset = dataset.remove_columns('features')
        dataset = dataset.remove_columns('source')




        not_human = dataset.filter(lambda x: x["LLM"] != "human" , num_proc = 4)

        not_human = balanced_sample_multi_cols(not_human, 
                                               cols=("LLM", "language"), 
                                               desired_n=len_dataset//2, 
                                               seed=self.seed)
        

        human = dataset.filter(lambda x: x["LLM"] == "human" , num_proc = 4)

        human = balanced_sample_multi_cols(human, 
                                           cols=("LLM", "language"), 
                                           desired_n=len_dataset//2, 
                                           seed=self.seed)
        


        dataset =concatenate_datasets([human, not_human])
        
        all_codes = [import_remover(comment_remover(x, language), language)  for x, language in zip(dataset["code"], dataset["language"])]
        dataset = dataset.add_column("cleared_code", all_codes)



        dataset = dataset.add_column('status_in_folder', [None] * len(dataset))
        dataset = dataset.add_column('prompt', [None]*len(dataset))



        def add_label(x):
            if x['LLM'] == 'human':
                y = 0
            else :
                y = 1

            return {'label': y}
        
        dataset = dataset.map(add_label, num_proc=4)

        dataset = dataset.filter(lambda ex: (ex.get("cleared_code") is not None) and isinstance(ex["cleared_code"], str) and (len(ex["cleared_code"]) >= 10), num_proc=4)
        


        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index

        

        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language', 'status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        return














class Uncovering(dataset):

    def __init__(self, len_dataset = 1000, seed = 30):
        super().__init__(seed)
        ROOT = Path(__file__).resolve().parent
        JSONL = ROOT / "local_dataset" / "Uncovering" / "train_in_domain.jsonl"

        self.DF : pd.DataFrame = pd.read_json(JSONL, lines=True)
        self.DF = self.DF.rename(columns={"func_str": "code"})

        df = balanced_sample_multi_cols(Dataset.from_pandas(self.DF), 
                                    cols=("label",), 
                                    desired_n=len_dataset, 
                                    seed=self.seed)
        

        all_codes = [import_remover(comment_remover(x, "python"), "python")  for x in df["code"]]
        df = df.add_column("cleared_code", all_codes)
        
        self.DF = df.to_pandas()
        self.DF["LLM"] = self.DF["label"].map({0: "human", 1: "LLM"})
        self.DF["language"] = "Python"
        self.DF ['index'] = self.DF.index
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language','status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        return
