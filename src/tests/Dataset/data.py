from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Dataset
from matplotlib import pyplot as plt
import numpy as np
from tests.Dataset.Code_preprocessing.code_cleaner import comment_remover, newline_remover, import_remover
from tests.Dataset.Code_preprocessing.balance_dataset import balanced_sample_multi_cols
import pandas as pd

import altair as alt

from sklearn.model_selection import train_test_split





class dataset():
    def __init__(self, seed=30):
        self.seed = seed
        self.DF :pd.DataFrame = None

    def __len__(self):
        return len(self.DF)
    
    def head(self):
        return self.DF.head()
    
    def LLM_count(self, color = 'FireBrick'):
        count = self.DF['LLM'].value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = ['LLM', 'count']
        chart = alt.Chart(df, title='different code souce', height=400).mark_bar(color = color).encode(
            x=alt.X('LLM:N', sort='-y', title='LLM'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['LLM', 'count']
        )
        return chart
    
    def language_count(self, color = 'Maroon'):
        name = 'language'
        count = self.DF[name].value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = [name, 'count']
        chart = alt.Chart(df, title='language', height=400).mark_bar(color = color).encode(
            x=alt.X('language:N', sort='-y', title='language'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['language', 'count']
        )
        return chart
    
    def status_in_folder(self, color = 'Maroon'):
        name = 'status_in_folder'
        count = self.DF[name].value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = [name, 'count']
        chart = alt.Chart(df, title='status_in_folder', height=400).mark_bar(color = color).encode(
            x=alt.X('status_in_folder:N', sort='-y', title='status in folder'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['status_in_folder', 'count']
        )
        return chart
    


    def cleared_code(self, color= 'Crimson'):
        lenx = self.DF['cleared_code'].str.len()
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
        dataset : Dataset  = load_dataset("HanxiGuo/CodeMirage")
        #dataset = concatenate_datasets([dataset["train"], dataset["test"]])
        dataset = dataset["train"]
        dataset_test = dataset_test["test"]

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
        


        dataset =concatenate_datasets([human, not_human])
        
        all_codes = [import_remover(comment_remover(x, language), language)  for x, language in zip(dataset["code"], dataset["language"])]
        dataset = dataset.add_column("cleared_code", all_codes)

        all_codes_test = [import_remover(comment_remover(x, language), language)  for x, language in zip(dataset_test["code"], dataset_test["language"])]
        dataset_test = dataset_test.add_column("cleared_code", all_codes_test)

        dataset = dataset.filter(lambda x: x["code"] is not None and len(x["code"]) >= 10)
        dataset_test = dataset_test.filter(lambda x: x["code"] is not None and len(x["code"]) >= 10)


        dataset = dataset.add_column('status_in_folder', [None] * len(dataset))
        dataset = dataset.add_column('prompt', [None]*len(dataset))
        dataset_test = dataset_test.add_column('status_in_folder', [None] * len(dataset))
        dataset_test = dataset_test.add_column('prompt', [None]*len(dataset))



        def add_label(x):
            if x['LLM'] == 'Human':
                y = 0
            else :
                y = 1

            return {'label': y}
        
        dataset = dataset.map(add_label)
        dataset_test = dataset_test.map(add_label)


        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language', 'status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        
        self.DF_test : pd.DataFrame = dataset_test.to_pandas()
        self.DF_test ['index'] = self.DF_test.index
        self.DF_test = self.DF_test.reindex(columns=['label', 'LLM', 'language', 'status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        return
    
    @override
    def convert_df_to_csv_split(self):
        train, val  = train_test_split(self.DF, test_size=0.1, random_state=42, stratify=train["label"])
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
        HERE = Path(__file__).resolve().parent
        ROOT = HERE.parents[1]
        CSV  = ROOT / "tests" / "Dataset" / "Pan" / "pan.csv"
        CSV_final  = ROOT / "tests" / "Dataset" / "Pan" / "pan_final.csv"


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
        ds_a = ds_a.map(lambda batch: {'test_result.passed': [-1] * len(batch['test_result.passed'])}, batched=True)
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
        
        dataset = dataset.map(status)


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
        
        dataset = dataset.map(add_label)


        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        #print(self.DF.head())
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language','status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        
        return




    
class CoDETM4(dataset):

    def __init__(self, len_dataset = 10000, seed = 30):
        super().__init__(seed)
        dataset : Dataset  = load_dataset("DaniilOr/CoDET-M4", split=f"train")

        #dataset = dataset["train"]

        dataset = dataset.shuffle(seed=self.seed)

        dataset = dataset.rename_column(original_column_name= "model", 
                                        new_column_name="LLM") # old name -> new name
        

        dataset = dataset.remove_columns('features')
        dataset = dataset.remove_columns('source')




        not_human = dataset.filter(lambda x: x["LLM"] != "human" )

        not_human = balanced_sample_multi_cols(not_human, 
                                               cols=("LLM", "language"), 
                                               desired_n=len_dataset//2, 
                                               seed=self.seed)
        

        human = dataset.filter(lambda x: x["LLM"] == "human" )

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
        
        dataset = dataset.map(add_label)


        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        self.DF = self.DF.reindex(columns=['label', 'LLM', 'language', 'status_in_folder', 'prompt', 'code', 'cleared_code', 'index'])
        return

