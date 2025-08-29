from datasets import load_dataset, concatenate_datasets, Dataset
from matplotlib import pyplot as plt
import numpy as np
from tests.Dataset.Code_preprocessing.code_cleaner import comment_remover
from tests.Dataset.Code_preprocessing.balance_dataset import balanced_sample_multi_cols
import pandas as pd

import altair as alt



class CodeMirage():

    def __init__(self, len_dataset = 1000, seed = 30):
        dataset : Dataset  = load_dataset("HanxiGuo/CodeMirage")
        dataset = concatenate_datasets([dataset["train"], dataset["test"]])

        dataset = dataset.shuffle(seed=seed)

        dataset = dataset.rename_column(original_column_name= "source", 
                                        new_column_name="LLM") # old name -> new name
        
        dataset = dataset.filter(lambda example: example['variant'] != 'Paraphrased')

        dataset = dataset.remove_columns('variant')


        not_human = dataset.filter(lambda x: x["LLM"] != "Human" )
        not_human = balanced_sample_multi_cols(not_human, 
                                               cols=("language","LLM"), 
                                               desired_n=len_dataset//2, 
                                               seed=seed)
        

        human = dataset.filter(lambda x: x["LLM"] == "Human" )
        human = balanced_sample_multi_cols(human, 
                                           cols=("language","LLM"), 
                                           desired_n=len_dataset//2, 
                                           seed=seed)
        


        dataset =concatenate_datasets([human, not_human])
        
        all_codes = [comment_remover(x, language) for x, language in zip(dataset["code"], dataset["language"])]
        dataset = dataset.add_column("cleared_code", all_codes)

        dataset = dataset.filter(lambda x: x["code"] is not None and len(x["code"]) >= 10)

        self.DF : pd.DataFrame = dataset.to_pandas()
        self.DF ['index'] = self.DF.index
        self.DF = self.DF.reindex(columns=['LLM', 'language', 'code', 'cleared_code', 'index'])



    def __len__(self):
        return len(self.Dataset)
    
    def head(self):
        return self.DF.head()
    
    def LLM_count(self):
        count = self.DF['LLM'].value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = ['LLM', 'count']
        chart = alt.Chart(df, title='different code souce', height=400).mark_bar(color = 'FireBrick').encode(
            x=alt.X('LLM:N', sort='-y', title='LLM'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['LLM', 'count']
        )
        return chart
    
    def language_count(self):
        name = 'language'
        count = self.DF[name].value_counts()
        df = pd.DataFrame({"count": count})
        df = df.reset_index()
        df.columns = [name, 'count']
        chart = alt.Chart(df, title='language', height=400).mark_bar(color = 'Maroon').encode(
            x=alt.X('language:N', sort='-y', title='language'),
            y=alt.Y('count:Q', title='Count'),
            tooltip=['language', 'count']
        )
        return chart
    
    def code(self):
        lenx = self.DF['code'].str.len()
        df = pd.DataFrame({"len": lenx})

        chart = alt.Chart(df, title='code length', height=400).mark_bar(color='Tomato').encode(
            x=alt.X("len:Q",
                    bin=alt.Bin(step=20),          # o maxbins=30
                    axis=alt.Axis(title="Lunghezza", tickCount=10, labelAngle=0)),
            y=alt.Y("count()", title="Frequenza")
        )
        
        return chart
    


    def cleared_code(self):
        lenx = self.DF['cleared_code'].str.len()
        df = pd.DataFrame({"len": lenx})
        chart = alt.Chart(df, title='clean code length', height=400).mark_bar(color = 'Crimson').encode(
            x=alt.X("len:Q",
                    bin=alt.Bin(step=20),          # o maxbins=30
                    axis=alt.Axis(title="Lunghezza", tickCount=10, labelAngle=0)),
            y=alt.Y("count()", title="Frequenza")
        )
        return chart
        #st.pyplot(fig)


    def convert_df_to_csv(self):
        return self.DF.to_csv(index=False).encode('utf-8')
    


