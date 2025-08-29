# Is This You, LLM? How to Recognize AI-written Multilingual Programs with Reproducible Code Stylometry

This is the replication package for the 
*Is This You, LLM? How to Recognize AI-written Multilingual Programs with Reproducible Code Stylometry* paper.
The repository contains two main experiments from the paper:
 - The data construction process
 - The authorship attribution models training


The package also contains the multilingual model checkpoint, the dataset *H_AIRosettaMP.csv*, and a mirror of the Rosetta code dataset, which is used for the dataset construction process, and the codeT5plus checkpoint to permit full reproducibility over time. We also included the baseline experiments from Table 4, the intra-language test for Fig. 7, and the Hypothesis tests in Table 3.
A *CLI tool* is also present, and a brief utilization guide is displayed at the end of this document.


## Requirements
The following software dependencies are required as prerequisites:

* [Python 3.9.2](https://www.python.org/)

We highly recommend isolating the dependencies by creating a Python virtual environment as follows:

```
python3 -m venv ./venv
```

And run the env with this command:

```
source ./venv/bin/activate
```

If you are using conda you can also run:

```
conda create -n "myenv" python=3.9.2 ipython
```
And activate the enviroment with:

```
conda activate myenv
```


Then, you should install additional dependencies using pip as follows:

```
pip3 install -r requirements.txt
```
These requirements are valid for the following experiments detailed in this readme file:
- *Dataset contruction*
- *Training process*
- *Baseline experiments* but only related to our models, the Li and Oedinger experiments needs the Weka software (Li) and for Oedinger there is a separated requirements file (details below).
- *Intra language test*
- *CLI tool*
The other experiments have other requirements detailed in their section, using separate environments is highly recommended.

### Dataset contruction 
The dataset construction experiment will produce a CSV file containing the data for the multi-provenance experiments. The data will be labeled by language and provenance in the *set* column and by the author (whether Human or AI) in the *target* column. The snippet of codes will be represented in the *code* column. Three meta-data columns are also present with respectively the task description (*task_description* column), task url of provenance from rosetta code (*task_url* column), task name (*task_name* column) and the destination language (*language_name*).


The process can be thus reproduced by running on the *dataset_construction* folder:

```
python3 data_cons.py --cache_dir *path_to_cache*
```
Where the *cache_dir* will handle all the huggingface data.
The resulting dataset will be in the same folder where the job execution began.

### Training process
In order to replicate the training process, first you have to download the complete dataset *H_AIRosettaMP.csv* from the Zenodo repository.


For the multi provenance models experiments, encompassing the monolingual models and the multilingual model, you should run this code in the *model_training* folder:

```
python3 trainer_multi_prov.py --cache_dir *path_to_cache* --path_to_dataset *path_to_dataset*
```

Respectively, the --cache_dir argument will handle all the huggingface data, and the --path_to_dataset argument should point to the *H_AIRosettaMP.csv* dataset file.
All the models are evaluated at runtime, and the results are displayed on the command line and in a dedicated logs and checkpoint folder.

### Baseline experiments

First, you must proceed with the previous point (training process) to replicate the zero-shot results in the baseline comparison experiments.
Here, we test our best models (Kotlin provenance, and in the case of Kotlin, Go as the second best provenance language) to out-distribution languages.
With this test, you have a full overview of the models in a zero-shot setup. The in-distribution results (same language and provenance of the training experiment) for this experiment should not be taken into account, as the training set is included in the dataset.
We use this data to compare monolingual models to the baselines even on languages they are not trained for.
You should run this code in the *baselines* folder:

```
python3 cross_tester.py --cache_dir *path_to_cache* --path_to_dataset *path_to_dataset* --path_checkpoint *path_to_checpoint_folder*
```
Respectively, the --cache_dir argument will handle all the huggingface data, and the --path_to_dataset argument should point to the *H_AIRosettaMP.csv* dataset file. The --path_checkpoint should in this case point to the checkpoint folder containing the models that you want to test. In the paper the results follow the baselines with Java, C++ and Python languages with Kotlin provenance.

The baseline work retrained with our dataset comes respectively from:
-  Ke Li, Sheng Hong, Cai Fu, Yunhe Zhang, and Ming Liu. Discriminating human-authored from chatgpt-generated code via discernable feature analysis.In 34th IEEE International Symposium on Software Reliability
Engineering, ISSRE 2023 - Workshops, Florence, Italy, October 9-12,
2023, pages 120–127. IEEE, 2023 [https://github.com/LiKe-rm/Human-and-ChatGPT-Code-Dataset]

A mirror of our dataset adapted with the features extracted for this methodology can be found in the baselines/Li folder, with both Java and C++ datasets. To replicate the experiment you should at first download the Weka software [https://ml.cms.waikato.ac.nz/weka], then you can open the datasets file *weka_set_C++Starcoder2_undersampled.arff* or *weka_set_javaStarcoder2_undersampled.arff* and use cross validadion=10, Author name as the classification column and J48 or Random forest as classification algorithms

- Marc Oedingen, Raphael C. Engelhardt, Robin Denz, Maximilian Hammer, and Wolfgang Konen. Chatgpt code detection: Techniques for uncovering the source of code. CoRR, abs/2405.15512, 2024. 
[https://github.com/MarcOedingen/ChatGPT-Code-Detection]
A mirror of the baseline methodology is also present in baselines/Oedingen. You can replicate the full experiment with our dataset by at first installing the requirements related to Oedingen's methodology in the baselines/Oedingen folder:


```
pip3 install -r requirements.txt
```

Then you can run the experiment in the you should run this code in the *baselines/Oedingen* folder:
```
python3 ChatGPT-Code-Detection/main.py --dataset Unformatted --embedding TFIDF --algorithm XGB --seed 32
```

### Intra language test
You should first complete the Training process to replicate the intra language results and test how our best-performing models (Kotlin provenance) perform over the same 'dst' language but with different 'src' (provenances).
You can run the experiment running the following command in the *intra_lan_experiment* folder:
```
python3 intraLan_provenance.py --cache_dir *path_to_cache* --path_to_dataset *path_to_dataset* --path_checkpoint *path_to_checpoint_folder*
```
Respectively, the --cache_dir argument will handle all the huggingface data, and the --path_to_dataset argument should point to the *H_AIRosettaMP.csv* dataset file. The --path_checkpoint should in this case point to the checkpoint folder containing the models that you want to test.

### Hypothesis tests
Several hypothesis tests are conducted in the paper, a notebook *hypotesis.ipynb* is also available to reproduce the results and it's present in the folder *hypothesis_tests*. Requirements are few requirements so we decided to write down anothe requirements file, also present in the *hypothesis_tests* folder. To run the requirements you should run this command in the *hypothesis_tests* folder:

```
pip3 install -r requirements.txt
```

### CLI tool

In order to utilize the model within the command line interface, you should at first download the model's checkpoint from the Zenodo repository and then run this code in the *CLImodel* folder:


```
python3 modelRunner.py --cache_dir *path_to_cache*  --path_checkpoint *path_to_model_checkpoint*
```

Respectively, the --cache_dir argument will handle all the huggingface data. The --path_checkpoint should point to the model checkpoint.