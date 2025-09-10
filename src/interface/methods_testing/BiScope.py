# from utils
import time
import warnings
import pandas as pd
import torch
from tqdm import tqdm
from stqdm import stqdm
import os
import random

import random
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
import torch
import gc
import os, json, pickle
from types import SimpleNamespace
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import joblib



from interface.methods_testing.biscope_utils import *


MODEL_ZOO = {
    'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13b': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'gemma-2b': 'google/gemma-1.1-2b-it',
    'gemma-7b': 'google/gemma-1.1-7b-it', 
    'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
    'CodeLlama' :'meta-llama/CodeLlama-7b-hf'          #<------------------ ADDED BY ME
}



# CHANGED BY ME
COMPLETION_PROMPT_ONLY = "Complete the following code: "
COMPLETION_PROMPT = "Given the code explanation:\n{prompt}\n Complete the following code: "


def detect_single_sample(args, model, tokenizer, summary_model, summary_tokenizer, sample, summary_override , device='cuda'):
    """
    Process a sample by generating a summary-based prompt, tokenizing (with clipping),
    obtaining model outputs, and computing loss-based features (FCE and BCE).
    Returns a list of loss features computed over 10 segments.
    """
    # summary_override  <------ ADDED BY ME
    if summary_override is not None:
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_override)

    elif 'gpt-' in args.summary_model:
        from openai import OpenAI
        openai_key = os.environ.get('OPENAI_API_KEY')
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in environment.")
        client = OpenAI(api_key=openai_key)
        from tenacity import (
            retry,
            stop_after_attempt,
            wait_random_exponential,
        )
        @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
        def openai_backoff(client, **kwargs):
            return client.chat.completions.create(**kwargs)
        summary_input = f"generate a very short and concise summary for the following text, just the summary: {sample}"
        response = openai_backoff(client, model=args.summary_model,
                                  messages=[{"role": "user", "content": summary_input}])
        summary_text = response.choices[0].message.content.strip()
        # if '"""' in summary_text:
        #     summary_text = summary_text.split('"""')[-1]
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_text)
    elif args.summary_model in MODEL_ZOO:
        summary_input = f"Write a title for this text: {sample}\nJust output the title:"
        summary_ids = summary_tokenizer(summary_input, return_tensors='pt',
                                        max_length=args.sample_clip, truncation=True).input_ids.to(device)
        summary_ids = summary_ids[:, 1:]  # Remove start token.
        gen_ids = generate(summary_model, summary_tokenizer, summary_ids, summary_ids.shape[1], 64)
        summary_text = summary_tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split('\n')[0]
        prompt_text = COMPLETION_PROMPT.format(prompt=summary_text)
    else:
        prompt_text = COMPLETION_PROMPT_ONLY
    ##################################################################################

    # Tokenize the prompt and sample with token-level clipping.
    prompt_ids = tokenizer(prompt_text, return_tensors='pt').input_ids.to(device)
    text_ids = tokenizer(sample, return_tensors='pt', max_length=args.sample_clip, truncation=True).input_ids.to(device)
    combined_ids = torch.cat([prompt_ids, text_ids], dim=1)
    text_slice = slice(prompt_ids.shape[1], combined_ids.shape[1])
    outputs = model(input_ids=combined_ids)
    logits = outputs.logits
    targets = combined_ids[0][text_slice]
    
    # Compute loss features from FCE and BCE losses.
    fce_loss = compute_fce_loss(logits, targets, text_slice)
    bce_loss = compute_bce_loss(logits, targets, text_slice)
    features = []
    for p in range(1, 10):
        split = len(fce_loss) * p // 10
        features.extend([
            np.mean(fce_loss[split:]), np.max(fce_loss[split:]), 
            np.min(fce_loss[split:]), np.std(fce_loss[split:]),
            np.mean(bce_loss[split:]), np.max(bce_loss[split:]), 
            np.min(bce_loss[split:]), np.std(bce_loss[split:])
        ])
    return features







def data_generation(out_dir: str, dataset_path: str, clear_code : bool = True, use_prompt : bool = True, quantization : bool = True, model: str = "CodeLlama"):
    """

    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    
    os.makedirs(out_dir, exist_ok=True)




    # (2) DETECTION MODEL (logits): Load detection model (required)
    ## use only trasformer inside the paper
    if model not in MODEL_ZOO:
        raise ValueError(f"Unknown detection model key: {model}. Must be one of: {list(MODEL_ZOO)}")
    ## 

    # quantizzation support
    if quantization:
        quant = BitsAndBytesConfig(load_in_8bit=True)
        kwargs = dict(
            quantization_config=quant,
            device_map="auto" if DEVICE.startswith("cuda") else None,
            low_cpu_mem_usage=True,
        )
    else:
        kwargs = dict(
            torch_dtype=torch.float16 if DEVICE.startswith("cuda") else None,
            device_map="auto" if DEVICE.startswith("cuda") else None,
            low_cpu_mem_usage=True,   # utile anche senza quant
        )

    # MODEL
    det_m = AutoModelForCausalLM.from_pretrained(
        MODEL_ZOO[model],
        **kwargs
    ).eval()
    
    # TOKENIZER
    # padding_side = left so the model will never predict padding token
    det_tok = AutoTokenizer.from_pretrained(MODEL_ZOO[model], padding_side="left")
    # if the model don't have a default pad token e use [end-fo-sequence]
    # this not introduce problem in eval time
    det_tok.pad_token = det_tok.eos_token



    # (3) Load local data

    # (3.0) setup
    # (3.0) Obtaining len of the dataset
    df = pd.read_csv(dataset_path)
    LEN_DS = len(df)
    LEN_HUMAN = (df['label'] == 0).sum()
    LEN_LLM = (df['label'] == 1).sum()
    del df
    gc.collect()
    # (3.0) using cleaned or not
    if clear_code:
        code = "clear_code"
    else:
        code = "code"
    print(f"Using {code} column")


    # (3.1) load data in streaming
    ds = load_dataset("csv", data_files=dataset_path, split="train", streaming=True)

    # (3.1) Humans
    ds_human = ds.filter(lambda ex: ex["label"]==0)
    if next(iter(ds_human), None)(ds_human) is None:
        raise Exception("Filter by label failed")
    
    # (3.1) LLM
    ds_LLM = ds.filter(lambda ex: ex["label"]==1)
    if next(iter(ds_LLM), None)(ds_LLM) is None:
        raise Exception("Filter by label failed")




    #____________________________




    # (4) arg needed by the method (all default pharameter)
    args_like = SimpleNamespace(
        summary_model=None,
        sample_clip=2048,   
        split_ratio=0.1,
        n_segments=10,
    )
    #############

    print("STARTING Features Extract...\n\n\n")
    

    
    # (5) Extract & save features
    torch.set_grad_enabled(False)
    # (5.1) human_features
    human_feat_path = os.path.join(out_dir, "human_features.pkl")
    if os.path.isfile(human_feat_path):
        raise FileExistsError('human_features.pkl just exsist, please erase it')
    

    none_list = [None] * LEN_HUMAN
    human_features = [
        detect_single_sample(args_like, det_m, det_tok, None, None, text, summary_override=prompt, device=DEVICE)
        for text, prompt in stqdm(
            zip(ds_human[code], ds_human["prompt"] if use_prompt else none_list),  # iterable
            total=LEN_HUMAN,                # tqdm
            desc="Human code features generation"     # tqdm
        )
    ]
    with open(human_feat_path, "wb") as f:
        pickle.dump(human_features, f)




    gpt_feat_path = os.path.join(out_dir, "LLM_features.pkl")
    if os.path.isfile(gpt_feat_path):
        raise FileExistsError('human_features.pkl just exsist, please erase it')
    
    none_list = [None] * LEN_LLM
    gpt_features = [
        detect_single_sample(args_like, det_m, det_tok, None, None, text, summary_override=prompt, device=DEVICE)
        for text, prompt in stqdm( 
            zip(ds_LLM[code], ds_LLM["prompt"] if use_prompt else none_list), 
            total=LEN_LLM,
            desc="LLM code features generation", 
        )
            
    ]
    with open(gpt_feat_path, "wb") as f:
        pickle.dump(gpt_features, f)


    torch.set_grad_enabled(True)
    # DEALLOCATION
    del det_m
    del det_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return human_feat_path, gpt_feat_path







def train(model, dataset_path, clear_code: bool, use_prompt: bool, seed: int=42 ):
    '''
    '''

    # Set seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ####
    

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = "./temp/BiScope/train"
    os.makedirs(out_dir, exist_ok=True)
    
    
    # Generate features for the training dataset.
    print("Generating train features...")

    human_feat_path, gpt_feat_path = data_generation(out_dir=out_dir,
                                                     dataset_path = dataset_path, 
                                                    clear_code = clear_code,
                                                    use_prompt = use_prompt,
                                                    model = model
                                                    )

    
    # Load train features.
    with open(human_feat_path, 'rb') as f:
        train_human = np.array(pickle.load(f))
    with open(gpt_feat_path, 'rb') as f:
        train_gpt = np.array(pickle.load(f))
    

    
    # CLASSIFICTION TRAINING
    print("TRAINING...")
    train_feats = np.concatenate([train_human, train_gpt], axis=0)
    train_labels = np.concatenate([np.zeros(len(train_human)), np.ones(len(train_gpt))], axis=0)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed)
    clf.fit(train_feats, train_labels)



    # SAVE MODEL
    metadata = {
        "data_generation": data_generation,
        "clear_code" : clear_code,
        "use_prompt" : use_prompt,
        "feature_schema": "10x(FCE+BCE)*4stats",  # ?
    }

    bundle = {"clf": clf, "metadata": metadata}

    model_path = os.path.join(out_dir, f"biscope_rf.joblib")
    joblib.dump(bundle, model_path)
    print("Saved:", model_path)


    return model_path








def test(model, dataset_path, clear_code: bool, use_prompt: bool, seed: int=42 ):
    """

    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set seeds.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    ####
    
    
    out_dir = "./temp/BiScope/test"
    os.makedirs(out_dir, exist_ok=True)
    
    
    # Generate features for the training dataset.
    print("Generating train features...")

    torch.set_grad_enabled(False)
    human_feat_path, gpt_feat_path = data_generation(out_dir=out_dir, 
                                                     dataset_path=dataset_path,
                                                    clear_code = clear_code,
                                                    use_prompt = use_prompt,
                                                    model = model
                                                    )
    torch.set_grad_enabled(True)



    # Carica feature test
    with open(human_feat_path, 'rb') as f:
        test_human = np.array(pickle.load(f))
    with open(gpt_feat_path, 'rb') as f:
        test_gpt = np.array(pickle.load(f))


    # load model
    bundle = joblib.load("./temp/BiScope/train/biscope_rf.joblib")
    clf : RandomForestClassifier = bundle["clf"]
    print(bundle["metadata"])

    # X_test e y_test
    X_test = np.concatenate([test_human, test_gpt], axis=0)
    y_real = np.concatenate([np.zeros(len(test_human)), np.ones(len(test_gpt))], axis=0)
    print(f"len human:{len(test_human)} len gpt:{len(test_gpt)}")

    # Prediction
    preds = clf.predict(X_test)
    print(len(preds))

    return preds, y_real