import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.model_selection import StratifiedKFold, KFold
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import Dataset
from itertools import chain
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score
from collections import defaultdict

def compute_metrics(y_true, y_score, base_thr=0.5, opposite=False):
    if y_true is None or y_score is None or len(y_true) != len(y_score) or len(y_true)==0:
        return None, None, None, None, None, None
    
    if opposite:
        y_score = 1 - y_score
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    
    if np.isnan(y_score).any():
        y_score = np.nan_to_num(y_score, nan=0.0)
    
    classes = np.unique(y_true)
    if classes.size != 2 or not np.all(np.isin(classes, [0,1])):
        return None, None, None, None, None, None
    
    # F1 
    y_pred = (y_score >= base_thr).astype(int)
    f1 = float(f1_score(y_true, y_pred, pos_label=1))
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    

    unique_fpr = []
    unique_tpr = []
    unique_thr = []
    
    i = 0
    while i < len(fpr):
        current_fpr = fpr[i]
        max_tpr = tpr[i]
        best_thr = thresholds[i]
        
        j = i + 1
        while j < len(fpr) and np.isclose(fpr[j], current_fpr, rtol=1e-15):
            if tpr[j] > max_tpr:
                max_tpr = tpr[j]
                best_thr = thresholds[j]
            j += 1
        
        unique_fpr.append(current_fpr)
        unique_tpr.append(max_tpr)
        unique_thr.append(best_thr)
        i = j
    
    fpr_clean = np.array(unique_fpr)
    tpr_clean = np.array(unique_tpr)
    thr_clean = np.array(unique_thr)
    
    def tpr_at(alpha):
        """TPR interpolato al valore FPR = alpha"""
        alpha = float(alpha)
        if alpha <= fpr_clean[0]:
            return float(tpr_clean[0])
        if alpha >= fpr_clean[-1]:
            return float(tpr_clean[-1])
        return float(np.interp(alpha, fpr_clean, tpr_clean))
    
    def thr_at(alpha):
        """Soglia e TPR effettivi (step function) al valore FPR = alpha"""
        alpha = float(alpha)
        
        # 
        finite_mask = np.isfinite(thr_clean)
        if not np.any(finite_mask):
            return np.nan, 0.0
        
        fpr_finite = fpr_clean[finite_mask]
        tpr_finite = tpr_clean[finite_mask]
        thr_finite = thr_clean[finite_mask]
        
        if alpha < fpr_finite[0]:
            # 
            return np.nan, float(tpr_at(alpha))  
        
        #  FPR <= alpha
        idx = np.searchsorted(fpr_finite, alpha, side="right") - 1
        return float(thr_finite[idx]), float(tpr_finite[idx])
    

    tpr_at_10 = tpr_at(0.10)
    tpr_at_01 = tpr_at(0.01)
    thr_10, tpr_step_10 = thr_at(0.10)
    thr_01, tpr_step_01 = thr_at(0.01)
    
    return f1, tpr_at_10, tpr_at_01, float(base_thr), float(thr_10), float(thr_01)




def thr_for_fpr(y_true, ppl, target_fpr=0.10, positive_is_higher=True):
    y_true = np.asarray(y_true).astype(int)
    ppl = np.asarray(ppl, dtype=float)
    score = ppl if positive_is_higher else -ppl
    fpr, tpr, thr = roc_curve(y_true, score, pos_label=1)
    m = np.isfinite(thr)
    fpr, tpr, thr = fpr[m], tpr[m], thr[m]

    if len(thr) == 0:
        return None, 0.0, 0.0

    m2 = fpr <= target_fpr

    if not np.any(m2):
        return None, float(np.interp(target_fpr, fpr, tpr)), float(target_fpr)

    idxs = np.where(m2)[0]

    best = idxs[np.argmax(tpr[m2])]


    thr_ppl = thr[best] if positive_is_higher else -thr[best]

    return float(thr_ppl)


def graph(y=[0.88, 0.82, 0.55], err=[0.03, 0.04, 0.06], method='we' ):
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    # graph
    y   = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    method = method

    x = np.arange(len(y))                  # [0,1,2]
    colors  = ["#bf8488", "#a3b1cd", "#4f709f"]
    hatches = ["", "/", "\\"]
    metric_labels  = ["F1 Score", "TPR@FPR=10%", "TPR@FPR=1%"]


    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(4, 3), facecolor="white")
    ax.set_facecolor("white")

    bars = ax.bar(
        x, y,
        color=colors, edgecolor="white",
        yerr=err, ecolor="red", capsize=6,
        error_kw={"elinewidth": 2},
    )
    for b, h in zip(bars, hatches):
        b.set_hatch(h)

    ax.set_xlim(-0.5, len(y)-0.5)
    ax.set_ylim(0, 1.0)

    center = float(x.mean())               # centro del gruppo (1.0)
    ax.set_xticks([center])
    ax.set_xticklabels([method], rotation=30, ha="right")

    ax.axhline(0.5, linestyle=":", color="k", alpha=0.6, linewidth=1)
    for s in ("top","right"):
        ax.spines[s].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, edgecolor="black", hatch=h, label=lab)
            for c, h, lab in zip(colors, hatches, metric_labels)]
    leg = ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=3, frameon=True,
        handlelength=1.2, handleheight=1.2   # quadrati
    )
    leg.set_in_layout(False)

    plt.tight_layout(rect=(0, 0, 1, 0.92))
    st.pyplot(fig)
    plt.close(fig)
    return 

    









# SPLIT-BY-CONLUM
def split_by(ds:Dataset, split_col:str="language") -> tuple[dict,dict]:
    split_col_val = ds.unique(split_col)
    labels_by_llm, scores_by_llm = {}, {}
    for val in split_col_val:
        sub = ds.filter(lambda b, v=val : [x == v for x in b[split_col]],    batched=True )  
        labels_by_llm[val] = sub["label"]
        scores_by_llm[val] = sub["score"]
    return labels_by_llm, scores_by_llm



### SPLIT K-FOLD

def split_by_kfold(ds: Dataset, k: int = 5, shuffle: bool = True, seed: int = 42) -> tuple[dict, dict]:
    labels = np.array(ds["label"]) 
    scores = np.array(ds["score"])  
    llm = np.array(ds["LLM"])      
    language = np.array(ds["language"])  

    # Verifica che le lunghezze siano compatibili
    if labels.shape[0] != scores.shape[0] or labels.shape[0] != llm.shape[0] or labels.shape[0] != language.shape[0]:
        raise ValueError("different len between labels, scores, LLM e language.")
    
    n = len(labels)

    # 
    human_mask = (llm == "human") | (llm == "Human")
    llm_non_human = llm[~human_mask]
    language_non_human = language[~human_mask]
    labels_non_human = labels[~human_mask]
    scores_non_human = scores[~human_mask]
    
    llm_human = llm[human_mask]
    language_human = language[human_mask]
    labels_human = labels[human_mask]
    scores_human = scores[human_mask]

    # 1. 
    human_splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    human_labels_by = defaultdict(list)
    human_scores_by = defaultdict(list)

    if len(language_human) == 0:
        raise ValueError("language_human empty")
    
    # 
    for i, (train_idx, test_idx) in enumerate(human_splitter.split(np.zeros(len(language_human)), language_human), start=1):
        human_labels_by[f"fold{i}"].extend(labels_human[test_idx].tolist())
        human_scores_by[f"fold{i}"].extend(scores_human[test_idx].tolist())

    # 2. Stratificazione
    combined = np.array([f"{l}-{lang}" for l, lang in zip(llm_non_human, language_non_human)]) # combino linguaggio e llm come fossero un solo gruppo
    splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    
    unique_classes, counts = np.unique(combined, return_counts=True)
    print(dict(zip(unique_classes, counts)))

    labels_by = defaultdict(list)
    scores_by = defaultdict(list)
    
    # 
    for i, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(combined)), combined), start=1):
        labels_by[f"fold{i}"].extend(labels_non_human[test_idx].tolist())
        scores_by[f"fold{i}"].extend(scores_non_human[test_idx].tolist())

    # 3. 
    for i in range(1, k + 1):
        labels_by[f"fold{i}"].extend(human_labels_by[f"fold{i}"])
        scores_by[f"fold{i}"].extend(human_scores_by[f"fold{i}"])

    return dict(labels_by), dict(scores_by)




def split_pair_dicts_into_kfolds(labels_by: dict, scores_by: dict, k: int = 5,
                                 stratify: bool = False, shuffle: bool = True, seed: int = 42):
    # check input
    if len(labels_by) != 1 or len(scores_by) != 1:
        raise ValueError("Entrambi i dizionari devono avere una sola chiave.")
    k_lab, y = next(iter(labels_by.items()))
    k_sco, s = next(iter(scores_by.items()))
    if k_lab != k_sco:
        raise ValueError("La chiave deve coincidere nei due dizionari.")
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.shape[0] != s.shape[0]:
        raise ValueError("Lunghezze diverse tra labels e scores.")
    n = len(y)
    if k > n:
        raise ValueError(f"k={k} > n={n}. Riduci k o aumenta i campioni.")


    if stratify:
        uniq, counts = np.unique(y, return_counts=True)
        if len(uniq) >= 2 and counts.min() >= k:
            splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
            split_iter = splitter.split(np.zeros(n), y)
        else:
            # fallback a KFold
            splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
            split_iter = splitter.split(np.arange(n))
    else:
        splitter = KFold(n_splits=k, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(np.arange(n))


    new_labels, new_scores = {}, {}
    base_key = k_lab
    for i, (_, test_idx) in enumerate(split_iter, start=1):
        new_labels[f"{base_key}_fold{i}"] = y[test_idx].tolist()
        new_scores[f"{base_key}_fold{i}"] = s[test_idx].tolist()

    return new_labels, new_scores



def compute_graph(labeles : dict, scores : dict, 
                  labels_by_llm : dict, scores_by_llm : dict,
                  labels_by_lang: dict, scores_by_lang: dict,
                  method: str = 'we', only_threshold: bool = False, opposite = False):

    def _collect_metrics(labels_by, scores_by):
        f1_list, t10_list, t01_list, thr_10_list, thr_1_list = [], [], [], [], []
        for k in labels_by:
            y_t = labels_by[k]; y_s = scores_by[k]
            f1_i, t10_i, t01_i, _ , thr_10_i, thr_1_i= compute_metrics(y_true=y_t, y_score=y_s, opposite=opposite)
            if f1_i is not None:  f1_list.append(f1_i)
            if t10_i is not None: t10_list.append(t10_i)
            if t01_i is not None: t01_list.append(t01_i)
            if thr_10_i is not None: thr_10_list.append(thr_10_i)
            if thr_1_i is not None: thr_1_list.append(thr_1_i)
        f1_arr  = np.asarray(f1_list,  dtype=float)
        t10_arr = np.asarray(t10_list, dtype=float)
        t01_arr = np.asarray(t01_list, dtype=float)
        thr_10_arr = np.asarray(thr_10_list, dtype=float)
        thr_1_arr = np.asarray(thr_1_list, dtype=float)
        # medie per split
        m_f1  = float(np.nanmean(f1_arr))  if f1_arr.size  else float("nan")
        m_t10 = float(np.nanmean(t10_arr)) if t10_arr.size else float("nan")
        m_t01 = float(np.nanmean(t01_arr)) if t01_arr.size else float("nan")
        m_thr_10 = float(np.nanmean(thr_10_arr)) if thr_10_arr.size else float("nan")
        m_thr_1 = float(np.nanmean(thr_1_arr)) if thr_1_arr.size else float("nan")

        f1_err  =  float(np.nanstd(f1_arr, ddof=1)) if np.isfinite(f1_arr).all() and f1_arr.size > 1 else float("nan")
        t10_err = float(np.nanstd(t10_arr, ddof=1)) if np.isfinite(t10_arr).all() and t10_arr.size > 1 else float("nan")
        t01_err = float(np.nanstd(t01_arr, ddof=1)) if np.isfinite(t01_arr).all() and t01_arr.size > 1 else float("nan")
        return m_f1, m_t10, m_t01, f1_err, t10_err, t01_err, m_thr_10, m_thr_1
    
    def _acc_by(labels_by: dict, scores_by: dict, thr: float = 0.5) -> dict:
        acc = {}
        for k in labels_by:
            y_t = np.asarray(labels_by[k]).astype(int)
            y_s = np.asarray(scores_by[k], dtype=float)
            y_s = np.nan_to_num(y_s, nan=0.0)
            y_p = (y_s >= thr).astype(int)
            if y_t.size:
                acc[k] = float(accuracy_score(y_t, y_p))
        return acc



    # mean over langauge
    labels_by_llm1 = labels_by_llm
    scores_by_llm1 = scores_by_llm
    labels_by_lang1 = labels_by_lang
    scores_by_lang1 = scores_by_lang

    #if len(labels_by_lang) == 1:
    #    labels_by_lang, scores_by_lang = split_pair_dicts_into_kfolds(labels_by_lang, scores_by_lang, k=5, stratify=True)
    m_f1, m_t10, m_t01, f1_err, t10_err, t01_err , m_thr_10, m_thr_1= _collect_metrics(labeles, scores)

    if only_threshold:
        return m_thr_10
    acc_LLM = _acc_by(labels_by_llm1, scores_by_llm1, thr = m_thr_10)
    acc_language = _acc_by(labels_by_lang1, scores_by_lang1, thr = m_thr_10)

    acc_LLM_50 = _acc_by(labels_by_llm1, scores_by_llm1, thr = 0.5)
    acc_language_50 = _acc_by(labels_by_lang1, scores_by_lang1, thr = 0.5)


    # usa il tuo grafico esistente a 3 barre
    graph(y=[m_f1, m_t10, m_t01],
          err=[f1_err, t10_err, t01_err],
          method=method)

    st.text("Accuracy over LLM and Language")
    c1, c2 = st.columns(2)
    with c1:
        if acc_LLM_50:  st.bar_chart(pd.Series(acc_LLM_50).sort_values(ascending=False))
    with c2:
        if acc_language_50: st.bar_chart(pd.Series(acc_language_50).sort_values(ascending=False))

    st.text("Accuracy over LLM and Language (FPR@10%)")
    c1, c2 = st.columns(2)
    with c1:
        if acc_LLM:  st.bar_chart(pd.Series(acc_LLM).sort_values(ascending=False))
    with c2:
        if acc_language: st.bar_chart(pd.Series(acc_language).sort_values(ascending=False))
    ###############################################
    metrics = {
        "LLM_accuracy":      {"acc_FPR@10%": acc_LLM, "acc": acc_LLM_50}, 
        "Language_accuracy": {"acc_FPR@10%": acc_language, "acc": acc_language_50}, 
        "threshold" : {"threshold at 10%": m_thr_10, "threshold at 1%": m_thr_1},
        "Graph":   {"F1": {'Macro_average': m_f1, 'std': f1_err}, "TPR@10%": {'Macro_average': m_t10, 'std':t10_err}, "TPR@1%": {'Macro_average': m_t01, 'std':t01_err}}
    }

    # 1) Readable JSON
    st.subheader("JSON")
    st.json(metrics)

    return





def head():

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    mpl.rcParams["hatch.color"] = "white"

    handles = [
        Patch(facecolor="#bf8488", edgecolor="black", label="F1 Score"),
        Patch(facecolor="#a3b1cd", edgecolor="black", hatch="////", label="TPR@FPR=10%"),
        Patch(facecolor="#4f709f", edgecolor="black", hatch="\\\\\\\\", label="TPR@FPR=1%"),
    ]

    # width x height in pollici + dpi
    fig, ax = plt.subplots(figsize=(8, 0.45), dpi=200, facecolor="white")  # 
    ax.axis("off")
    ax.legend(
        handles=handles, ncol=3, loc="center", frameon=False,
        handlelength=0.9, handleheight=0.9, fontsize=11  #
    )

    #
    st.pyplot(fig)



def auto_compute(ds:Dataset, split_col:str="LLM", method = 'we', opposite=False):
    head()
    labels_by_llm,  scores_by_llm  = split_by(ds, split_col="LLM") # dict 
    labels_by_lang, scores_by_lang = split_by(ds, split_col="language") # dict 
    labeles, scores = split_by_kfold(ds)
    compute_graph(labeles, scores ,labels_by_llm, scores_by_llm, labels_by_lang, scores_by_lang, method=method, opposite=opposite)