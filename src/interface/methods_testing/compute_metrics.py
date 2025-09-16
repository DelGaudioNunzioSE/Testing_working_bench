import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import Dataset
from itertools import chain
from matplotlib.patches import Patch


def compute_metrics(y_true, y_score):
    # guard clauses
    if y_true is None or y_score is None:
        return None, None, None
    if len(y_true) == 0 or len(y_score) == 0 or len(y_true) != len(y_score):
        return None, None, None

    y_true  = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if np.isnan(y_score).any():
        y_score = np.nan_to_num(y_score, nan=0.0)

    # serve binario e entrambe le classi presenti
    classes = np.unique(y_true)
    if classes.size != 2:
        return None, None, None

    y_pred_05 = (y_score >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_05)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    # interp richiede x crescente e senza duplicati
    fpr_u, idx = np.unique(fpr, return_index=True)
    tpr_u = tpr[idx]

    tpr_at_10 = float(np.interp(0.10, fpr_u, tpr_u))
    tpr_at_01 = float(np.interp(0.01, fpr_u, tpr_u))
    return f1, tpr_at_10, tpr_at_01




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
    st.pyplot(fig, width='stretch')
    plt.close(fig)
    return 

    

# Exemples
#methods = {"LogRank": (y_true_logrank, y_score_logrank) }
#metrics = {name: compute_metrics(y, s) for name, (y, s) in methods.items()}





def split_by(ds:Dataset, split_col:str="LLM", no_split_col = 'Human') -> tuple[dict,dict]:
    split_col_val = ds.unique(split_col)
    labels_by_llm, scores_by_llm = {}, {}
    for val in split_col_val:
        sub = ds.filter(lambda b, v=val : [x in [v, no_split_col] for x in b[split_col]],    batched=True )  
        labels_by_llm[val] = sub["label"]
        scores_by_llm[val] = sub["score"]
    return labels_by_llm, scores_by_llm



def compute_graph(labels_by_llm:dict, scores_by_llm:dict, method = 'we'):

    fi :list= []
    tpr_at_10 :list= []
    tpr_at_01 :list= []
    n_none = 0
    for y_t, y_s in zip(labels_by_llm.values(), scores_by_llm.values()):
        fi_i, tpr_at_10_i, tpr_at_01_i = compute_metrics(y_true = y_t, y_score= y_s)
        if fi_i is not None and fi_i is not None and tpr_at_01_i is not None :
            fi.append(fi_i)
            tpr_at_10.append(tpr_at_10_i)
            tpr_at_01.append(tpr_at_01_i)
        else:
            n_none += 1


    fi        = np.asarray(fi, dtype=float)
    tpr_at_10 = np.asarray(tpr_at_10, dtype=float)
    tpr_at_01 = np.asarray(tpr_at_01, dtype=float)

    fi_avg, tpr_at_10_avg, tpr_at_01_avg = compute_metrics(y_true = list(chain.from_iterable(labels_by_llm.values())), y_score= list(chain.from_iterable(scores_by_llm.values())))


    fi_err        = float(np.nanstd(fi, ddof=1))        if fi.size        > 1 else float("nan")
    tpr_at_10_err = float(np.nanstd(tpr_at_10, ddof=1)) if tpr_at_10.size > 1 else float("nan")
    tpr_at_01_err = float(np.nanstd(tpr_at_01, ddof=1)) if tpr_at_01.size > 1 else float("nan")
    graph(y=[fi_avg, tpr_at_10_avg, tpr_at_01_avg], err=[fi_err, tpr_at_10_err, tpr_at_01_err], method=method )






def head():

    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use("default")
    mpl.rcParams["hatch.color"] = "white"

    handles = [
        Patch(facecolor="#bf8488", edgecolor="black", label="F1 Score"),
        Patch(facecolor="#a3b1cd", edgecolor="black", hatch="////", label="TPR@FPR=10%"),
        Patch(facecolor="#4f709f", edgecolor="black", hatch="\\\\\\\\", label="TPR@FPR=1%"),
    ]

    # largo e molto basso: width x height in pollici + dpi
    fig, ax = plt.subplots(figsize=(8, 0.45), dpi=200, facecolor="white")  # “più piccola e wide”
    ax.axis("off")
    ax.legend(
        handles=handles, ncol=3, loc="center", frameon=False,
        handlelength=0.9, handleheight=0.9, fontsize=11  # quadrati compatti
    )

    # usa tutta la larghezza del container, salva senza margini extra
    st.pyplot(fig)



def auto_compute(ds:Dataset, split_col:str="language", method = 'we'):
    labels_by_llm, scores_by_llm = split_by(ds = ds, split_col=split_col)
    head()
    compute_graph(labels_by_llm = labels_by_llm, scores_by_llm = scores_by_llm, method = method)



