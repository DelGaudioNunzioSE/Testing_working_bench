import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use("default")
# graph
y   = np.asarray([5,6], dtype=float)
err = np.asarray([0.2,0.3], dtype=float)


x = np.arange(len(y))                  # [0,1,2]
colors  = ["#c98f92", "#b9c6e8", "#345a8a"]
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
ax.set_xticklabels(["we"], rotation=30, ha="right")

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
plt.show()
