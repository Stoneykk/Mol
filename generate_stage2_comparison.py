"""Stage 2: Our MoLFormer reproduction vs official c3-MoLFormer-1.1B (DeepChem split)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Data ──────────────────────────────────────────────────────────────────────

# Official c3-MoLFormer-1.1B (DeepChem scaffold split)
# Source: https://github.com/deepforestsci/chemberta3
#         results/images/Deepchem-splits-benchmark1.png
official_cls  = {"BBBP": (0.735, 0.019), "Tox21": (0.723, 0.012), "ClinTox": (0.839, 0.013)}
official_reg  = {"ESOL": (0.829, 0.019), "FreeSolv": (0.572, 0.023), "Lipophilicity": (0.728, 0.016)}

# Our reproduction (DeepChem scaffold split, AdamW, HuggingFace)
ours_cls = {"BBBP": (0.727, 0.006), "Tox21": (0.747, 0.004), "ClinTox": (0.989, 0.001)}
ours_reg = {"ESOL": (0.787, 0.019), "FreeSolv": (2.175, 0.026), "Lipophilicity": (0.686, 0.019)}

COLOR_OFFICIAL = "#2E86AB"
COLOR_OURS     = "#E74C3C"

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, (ax_cls, ax_reg) = plt.subplots(1, 2, figsize=(14, 6))

def plot_bars(ax, official, ours, ylabel, title, ylim=None, higher_better=True):
    datasets = list(official.keys())
    x = np.arange(len(datasets))
    w = 0.32

    off_means = [official[d][0] for d in datasets]
    off_stds  = [official[d][1] for d in datasets]
    our_means = [ours[d][0]     for d in datasets]
    our_stds  = [ours[d][1]     for d in datasets]

    bars1 = ax.bar(x - w/2, off_means, w, yerr=off_stds, capsize=5,
                   label="Official c3-MoLFormer-1.1B", color=COLOR_OFFICIAL,
                   alpha=0.85, error_kw={"elinewidth": 1.8, "ecolor": "k"})
    bars2 = ax.bar(x + w/2, our_means, w, yerr=our_stds, capsize=5,
                   label="Our Reproduction (AdamW)", color=COLOR_OURS,
                   alpha=0.85, error_kw={"elinewidth": 1.8, "ecolor": "k"})

    # value labels
    for bar, mean, std in zip(bars1, off_means, off_stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + (0.008 if higher_better else max(our_means)*0.02),
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_OFFICIAL)
    for bar, mean, std in zip(bars2, our_means, our_stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + std + (0.008 if higher_better else max(our_means)*0.02),
                f"{mean:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=COLOR_OURS)

    # win/lose markers
    for i, d in enumerate(datasets):
        om, um = official[d][0], ours[d][0]
        ours_wins = (um > om) if higher_better else (um < om)
        marker = "▲" if ours_wins else "▼"
        color  = "#27AE60" if ours_wins else "#E67E22"
        diff = abs(um - om)
        sign = "+" if ours_wins else "-"
        top = max(om + official[d][1], um + ours[d][1])
        offset = top * 0.04 if not higher_better else 0.03
        ax.text(i, top + offset, f"{sign}{diff:.3f}", ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(fontsize=9.5, loc="upper left" if not higher_better else "lower left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)

plot_bars(ax_cls, official_cls, ours_cls,
          ylabel="Test ROC-AUC (↑ higher is better)",
          title="Classification Tasks",
          ylim=(0.6, 1.08),
          higher_better=True)

plot_bars(ax_reg, official_reg, ours_reg,
          ylabel="Test RMSE (↓ lower is better)",
          title="Regression Tasks",
          higher_better=False)

fig.suptitle(
    "Stage 2: MoLFormer-c3-1.1B Reproduction vs Official\n"
    "(DeepChem Scaffold Split · 3 Seeds · AdamW vs FusedLAMB)",
    fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()

os.makedirs("assets", exist_ok=True)
out = "assets/stage2_comparison.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved to {out}")
