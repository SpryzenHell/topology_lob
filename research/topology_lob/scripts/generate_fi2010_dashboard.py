from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "fi2010_validation_summary.json"
OUT = ROOT / "assets" / "figures" / "fi2010_validation_dashboard.png"


def main() -> None:
    data = json.loads(SUMMARY.read_text())
    days = data["days"]
    ablation = data["ablation"]
    names = list(ablation)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    x = np.array([d["day"] for d in days])

    ax[0, 0].plot(x, [d["pearson_ic"] for d in days], marker="o", label="Pearson IC")
    ax[0, 0].plot(x, [d["rank_ic"] for d in days], marker="o", label="Rank IC")
    ax[0, 0].axhline(0.0, linewidth=1)
    ax[0, 0].set_title("Daily FI-2010 signal stability")
    ax[0, 0].set_xlabel("held-out segment")
    ax[0, 0].set_ylabel("IC")
    ax[0, 0].legend()

    ax[0, 1].plot(x, [d["roc_auc"] for d in days], marker="o")
    ax[0, 1].set_title("Daily ROC-AUC")
    ax[0, 1].set_xlabel("held-out segment")
    ax[0, 1].set_ylabel("ROC-AUC")
    ax[0, 1].set_ylim(0.5, 0.8)

    xx = np.arange(len(names))
    ax[1, 0].bar(xx - 0.18, [ablation[n]["pearson_ic"] for n in names], 0.36, label="Pearson IC")
    ax[1, 0].bar(xx + 0.18, [ablation[n]["rank_ic"] for n in names], 0.36, label="Rank IC")
    ax[1, 0].axhline(0.0, linewidth=1)
    ax[1, 0].set_xticks(xx, names, rotation=25, ha="right")
    ax[1, 0].set_title("Pooled ablation: information coefficient")
    ax[1, 0].set_ylabel("IC")
    ax[1, 0].legend()

    ax[1, 1].bar(xx, [ablation[n]["roc_auc"] for n in names])
    ax[1, 1].set_xticks(xx, names, rotation=25, ha="right")
    ax[1, 1].set_title("Pooled ablation: ROC-AUC")
    ax[1, 1].set_ylabel("ROC-AUC")
    ax[1, 1].set_ylim(0.68, 0.72)

    fig.suptitle(
        "FI-2010 Decimal-Precision — verified benchmark and ablation",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
