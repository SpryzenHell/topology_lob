from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from topology_lob import benchmark_fd, run_experiment, run_walk_forward


def save_experiment(result, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result["metrics"], indent=2))
    pd.DataFrame({
        "event_index": result["test_event_index"],
        "score": result["test_probability"],
    }).to_csv(out / "test_scores.csv", index=False)

    topo = result["topology"]
    b0 = [c for c in topo.columns if c.startswith("betti0_")]
    b1 = [c for c in topo.columns if c.startswith("betti1_")]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(result["radii"], topo[b0].median(0), marker="o", label="Betti-0 median")
    ax.plot(result["radii"], topo[b1].median(0), marker="o", label="Betti-1 median")
    ax.set(xlabel="Vietoris-Rips radius", ylabel="Betti count", title="L2 liquidity topology")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "betti_curves.png", dpi=170)
    plt.close(fig)

    rep = result["metrics"]["stationarity"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [r["d"] for r in rep["candidates"]],
        [r["adf_pvalue"] for r in rep["candidates"]],
        marker="o",
    )
    ax.axhline(0.05, linestyle="--", label="ADF threshold")
    ax.set_yscale("log")
    ax.set(
        xlabel="d",
        ylabel="ADF p-value",
        title=f"Stationarity selection (d={rep['selected_d']:.2f})",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fractional_stationarity.png", dpi=170)
    plt.close(fig)

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Topology LOB run</title>
<style>
body{{font:15px system-ui;max-width:1050px;margin:36px auto;padding:0 18px}}
section{{border:1px solid #ddd;border-radius:12px;padding:18px;margin:14px 0}}
pre{{background:#f7f7f7;padding:12px;border-radius:8px;overflow:auto}}
img{{max-width:100%}}
</style>
<h1>Topology LOB</h1>
<p>Data: <code>{result['metrics']['data_source']}</code></p>
<section><h2>Focal OOS</h2><pre>{json.dumps(result['metrics']['focal_loss'], indent=2)}</pre></section>
<section><h2>Topology</h2><pre>{json.dumps(result['metrics']['tda'], indent=2)}</pre><img src="betti_curves.png"></section>
<section><h2>Fractional differentiation</h2><pre>{json.dumps(result['metrics']['stationarity'], indent=2)}</pre><img src="fractional_stationarity.png"></section>
<p>Resume reference IC: <b>{result['metrics']['target_resume_ic']}</b>; target/reference only.</p>
"""
    (out / "index.html").write_text(html)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("demo")
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--data")
    p.add_argument("--out", default="results/demo")
    p.add_argument("--require-gtda", action="store_true")
    p.add_argument("--require-cuda", action="store_true")

    p = sub.add_parser("ablation")
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--out", default="results/ablation")

    p = sub.add_parser("walk-forward")
    p.add_argument("--config", default="configs/demo.json")
    p.add_argument("--out", default="results/walk_forward.json")

    p = sub.add_parser("ffd-benchmark")
    p.add_argument("--events", type=int, default=200000)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--d", type=float, default=0.45)
    p.add_argument("--out", default="results/fractional_diff_benchmark.json")

    a = ap.parse_args()

    if a.cmd == "demo":
        cfg = json.loads(Path(a.config).read_text())
        cfg["data_path"] = a.data
        cfg["require_gtda"] = a.require_gtda or cfg.get("require_gtda", False)
        result = run_experiment(cfg)
        if a.require_cuda and not result["metrics"]["gpu_used"]:
            raise SystemExit("CUDA required but the CUDA backend was not used")
        save_experiment(result, Path(a.out))
        print(json.dumps(result["metrics"], indent=2))

    elif a.cmd == "ablation":
        cfg = json.loads(Path(a.config).read_text())
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        experiments = {
            "micro_only": {"include_topology": False, "include_fracdiff": False},
            "micro_plus_fd": {"include_topology": False, "include_fracdiff": True},
            "micro_plus_topology": {"include_topology": True, "include_fracdiff": False},
            "full_focal": {"include_topology": True, "include_fracdiff": True},
        }
        for name, flags in experiments.items():
            result = run_experiment({**cfg, **flags})
            m = result["metrics"]
            rows.append({
                "experiment": name,
                "selected_d": m["selected_d"],
                "focal_pearson_ic": m["focal_loss"]["pearson_ic"],
                "focal_rank_ic": m["focal_loss"]["rank_ic"],
                "focal_roc_auc": m["focal_loss"]["roc_auc"],
                "logloss_pearson_ic": m["logloss_baseline"]["pearson_ic"],
            })
        (out / "summary.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2))

    elif a.cmd == "walk-forward":
        cfg = json.loads(Path(a.config).read_text())
        result = run_walk_forward(cfg)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))

    else:
        result = benchmark_fd(a.events, a.width, a.d)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()