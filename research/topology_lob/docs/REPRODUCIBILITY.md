# Reproducibility checklist

This is the runbook for producing a result that another developer can reproduce and audit.

## Environment

Use Python 3.11 and install the pinned application environment:

```bash
cd research/topology_lob
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[tda,dev]"
```

Record the interpreter and package environment with the result:

```bash
python --version
python -m pip freeze > results/pip-freeze.txt
```

The generated environment file is a run artifact and should only be committed when the run is intended for archival purposes.

## Code checks

Run the tests and the synthetic pipeline:

```bash
python -m pytest -q
python run.py demo --config configs/demo.json --out results/demo --require-gtda
python run.py walk-forward --config configs/demo.json --out results/walk_forward.json
python run.py ablation --config configs/demo.json --out results/ablation
```

The production TDA path must report:

```json
"backend": "giotto-tda"
```

For a CUDA-verified run, add `--require-cuda` and confirm:

```json
"gpu_used": true
```

A CPU run is not treated as evidence for the CUDA claim.

## Real-data record

For every real-data run, retain the input dataset identifier or path, input SHA-256 checksum when available, exact configuration, exact command line, Python version, package versions, and the generated result files.

Do not replace an unsuccessful run by editing its metric file.

## FI-2010 boundary rule

The FI-2010 benchmark code constructs forward-return labels independently within the training stream and each held-out segment. The target for the end of one segment therefore cannot use observations from the next segment.

## Resume metric

The value 0.064 is stored as a reproduction target. It is not a default result. A resume claim should be updated only after the intended real L2 dataset has produced the corresponding measured out-of-sample metric.
