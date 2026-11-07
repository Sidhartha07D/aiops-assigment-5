# SwiftParcel ETA assignment — clean-start instructions

Run all commands from the directory containing this README. Use Python 3.11 on a Linux ARM64 or x86_64 machine with at least two CPU cores. The measured results in `REPORT.md` came from the VM described there; latency and throughput will vary by machine.

## Prepare data and environment

```bash
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda create -n assignment5 python=3.11 pip -y
conda activate assignment5
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/partA:$PWD/partB:$PWD/partC"
python eta_data.py
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
python -m pytest -q partA/test_eta_app.py partB/test_gate.py
```

The generated `data/` files and MLflow SQLite database are intentionally absent from the submission.

## Part A: REST and load tests

In one terminal, after activating the environment and exporting `PYTHONPATH` as above:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export ETA_MODEL=data/eta_v1.joblib
uvicorn eta_app:app --host 127.0.0.1 --port 8000 --no-access-log
```

In another terminal:

```bash
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --batch 1 --out partA/loadtest_results.json
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100
```

For the capacity experiment, stop the server and restart it with `--workers 2`, keeping the other options the same; rerun the five-concurrency load command. Stop the server before starting Ray Serve on port 8000.

## Part B: MLflow

```bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow_eta.db"
python partB/eta_mlflow.py register
python partB/eta_mlflow.py gate
python partB/eta_mlflow.py gate --min-rel-gain 0.60
```

To repeat the shadow comparison, serve `models:/eta-model@prod` on port 5101 and `models:/eta-model@candidate` on port 5102, each with `mlflow models serve --env-manager local --workers 1`, then run `python partB/shadow_eta.py`. Keep the same `MLFLOW_TRACKING_URI` in both server terminals. Stop both servers afterward.

## Part C: Ray Serve

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python partC/verify_ray.py
```

The script starts a local Ray cluster with two CPUs, deploys `build(weight=0.2)`, checks all five required properties, and shuts it down. Add `--keep-running` to pause before shutdown for a localhost browser demonstration.

## Part D: smoke release gate

Start the Part A v1 server again, then run:

```bash
python partD/smoke_eta.py --url http://127.0.0.1:8000/v1/models/eta:predict --expect-version v1 --max-p95-ms 80
```

The command exits 0 only when every check passes, and 1 otherwise. For a deliberate failure check, use `--expect-version v2` against the v1 server.
