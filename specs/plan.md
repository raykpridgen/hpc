# Surrogate modelling of HPC filesystem traces (OLCF Darshan archive)

**Locked design principles, targets, leakage rules, and evaluation:** see [`design.md`](design.md).

**Open implementation choices:** see [`issues.md`](issues.md).

---

## Overview

Train a surrogate model on published OLCF Summit Darshan **tabular** data to predict **POSIX I/O intensity** (bytes/s) at job granularity, using **XGBoost** with **Optuna** tuning and **time-based** validation.

---

## Dataset layout

**Source:** [OLCF Summit Darshan share](https://g-e320e6.63720f.75bc.data.globus.org/gen101/world-shared/doi-data/OLCF/202402/10.13139_OLCF_2305496/summit_darshan_share.tar.gz) (~48 GB unpacked).

```
darshan_share/
├── darshan_detail/     # per-job detail from darshan-parser (by date hierarchy)
├── darshan_total/      # one App*.parquet per application — “--total” style counters
└── data-archive-demo.ipynb
```

Local dev: `data/App10.parquet`; optional detail under `data/2021/...`.

---

## Model

- **Learner:** XGBoost regressor; train on **`log1p(y)`**, report errors on original **`y`** (see `design.md` §4, §7).
- **Tuning:** Optuna with **patience / no-improvement stop** — **no nested CV** (see `design.md` §8).
- **Validation:** Chronological time split (not random shuffle for reported scores).

---

## Preprocessing & derived fields

- Drop zero-variance `total_*` on train; `fillna(0)` where appropriate; column inventory in `specs/total_features.md`.
- **Target:** POSIX I/O intensity from total row; **features:** `nprocs`, **one-hot** calendar from `start_time`, **`app_id`** when multi-app, all non-leaking `total_*` (see `design.md` §5–§6); **`exe` and `uid` omitted**; **detail** optional aggregates joined to the same row (missing detail → **drop** + **log** counts when detail features are on).
- **Leakage:** Exclude `total_POSIX_BYTES_READ`, `total_POSIX_BYTES_WRITTEN`, and **`runtime`** from `X`.

---

## Training

- **Global model** on **multiple** apps; validate on a **later slice** of time.
- Record **RMSE**, **MAE**, **R²** vs baseline on **original `y`**.

---

## Visualization

**Matplotlib** — data distributions, metrics, predicted vs actual, feature importance, inference examples (`design.md` §12).

---

## Implementation

- **CLI:** `src/train_parquet_surrogate.py` — **`conda activate ml_base`** then `pip install -r requirements.txt` then `python src/train_parquet_surrogate.py ...` (`design.md` §13).
- **Library:** `src/darshan_surrogate/` — I/O, targets, features, split, detail join, training, artifacts.
- **Artifacts:** CSVs under **`out/eda/`**, **`out/train/`** (`design.md` §11.1). **Quality checks:** optional local only (`design.md` §14).
