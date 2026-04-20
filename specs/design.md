# Design: I/O intensity surrogate (OLCF Darshan Parquet)

This document locks **design principles** and **defaults** for the project. The high-level roadmap lives in `plan.md`; **phase 1 EDA / feature-selection** procedure lives in **`eda.md`**; residual open items live in `issues.md`.

---

## 1. Purpose

Build a **regression surrogate** that estimates **job-level POSIX I/O intensity** from tabular Darshan summaries, so we can reason about I/O load without replaying raw logs.

**Non-goals for v1:** generative trace synthesis; sequence models over raw `.darshan` binaries (see `src/train_xgboost_darshan.py` only as a reference).

---

## 2. Definitions

| Term | Meaning |
|------|---------|
| **Surrogate** | A cheap predictor trained on trace **summaries** (tabular totals ± detail aggregates), not a full simulator. |
| **Grain** | One **job** = one training row after joining total and (when used) aggregated detail. |
| **Primary target** | **POSIX I/O intensity** (§4). Intensity is **computed**, not a native column. |

---

## 3. Data sources

### 3.1 `darshan_total` (`App*.parquet`) — **required**

- One file per application; **one row per job**.
- Full column inventory for the App10-style schema: `specs/total_features.md` (481 named columns including meta + `total_*`).
- Local reference: `data/App10.parquet`.

### 3.2 `darshan_detail` (`YYYY/M/D/<jobid>-<shard>.parquet`) — **combined with total**

- **Design intent:** Final model **input** = **total row features** + **job-level aggregates** derived from detail (after matching on `jobid` and date from `start_time`).
- Detail rows are per-record; they are **not** fed row-by-row into the same XGBoost row — only **aggregates** (§5.2, §3.3).

### 3.3 Joining total and detail

- **Keys:** `jobid` + calendar date derived from `start_time` (path `YYYY/M/D/` aligned to that date).
- **Missing detail (locked default):** When training uses detail-derived features, **drop** jobs that have no matching detail files. **Log** the **count and percentage** of dropped rows.
- **Fallback trigger:** If the fraction of jobs that would be dropped (missing detail) is **≥ 20%** of the pre-join total row count, switch to **`missing_detail=1`** plus **zero-filled** detail aggregates (and still log counts). The **20%** threshold is a starting point — adjust only with justification if joins are systematically incomplete.
- **Observability:** Training logs must always record dropped-job stats (or imputation flags) so join health is visible.

---

## 4. Target (locked)

**Primary regression target** (computed on the total row):

```text
posix_bytes = total_POSIX_BYTES_READ + total_POSIX_BYTES_WRITTEN   # NaNs → 0
y = posix_bytes / runtime   # seconds; exclude jobs with runtime ≤ 0
```

- **Training label:** Fit the model on **`z = log1p(y)`** (log1p of intensity) to stabilise heavy tails; **report metrics on `y`** in original space (inverse transform predictions for RMSE/MAE/R² vs true `y`).
- **Do not** put `y` or `posix_bytes` into `X` as features (see §6).

**Heavy-tail / “large jobs” (locked process):** Before changing the objective (weights, robust loss), run **EDA** to quantify how prevalent **large** `y` values are (e.g. tail mass, top-percentile share). Only then consider alternative metrics or weighting — see §9.

**Optional EDA-only:** total POSIX bytes before division; detail-based skew — not additional targets unless explicitly extended later.

---

## 5. Inputs (features)

### 5.1 Meta and exclusions

- **`exe`:** **Omit** from v1 (masked strings; avoid spurious hash-like signals).
- **`uid`:** **Do not use** — **no precedence** in the model (omit entirely; not a feature).
- **`jobid`:** **Never** use as a feature.

### 5.2 `total_*` feature set (automatic + inventory)

- **Inventory:** All candidate columns are listed in `specs/total_features.md`.
- **Pipeline default:** Start from **all** `total_*` columns **except** those excluded for leakage (§6), then **drop zero-variance** columns on the training split. Optionally drop **>99% zero** columns after EDA.
- **Phase 1 (EDA):** Before committing to the full high-dimensional model, run the **feature-selection experiment** in **`specs/eda.md`** — mutual-information and tree-based rankings, an initial **union-of-top-K** pool (e.g. K≈40), then **iterative** add/remove in small batches with time-ordered validation. The **refined pool** feeds the “production” surrogate training; leakage rules in §6 are **not** relaxed by EDA.
- **Illustrative cross-stack signals** (examples only — the model may use hundreds of `total_*` features):  
  - **`total_H5D_READS`** — HDF5 read activity alongside POSIX.  
  - **`total_MPIIO_COLL_READS`** — collective MPI-IO read path.  
  - **`total_BGQ_NNODES`** — job scale on the BGQ/Summit-side metadata where present.  
  These illustrate *why* non-POSIX modules help: they capture **other I/O stacks and machine scale** without duplicating POSIX byte totals used in `y`.

### 5.3 Calendar

- Derive **month** (and optionally **day-of-week**, **hour**) from `start_time`.
- **Encoding:** **One-hot** month (and other calendar fields if used) so months are **not** ordered by magnitude.
- **Do not** put raw **`start_time`** or **`end_time`** into `X` — together they recover **duration** (or encode job order in ways that overlap the time split); calendar signals use **one-hot** encodings only (see §6).

### 5.4 Multi-app training

- Train **one global model** on **multiple** `App*.parquet` files concatenated.
- Include **`app_id`** (categorical: `App1`, `App2`, …) as a feature so the model can separate workload families.

### 5.5 Detail-derived aggregates (v2+)

When detail is joined, add **one row per job** after aggregation, e.g.:

1. **Distinct file count** — `nunique(filename)` over all detail rows for the job (across shards).
2. **Spread / imbalance of I/O** — prefer using **existing Darshan variance-style fields** where available (e.g. `POSIX_F_VARIANCE_RANK_BYTES`, `total_POSIX_F_VARIANCE_RANK_BYTES`-family in total, or analogous columns in detail), since they summarise rank imbalance directly.
3. **Fallback** if variance columns are insufficient: **coefficient of variation** or **max/mean** on **per-record** pooled `(POSIX_BYTES_READ + POSIX_BYTES_WRITTEN)` across **all shards** for the job (locked default from prior iteration).

**Shard aggregation:**

1. Within each shard, rows are per-record; optional per-file subtotals if needed.
2. **Across shards** for the same `jobid`: **sum** additive counts/bytes; pool per-record byte totals for CV / max-mean when variance features are not used.

### 5.6 Phase 1 — EDA and feature selection (first experiment)

**Normative companion spec:** **`specs/eda.md`** (families of candidates, ranking methods, iteration loop).

- **Goal:** Produce a **leakage-safe, ranked** input set so the main surrogate is not forced to ingest hundreds of weak or redundant columns before we understand signal.
- **Leakage:** Identical to §6. EDA **screening heuristics** (e.g. flagging timer-like names) **must not** override §6 — in particular, **do not** drop all columns whose names contain `READ` / `WRITE` / `BYTES`; non-POSIX counters (HDF5, MPI-IO, …) often **should** remain (`eda.md` explains; §6 lists direct exclusions).
- **Preprocessing (EDA):** Zero-variance drop on train; **log1p** (or log) on selected skewed counters; record transforms for inference.
- **Ranking (fit scores on train only):** **Mutual information** (`mutual_info_regression` + `SelectKBest` or equivalent) and **Random Forest** (or similar) **feature importances**, each producing a top-**K** list (starting point **K = 40** per method — tune with justification).
- **Initial pool:** **Union** of the two top-K lists (deduplicated), plus fixed context features (`app_id`, calendar one-hots) unless ablated.
- **Iteration:** Train a **small** surrogate on the pool; measure **RMSE / MAE** on the **time-ordered** validation slice; export importances; **add or remove** candidates in **batches** (e.g. **10** features); **stop** when marginal validation improvement **diminishes** (document stopping rule in run notes).
- **Handoff:** Freeze **feature list + preprocessing** for the full **XGBoost** pipeline (§8) and **`metadata.json`** (§11).

---

## 6. Leakage rules (non-negotiable)

1. **Intensity target:** Build `y` from `total_POSIX_BYTES_READ`, `total_POSIX_BYTES_WRITTEN`, and `runtime`.

2. **Exclude from `X` (direct leakage):**  
   - `total_POSIX_BYTES_READ`, `total_POSIX_BYTES_WRITTEN` (parts of the `posix_bytes` numerator)  
   - **`runtime`** (denominator of `y`)  
   - **`start_time`**, **`end_time`** — raw timestamps are excluded; use calendar one-hot only (§5.3).  
   (Other modules’ `total_*` columns, including non-POSIX bytes such as `total_H5D_BYTES_READ`, **are allowed** in `X` — they do not trivially reconstruct POSIX `posix_bytes`.)

3. **Proxy leakage (train-fit filter):** On the **training split only**, optionally drop `total_*` and `detail_*` (excluding `month_*` and `app_id_code`) whose **Pearson |corr(x, y)|** exceeds a threshold (default **0.95**). This catches near-collinear proxies for the target; statistics are **never** computed on the validation fold for this step. The dropped list is logged in **`out/eda/correlation_leakage_drops.csv`** and in model **`metadata.json`**. CLI: `--max-abs-corr-with-y`, `--no-corr-filter` to disable (not recommended).

4. **Split:** **Time-ordered** — train on earlier `start_time`, validate on a later window. Primary reported scores use this split, **not** a random shuffle.

5. **Tuning:** Optuna + early stopping on the validation window; **no nested cross-validation** (cost vs gain; see §8).

---

## 7. Preprocessing

| Step | Rule |
|------|------|
| **Constants** | Drop zero-variance columns on **train**; persist list for inference. |
| **Missing** | `fillna(0)` on counter columns where absence = no activity. |
| **Target** | Train on `log1p(y)`; inverse `expm1` for predictions when reporting error on `y`. |
| **Sparse** | Optional: drop columns with >99% zeros on train after checking importance. |

---

## 8. Model & tuning

- **Algorithm:** **XGBoost** regressor (`XGBRegressor`).
- **Tuning:** **Optuna** over hyperparameters; **stop** when validation score shows **no improvement** for **patience** rounds (or equivalent early stopping on study / trials), **not** nested CV.
- **Baseline:** Dummy predictor (e.g. training-median `y`) on the **same validation window**.

---

## 9. Evaluation

- On **held-out time slice**, report **RMSE**, **MAE**, **R²** on **original `y`** (after inverse transform from `log1p` predictions).
- **Optional (after EDA on tail):** report the same metrics **with and without** the top **p**% of `y` (e.g. p=1 or 5) if large jobs dominate error — only if EDA shows heavy-tail concern.
- **Feature importances** (gain); optional **SHAP** on validation for interpretability.

---

## 10. Data volume & scope

- Full archive ~48 GB; **subset** of apps/months for dev; expand until metrics stabilise.
- **Multi-app** global model with **`app_id`** as above (§5.4).

---

## 11. Reproducibility

- Save model (JSON/UBJ) + **feature list** + preprocessing metadata (dropped columns, encodings, `app_id` mapping).
- Pin random seeds; record **`conda activate ml_base`** (or equivalent env file).
- **Dataset checksum / download pinning:** deferred until publication needs (Globus URL + date/checksum).

### 11.1 EDA & run artifacts (CSV)

- **Record** join statistics, EDA summaries, and metric tables in **CSV** files under a single top-level **`out/`** tree, separated by purpose, e.g.:
  - **`out/eda/`** — join stats, **`y`** percentiles / histogram summaries, tail diagnostics.
  - **`out/train/`** — validation metrics, optional Optuna trial logs, feature-importance tables.
  - (Add other subdirs under `out/` as needed, e.g. `out/models/` for saved model JSON.)
- **Minimum suggested exports:** (1) total↔detail join — dropped count, drop %, whether 20% fallback applied; (2) **`y`** — histogram bins or percentile rows (e.g. p50, p90, p99); (3) training run — validation RMSE/MAE/R² per run id; (4) optional proxy-leakage — **`correlation_leakage_drops.csv`** when any `total_*` / `detail_*` columns are dropped by the train-only correlation rule (§6); (5) **phase 1 EDA** (`eda.md`) — e.g. **`feature_rank_mutual_info.csv`**, **`feature_rank_rfr.csv`**, **`feature_pool_iteration_*.csv`** or a single **`feature_pool_final.csv`** listing the chosen columns per iteration or at freeze.
- **Purpose:** offline analysis and reporting; not a substitute for version-controlling **code** and **saved models**.

---

## 12. Visualization (locked stack)

- **Matplotlib** for figures: data distributions, metric curves, predicted vs actual, feature importance, inference examples.

---

## 13. Implementation

- **Training module:** `src/train_parquet_surrogate.py` — primary Parquet pipeline (create when ready).
- **Data loading (locked policy):** Use **pandas only** — `pd.read_parquet` and `pd.concat` for multiple `App*.parquet` files. **No** separate PyArrow-Dataset code path and **no** loader benchmark in scope. (The `pyarrow` package may still be installed as pandas’ Parquet engine; that is an implementation detail, not a second API surface in project code.)

---

## 14. Quality checks (local, optional)

- **No** required **pre-commit** hooks and **no** automated **CI** (GitHub Actions, etc.) — **ignore** remote CI for this project unless priorities change.
- **Optional:** run **`pytest`** (see `tests/` — leakage guards + small synthetic checks) and **lint** locally before a big training job or a shareable commit. Convenience only, not a gate.

---

## 15. Ethics & data use

- Dataset already masks sensitive fields; treat sample rows as **non-sensitive** for internal reporting unless institutional policy requires extra review.

---

## 16. Code map

| Path | Role |
|------|------|
| `specs/total_features.md` | Column inventory for `darshan_total` |
| `specs/eda.md` | Phase 1 — EDA, ranking, iterative feature pool (`design.md` §5.6) |
| `data/data-archive-demo.ipynb` | Loading patterns for total/detail Parquet |
| `data/App10.parquet` | Local development slice |
| `src/explore_parquet_schema.py` | Column inventory helper |
| `src/train_parquet_surrogate.py` | Parquet surrogate pipeline (CLI entry) |
| `src/darshan_surrogate/` | Library: I/O, targets, features, split, detail join, training, artifacts, **`leakage.py`** (forbidden columns + train-only correlation filter) |
| `requirements.txt` | Pip dependencies; use with **`conda activate ml_base`** then `pip install -r requirements.txt` |
| `src/train_xgboost_darshan.py` | Reference: raw `.darshan` path only |
| `out/eda/`, `out/train/`, `out/models/` | CSV exports, saved XGBoost JSON + `metadata.json` (`design.md` §11.1) |

---

## 17. Acceptance checklist

- [ ] `y` / `log1p(y)` and `X` satisfy §6 (direct exclusions + optional train-only proxy filter; raw `start_time`/`end_time` not in `X`).
- [ ] Phase 1 EDA (`eda.md`) run or planned: MI + RFR ranking, iterative pool, frozen feature list before full surrogate.
- [ ] Time-based validation + Optuna patience / early stopping (no nested CV).
- [ ] Constant columns dropped on train; list frozen for inference.
- [ ] Metrics on original `y` + baseline documented; tail EDA done before optional weighted metrics (§4, §9).
- [ ] Model + metadata saved; `app_id` handling documented if multi-app.
- [ ] Detail join: drop + logged counts (§3.3); fallback if missing-detail rate ≥ 20% (§3.3).
- [ ] Pandas-only loading (§13).
- [ ] Matplotlib figures for §12 categories.
- [ ] CSV artifacts for join + EDA + key metrics (`design.md` §11.1).
- [ ] (Optional) Local smoke/lint before major runs (`design.md` §14).
