# Implementation status

Policy and behaviour are defined in **`design.md`**. **Phase 1 EDA / feature selection** is specified in **`eda.md`** (`design.md` §5.6). The training pipeline lives under **`src/darshan_surrogate/`** and **`src/train_parquet_surrogate.py`**.

**Optional later:** dataset checksum for publication (`design.md` §11).

**Leakage / tuning:** If the proxy-leakage filter (`--max-abs-corr-with-y`, default `0.95`) drops **too many** legitimate `total_*` / `detail_*` predictors (or harms validation error), **raise** the threshold slightly or review `out/eda/correlation_leakage_drops.csv` — see `design.md` §6. If it drops **too few** near-collinear proxies, **lower** the threshold. **`--no-corr-filter`** disables the filter only for debugging or ablations.

**EDA spec / leakage (needs follow-up):**

- **`eda.md` vs early drafts:** Resolve any leftover tension — e.g. counters like **`total_STDIO_OPENS`** (and similar) are **not** part of the POSIX intensity numerator and are **reasonable** candidates for `X`; they should **not** be excluded by a vague “drop READ/WRITE/BYTES” rule. Authoritative exclusions remain **`design.md` §6** + code.
- **Automatic intensity encoders:** Need a clear rule (or code path) to drop or flag variables that **mechanistically encode** the same quantity as `y` (beyond the explicit forbidden set) — e.g. near-duplicate byte totals, duration proxies; train-only correlation filter is one tool; document others if we add them.

**EDA / feature selection (open):**

- **Notebook or script** to run MI + RFR rankings on a train split, write **`out/eda/feature_rank_mutual_info.csv`** and **`out/eda/feature_rank_rfr.csv`**, and build the **union-of-top-K** pool (default **K = 20** per method — revisit if the union is too large or too small).
- **Iteration loop** (batch size **10**, stopping rule when validation gain flattens) is **not** yet automated in the main CLI — document runs manually or add a thin driver that calls the same split + metrics as `train_parquet_surrogate.py`.
- **Cadence:** aim for **2–4 quick** EDA/surrogate runs first before locking a pool.
- **Log transforms:** per-column (or per-family) rule for skewed counters (`eda.md`); persist in **`metadata.json`** for inference.
- **Reconcile** EDA’s frozen pool with the **“all `total_*` minus leakage”** default in `design.md` §5.2 — prefer a **CLI flag** (or config) for an explicit feature list, or a code path that restricts columns to the EDA pool.
