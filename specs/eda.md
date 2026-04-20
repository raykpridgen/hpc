# Exploratory Data Analysis: Selecting Input features

Output feature / target: I/O intensity
- Computed from reads + writes / runtime
- which features are these exactly?

# Exploration
Need a portable script that can access multiple parquets and summarize the features. 
- Start with totals for baseline
- Counts filled entries
- Majority value in entires
Goal is to get a good understanding of entire dataset before making assumptions about shape

The data will have the format:
darshan_share/
    darshan_detail/
        2021/
            {M}/ (single digits have no leading 0)
                {D}/ (same for day)
                    {JOBID}-0.parquet

Each parquet should be accounted for, and it should record:
features seen
- num times filled (not NAN)


# Preprocessing

## Eliminate Leakage
Explicitly drop all columns containing:  BYTES, TIME
- this leaves i/o behavior to choose from

## Features to Consider
- Job scale: nprocs, num_ranks
- Metadata: POSIX_OPENS, num_files, unique_files, etc
- Access patterns: conescutive_reads, max_read_size
- IO libary: HDF5*, MPI*, STDIO* counters
- parallel FS hints: LUSTRE*

## Types
If features are group activated (ex MPI vars NAN for whole job, or active for whole job), consider sleecting these into new frames, and training seperate models per feature pool
- POSIX predictor
- MPI predictor
- etc

# Feature Selection Plan

## Processing
- Drop zero variance columns
- Log transform skewed counters, like large ones mentioned in design / issues

## Feature Ranking
- Mutual information: mutual_info_selection, SelectKBest
- Tree based importance: RFR

## Routine / Iteration
- Start with 40 top features found from union of ranking methods
- train small surrogate with features
- measure RMSE / MAE, construct feature imporance data
- Add remove features in batches of 10
- Stop when returns diminish
## Final Surrogate
- Use a pool of most imporant features
- Train on time pools as mentioned earlier
