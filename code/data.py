"""
Dataset loading/caching for the model-collapse harness.

Fetches datasets via sklearn.datasets.fetch_openml, caches the cleaned
frame to code/data_cache/<name>.pkl (pickle of a dict with df/meta), drops
NaN rows, caps to at most MAX_ROWS rows, and produces a fixed stratified
train/holdout split (seed 0).
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MAX_ROWS = 20000
HOLDOUT_FRAC = 0.2
SPLIT_SEED = 0

# name -> (openml_name, version, target_col, fallback)
DATASET_SPECS = {
    "adult": ("adult", 2, "class", None),
    "credit-g": ("credit-g", 1, "class", None),
    "bank-marketing": ("bank-marketing", 1, "class", ("electricity", 1, "class")),
}


def _fetch_raw(name, version, target_col):
    bunch = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    df = bunch.frame.copy()
    # bunch.frame includes the target column already (as_frame=True keeps it
    # in .frame), but OpenML sometimes names it differently in case
    # (e.g. bank-marketing's real target column is "Class", not "class").
    # Identify it via feature_names (whatever's left over is the target) and
    # rename in place -- do NOT just append bunch.target under target_col,
    # since that would leave the original target column in the *features*,
    # leaking the label.
    feat = set(bunch.feature_names)
    non_feat_cols = [c for c in df.columns if c not in feat]
    if len(non_feat_cols) == 1 and non_feat_cols[0] != target_col:
        df = df.rename(columns={non_feat_cols[0]: target_col})
    elif target_col not in df.columns:
        df[target_col] = bunch.target
    # safety net: drop any leftover column that duplicates the target under
    # a different name/case (would otherwise leak the label into features)
    dup_cols = [c for c in df.columns
                if c != target_col and c.lower() == target_col.lower()]
    if dup_cols:
        df = df.drop(columns=dup_cols)
    return df


def _clean(df, target_col):
    df = df.dropna(axis=0, how="any").reset_index(drop=True)
    # drop columns that are entirely NaN or constant (uninformative, breaks metrics)
    nunique = df.nunique(dropna=False)
    const_cols = [c for c in nunique.index if nunique[c] <= 1 and c != target_col]
    if const_cols:
        df = df.drop(columns=const_cols)
    return df


def _cap_rows(df, target_col, max_rows, seed):
    if len(df) <= max_rows:
        return df
    # stratified subsample to preserve class balance
    df_sub, _ = train_test_split(
        df, train_size=max_rows, stratify=df[target_col], random_state=seed
    )
    return df_sub.reset_index(drop=True)


def _infer_types(df, target_col):
    numeric_cols = []
    categorical_cols = []
    for c in df.columns:
        if c == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return numeric_cols, categorical_cols


def load_dataset(name, max_rows=MAX_ROWS, holdout_frac=HOLDOUT_FRAC, split_seed=SPLIT_SEED,
                  force_refetch=False):
    """Returns dict: df_real (D_real, capped), df_holdout (H), numeric_cols,
    categorical_cols, target_col, meta (shape info)."""
    cache_path = os.path.join(CACHE_DIR, f"{name}.pkl")
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(cache_path) and not force_refetch:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        df = cached["df"]
        target_col = cached["target_col"]
    else:
        if name not in DATASET_SPECS:
            raise ValueError(f"Unknown dataset {name}")
        openml_name, version, target_col, fallback = DATASET_SPECS[name]
        try:
            df = _fetch_raw(openml_name, version, target_col)
        except Exception as e:
            if fallback is None:
                raise
            print(f"[data] fetch of {openml_name} failed ({e}); falling back to {fallback}")
            openml_name, version, target_col = fallback
            df = _fetch_raw(openml_name, version, target_col)
        df = _clean(df, target_col)
        with open(cache_path, "wb") as f:
            pickle.dump({"df": df, "target_col": target_col}, f)

    numeric_cols, categorical_cols = _infer_types(df, target_col)

    # fixed stratified split BEFORE capping, so holdout is drawn from the full
    # cleaned pool and D_real capping doesn't bias the holdout.
    df_train_pool, df_holdout = train_test_split(
        df, test_size=holdout_frac, stratify=df[target_col], random_state=split_seed
    )
    df_train_pool = df_train_pool.reset_index(drop=True)
    df_holdout = df_holdout.reset_index(drop=True)

    df_real = _cap_rows(df_train_pool, target_col, max_rows, split_seed)

    meta = {
        "name": name,
        "final_shape_full": list(df.shape),
        "n_real_pool": int(len(df_real)),
        "n_holdout": int(len(df_holdout)),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "target": target_col,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
    }
    return {
        "df_real": df_real,
        "df_holdout": df_holdout,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": target_col,
        "meta": meta,
    }


def load_all_and_report(names=None, out_json=None):
    names = names or list(DATASET_SPECS.keys())
    report = {}
    for name in names:
        d = load_dataset(name)
        report[name] = d["meta"]
        print(f"[{name}] shape(real)={d['df_real'].shape} shape(holdout)={d['df_holdout'].shape} "
              f"n_numeric={d['meta']['n_numeric']} n_categorical={d['meta']['n_categorical']} "
              f"target={d['meta']['target']}")
    out_json = out_json or os.path.join(RESULTS_DIR, "datasets.json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[data] wrote {out_json}")
    return report


if __name__ == "__main__":
    load_all_and_report()
