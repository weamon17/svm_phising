"""Headless run of the preprocessing comparison (same function and FULL settings as notebook 02).

Writes outputs/preprocessing/<timestamp>/results.csv and grid_<variant>.csv.
"""

import sys
import time
from datetime import datetime

import pandas as pd

from src.data import PROCESSED_CSV, ROOT
from src.features import extract_features
from src.preprocessing import compare_preprocessing
from src.splits import group_split, stratified_sample


SEED, N_KERNEL, CV_FOLDS = 42, 8_000, 5
RBF_GRID = {"svc__C": [0.1, 0.3, 1, 3, 10, 30, 100], "svc__gamma": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]}
LINEAR_C = {"linearsvc__C": [0.001, 0.01, 0.1, 1, 10]}["linearsvc__C"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    hosts = pd.read_csv(PROCESSED_CSV, keep_default_na=False)
    train_all, test = group_split(hosts, seed=SEED)
    train_k = stratified_sample(train_all, hosts["phishing"], N_KERNEL, SEED)
    features = extract_features(hosts)
    start = time.perf_counter()
    results, grids = compare_preprocessing(features, hosts["phishing"], hosts["registered_domain"], train_k,
                                           train_all, test, RBF_GRID, LINEAR_C, CV_FOLDS, SEED)
    out = ROOT / "outputs" / "preprocessing" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    results.to_csv(out / "results.csv", index=False)
    for kind, grid in grids.items():
        grid.to_csv(out / f"grid_{kind.replace('+', '_')}.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(results.drop(columns="params").round(3).to_string(index=False))
        print(results[["preprocessing", "model", "params"]].to_string(index=False))
    print(f"{time.perf_counter() - start:.0f}s · saved {out}")


if __name__ == "__main__":
    main()
