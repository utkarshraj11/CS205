import numpy as np
import pandas as pd
from time import time


def preprocess(df, sample_per_class=None, random_seed=42):  # The ultimate answer to life, the universe, and machine learning models LOL
    # https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/
    print("PREPROCESSING......")

    # drop leakage / redundant
    df = df.drop(columns=["PitNextLap", "Normalized_TyreLife"])
    # print("Dropped: PitNextLap (future leakage), Normalized_TyreLife (redundant)")

    # drop null Compound
    before = len(df)
    df = df.dropna(subset=["Compound"])
    # print(f"Dropped {before - len(df)} rows with null Compound  ({len(df)} remain)")

    # filter outlier lap times to avoid skewing distance calculations
    lap_cap = df["LapTime (s)"].quantile(0.99)
    before = len(df)
    df = df[df["LapTime (s)"] <= lap_cap]
    # print(f"Removed {before - len(df)} laps with LapTime > {lap_cap:.1f}s (99th pct)  ({len(df)} remain)")

    # drop rare compounds since we won't have enough data to learn from them. Also, for safety they don't really allow usage of these tires anymore.
    rare = ["INTERMEDIATE", "WET"]
    before = len(df)
    df = df[~df["Compound"].isin(rare)]
    # print(f"Removed {before - len(df)} rows with compounds {rare}  ({len(df)} remain)")

    # encode Compound
    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3}
    df = df.copy()
    df["Compound"] = df["Compound"].map(compound_map)
    # print(f"Compound encoded: {compound_map}")

    # encode Driver and Race. Sorted alphabetically, 1-indexed for display. Not by actual driver's number.
    driver_map = {d: i+1 for i, d in enumerate(sorted(df["Driver"].unique()))}
    race_map   = {r: i+1 for i, r in enumerate(sorted(df["Race"].unique()))}
    df["Driver"] = df["Driver"].map(driver_map)
    df["Race"]   = df["Race"].map(race_map)
    # print(f"Driver encoded ({len(driver_map)} drivers → 1–{len(driver_map)})")
    # print(f"    Driver map : {driver_map}")
    # print(f"    Race encoded ({len(race_map)} races → 1–{len(race_map)})")
    # print(f"    Race map   : {race_map}")

    # balance classes so that we have equal pit/no-pit.
    pit_rows    = df[df["PitStop"] == 1]
    no_pit_rows = df[df["PitStop"] == 0]
    n_pit = len(pit_rows)

    if sample_per_class is None:
        n_each = n_pit  # use ALL pit laps; match with same count of no-pit since no-pit is more common. This results in a balanced dataset with 2*n_pit rows.
    else:
        n_each = min(sample_per_class, n_pit)

    pit_sample    = pit_rows.sample(n=n_each, random_state=random_seed)
    no_pit_sample = no_pit_rows.sample(n=n_each, random_state=random_seed)
    df_bal = pd.concat([pit_sample, no_pit_sample]).sample(frac=1, random_state=random_seed)    # shuffle 

    # print(f"\nClass balancing:")
    # print(f"    Raw     — pit: {n_pit:,}, no-pit: {len(no_pit_rows):,}")
    # print(f"    Sampled — {n_each:,} pit + {n_each:,} no-pit = {len(df_bal):,} rows")

    # Z-normalize because KNN is distance-based
    feature_cols = [
        "LapNumber", "Stint", "TyreLife", "Position",
        "LapTime (s)", "LapTime_Delta", "Cumulative_Degradation",
        "RaceProgress", "Position_Change",
        "Compound", "Driver", "Race", "Year"
    ]
    X = df_bal[feature_cols].copy().astype(float)   # ensure all features are float
    stds = X.std().replace(0, 1)    # avoid division by zero
    X_norm = (X - X.mean()) / stds  # Z-normalize
    # print(f"[8] Z-normalized {len(feature_cols)} features: {feature_cols}")

    # recode labels 0/1 → 1/2 to avoid 0 causing issues
    labels = df_bal["PitStop"].values.astype(int) + 1   # 0 → no pit = 1, 1 → pit = 2
    features = X_norm.values    # convert to numpy array
    cls, counts = np.unique(labels, return_counts=True) # check class balance
    # print(f"Class labels: " + ", ".join(f"class {c}={n:,}" for c, n in zip(cls, counts)))
    # print(f"    class 1 = no pit,  class 2 = pit this lap")
    # print(f"\nFinal : {features.shape[0]:,} instances × {features.shape[1]} features")

    return labels, features, feature_cols


def loo_accuracy(labels, features, feature_subset):
    X = features[:, list(feature_subset)]
    n = len(labels)
    correct = 0
    for i in range(n):
        diffs = X - X[i]    # compute difference to all other points
        dists = np.einsum('ij,ij->i', diffs, diffs) # squared L2 distance
        dists[i] = np.inf   # ignore self by setting distance to infinity
        if labels[np.argmin(dists)] == labels[i]:   # check if nearest neighbor has the same label
            correct += 1
    return correct / n
 
 
def forward_selection(labels, features, feature_names):
    total_features = features.shape[1]
    current_set = set()
    best_overall_set = set()
    best_overall_acc = 0.0

    print(f"\nBeginning Forward Selection search.\n")
 
    for _ in range(total_features):
        best_acc_level = -1
        best_feat_level = None
 
        for f in range(total_features):
            if f in current_set:
                continue
            candidate = current_set | {f}
            acc = loo_accuracy(labels, features, candidate)
            names = [feature_names[x] for x in sorted(candidate)] 
            print(f"        Using feature(s) {{{', '.join(names)}}}  "
                  f"accuracy is {acc*100:.2f}%")
            if acc > best_acc_level:
                best_acc_level = acc
                best_feat_level = f
 
        current_set = current_set | {best_feat_level}
        names_cur = [feature_names[x] for x in sorted(current_set)]
 
        if best_acc_level > best_overall_acc:
            best_overall_acc = best_acc_level
            best_overall_set = set(current_set)
        else:
            print("\n(Warning: Accuracy decreased! Continuing in case of local maxima)")
 
        print(f"\nFeature set {{{', '.join(names_cur)}}} was best, accuracy is {best_acc_level*100:.2f}%\n")
 
    best_names = [feature_names[x] for x in sorted(best_overall_set)]
    print(f"Finished search!!")
    print(f"Best feature subset : {{{', '.join(best_names)}}}")
    print(f"Best accuracy       : {best_overall_acc*100:.2f}%")
    return best_overall_set, best_overall_acc
 
 
def backward_elimination(labels, features, feature_names):
    total_features = features.shape[1]
    current_set = set(range(total_features))
    best_overall_set = set(current_set)
    best_overall_acc = loo_accuracy(labels, features, current_set)

    print(f"\nBeginning Backward Elimination search.\n")
 
    for _ in range(total_features - 1):
        best_acc_level = -1
        worst_feat = None
 
        for f in current_set:
            candidate = current_set - {f}
            acc = loo_accuracy(labels, features, candidate)
            names = [feature_names[x] for x in sorted(candidate)]
            print(f"        Using feature(s) {{{', '.join(names)}}}  accuracy is {acc*100:.2f}%")
            if acc > best_acc_level:
                best_acc_level = acc
                worst_feat = f
 
        current_set = current_set - {worst_feat}
        names_cur = [feature_names[x] for x in sorted(current_set)]
 
        if best_acc_level > best_overall_acc:
            best_overall_acc = best_acc_level
            best_overall_set = set(current_set)
        else:
            print("\n(Warning: Accuracy decreased! Continuing in case of local maxima)")
 
        print(f"\nFeature set {{{', '.join(names_cur)}}} was best, accuracy is {best_acc_level*100:.2f}%\n")
 
    best_names = [feature_names[x] for x in sorted(best_overall_set)]
    print(f"Finished search!!")
    print(f"Best feature subset : {{{', '.join(best_names)}}}")
    print(f"Best accuracy       : {best_overall_acc*100:.2f}%")
    return best_overall_set, best_overall_acc
 
 
DATASET_PATH = "f1_strategy_dataset_v4.csv"

def main():
    print("F1 Pit Stop Feature Selection")
    print(f"\nDataset: {DATASET_PATH}\n")

    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns\n")

    # Prompt sample size choice
    print("Run on full dataset or a smaller sample?")
    print("  Default (Enter) : full dataset (101,372 rows unbalanced)")
    print("  Enter a number  : instances per class")
    raw = input("\nInstances per class [default = full]: ").strip()

    if raw == "":
        sample_per_class = None
        print("→ Running on full dataset.\n")
    else:
        sample_per_class = int(raw)
        print(f"→ Sampling {sample_per_class:,} per class ({sample_per_class*2:,} total).\n")

    # Preprocess
    labels, features, feature_names = preprocess(df, sample_per_class=sample_per_class)
    n, num_features = features.shape

    # Baseline: all features 
    print(f"\nThis dataset has {num_features} features with {n:,} instances.")
    print('\nRunning nearest neighbor with all features using "leaving-one-out" evaluation ...')
    t0 = time()
    all_acc = loo_accuracy(labels, features, set(range(num_features)))
    print(f"Accuracy with all {num_features} features: {all_acc*100:.2f}%  "
          f"({time()-t0:.1f}s)\n")

    # Forward Selection
    t0 = time()
    fwd_set, fwd_acc = forward_selection(labels, features, feature_names)
    fwd_time = time() - t0
    print(f"[Forward Selection completed in {fwd_time:.1f}s]\n")

    # Backward Elimination
    t0 = time()
    bwd_set, bwd_acc = backward_elimination(labels, features, feature_names)
    bwd_time = time() - t0
    print(f"[Backward Elimination completed in {bwd_time:.1f}s]\n")

    # Summary
    print("SUMMARY")
    print(f"Dataset          : {DATASET_PATH}")
    print(f"Instances        : {n:,} ({n//2:,} pit + {n//2:,} no-pit)")
    print(f"Features         : {num_features}")
    print(f"Baseline (all)   : {all_acc*100:.2f}%")

    fwd_names = [feature_names[x] for x in sorted(fwd_set)]
    bwd_names = [feature_names[x] for x in sorted(bwd_set)]
    print(f"\nForward Selection : {fwd_acc*100:.2f}%  →  {{{', '.join(fwd_names)}}}")
    print(f"Backward Elim.    : {bwd_acc*100:.2f}%  →  {{{', '.join(bwd_names)}}}")


if __name__ == "__main__":
    main()