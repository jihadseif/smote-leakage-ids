"""
SMOTE leakage in intrusion-detection evaluation.

Compares two ways of evaluating the same model on imbalanced network-attack data:

  leaky : oversample the WHOLE dataset with SMOTE, then split / cross-validate.
          Synthetic points built from test samples end up in training, and
          synthetic points sit in the test set, so scores are inflated.
  safe  : split first, oversample the TRAINING data only (or inside each CV fold
          via an imblearn Pipeline). The test data stays untouched.

Runs on a synthetic, IDS-shaped dataset by default, or on your own CSV:

    python leakage_demo.py
    python leakage_demo.py --csv features.csv --label target
"""
import argparse
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        y = LabelEncoder().fit_transform(df[args.label].astype(str))
        X = df.drop(columns=[args.label]).select_dtypes("number").fillna(0).to_numpy()
        return X, y, f"{args.csv} ({len(y):,} rows)"
    # IDS-shaped synthetic data: one dominant "normal" class, four rarer attack classes.
    X, y = make_classification(
        n_samples=args.n, n_features=30, n_informative=10, n_redundant=8,
        n_classes=5, n_clusters_per_class=2, weights=[0.82, 0.10, 0.05, 0.02, 0.01],
        class_sep=0.7, flip_y=0.03, random_state=0)
    return X, y, f"synthetic IDS-shaped data ({len(y):,} rows, 5 classes, rarest = 1%)"


def model(seed):
    return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed)


def scores(y_true, y_pred, rare):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": accuracy_score(y_true, y_pred),
        "rarest_class_recall": recall_score(y_true, y_pred, labels=[rare], average="macro"),
    }


def holdout(X, y, seed, leaky, rare):
    if leaky:
        Xr, yr = SMOTE(random_state=seed).fit_resample(X, y)
        Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=0.3, stratify=yr, random_state=seed)
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
        Xtr, ytr = SMOTE(random_state=seed).fit_resample(Xtr, ytr)
    m = model(seed).fit(Xtr, ytr)
    return scores(yte, m.predict(Xte), rare)


def cv(X, y, seed, leaky):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    if leaky:
        X, y = SMOTE(random_state=seed).fit_resample(X, y)
        est = model(seed)
    else:
        est = Pipeline([("smote", SMOTE(random_state=seed)), ("rf", model(seed))])
    r = cross_validate(est, X, y, cv=skf, scoring="f1_macro")
    return {"macro_f1": r["test_score"].mean()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="CSV of numeric features plus a label column")
    ap.add_argument("--label", default="target", help="label column name (default: target)")
    ap.add_argument("--n", type=int, default=8000, help="rows of synthetic data (default: 8000)")
    ap.add_argument("--seeds", type=int, default=5, help="repeats with different seeds (default: 5)")
    ap.add_argument("--out", default="results", help="output folder (default: results)")
    ap.add_argument("--plot-only", action="store_true", help="redraw the chart from results_raw.csv")
    args = ap.parse_args()
    if args.plot_only:
        plot(pd.read_csv(f"{args.out}/results_raw.csv"), args.out)
        return

    X, y, desc = load_data(args)
    rare = int(np.argmin(np.bincount(y)))
    print(f"Data: {desc}")
    t0 = time.time()
    rows = []
    for s in range(args.seeds):
        for leaky in (True, False):
            name = "leaky (SMOTE before split)" if leaky else "safe (SMOTE on train only)"
            rows.append({"protocol": name, "setting": "70/30 hold-out", "seed": s, **holdout(X, y, s, leaky, rare)})
            rows.append({"protocol": name, "setting": "5-fold CV", "seed": s, **cv(X, y, s, leaky)})
        print(f"  seed {s} done")

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}/results_raw.csv", index=False)
    summary = (df.groupby(["setting", "protocol"])[["macro_f1", "accuracy", "rarest_class_recall"]]
                 .agg(["mean", "std"]).round(4))
    summary.to_csv(f"{args.out}/results_summary.csv")
    print("\n", summary.to_string(), sep="")
    plot(df, args.out)
    print(f"\nSaved to {args.out}/ in {time.time() - t0:.0f}s")


def plot(df, out):
    """Bar chart of macro F1, leaky vs safe, for each evaluation setting."""
    agg = df.groupby(["setting", "protocol"])["macro_f1"].agg(["mean", "std"]).reset_index()
    settings = ["70/30 hold-out", "5-fold CV"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    w = 0.36
    for i, (proto, color) in enumerate([("leaky (SMOTE before split)", "#c2283b"),
                                         ("safe (SMOTE on train only)", "#1d4ed8")]):
        sub = agg[agg.protocol == proto].set_index("setting").loc[settings]
        xs = np.arange(len(settings)) + (i - 0.5) * w
        ax.bar(xs, sub["mean"], w, yerr=sub["std"], color=color, label=proto, capsize=3)
        for x, v in zip(xs, sub["mean"]):
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(settings)), settings)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("Macro F1")
    ax.set_title("Same model, same data: where SMOTE runs changes the score", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(f"{out}/leakage_macro_f1.png")


if __name__ == "__main__":
    main()
