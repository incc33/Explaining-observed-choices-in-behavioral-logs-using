\
"""
EdNet(KT1) sample500 – cognitive/behavioral strategy discovery
- User-level strategy clustering (KMeans + GMM) + scoring baseline
- Strategy transition modeling:
    * Preferred: HMM (GaussianHMM via hmmlearn) on per-interaction signals
    * Fallback: Markov transitions on cluster labels (per interaction / per user)
- Paper-ready visualizations saved to outputs/figs
- Summary tables saved to outputs/tables
Run:
    python main.py --data ednet_step0_prepared_sample500.csv.xlsx
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Utils
# -----------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")

def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = list(df.columns)
    # exact
    for c in candidates:
        if c in cols:
            return c
    # normalized
    norm_map = {norm_col(c): c for c in cols}
    for c in candidates:
        k = norm_col(c)
        if k in norm_map:
            return norm_map[k]
    # substring
    for c in candidates:
        k = norm_col(c)
        for col in cols:
            if k in norm_col(col) or norm_col(col) in k:
                return col
    if required:
        raise KeyError(f"Missing required column. candidates={candidates} cols={cols}")
    return None

def clip_series(s: pd.Series, low_q=0.01, high_q=0.99) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo = x.quantile(low_q)
    hi = x.quantile(high_q)
    return x.clip(lower=lo, upper=hi)

def longest_run(arr: np.ndarray, value: int) -> int:
    best = cur = 0
    for x in arr:
        if int(x) == value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)

# -----------------------------
# Loading / preprocessing
# -----------------------------
@dataclass
class Cols:
    user: str
    t: str
    ts: str
    qid: str
    tags: str | None
    part: str | None
    elapsed: str | None
    reward: str

def load_data(path: Path) -> tuple[pd.DataFrame, Cols]:
    if not path.exists():
        raise FileNotFoundError(path)

    # Excel only (as provided)
    df = pd.read_excel(path)

    user = pick_col(df, ["user_id", "user", "uid"])
    t = pick_col(df, ["t", "round", "trial", "trial_index"], required=False)
    ts = pick_col(df, ["timestamp", "ts", "time"], required=False)
    qid = pick_col(df, ["question_id", "qid", "content_id", "item_id"])
    tags = pick_col(df, ["tags", "tag"], required=False)
    part = pick_col(df, ["part"], required=False)
    elapsed = pick_col(df, ["elapsed_time", "elapsed", "duration"], required=False)
    reward = pick_col(df, ["reward"], required=False)

    # If reward not present, try to compute from correct/user_answer
    if reward is None:
        ua = pick_col(df, ["user_answer", "answer", "response"], required=True)
        ca = pick_col(df, ["correct_answer", "answer_key", "correctanswer"], required=False)
        correct = pick_col(df, ["correct", "answered_correctly", "is_correct", "correctness"], required=False)
        if correct is not None:
            df["reward"] = pd.to_numeric(df[correct], errors="coerce").fillna(0).astype(int)
        elif ca is not None:
            df["_ua"] = pd.to_numeric(df[ua], errors="coerce")
            df["_ca"] = pd.to_numeric(df[ca], errors="coerce")
            df["reward"] = (df["_ua"] == df["_ca"]).astype(int)
            df.drop(columns=["_ua","_ca"], inplace=True, errors="ignore")
        else:
            raise KeyError("No reward/correct/correct_answer column to build reward.")
        reward = "reward"

    # If ts missing, use t or create per-user index later
    if ts is None:
        ts = t if t is not None else "ts_tmp"
        if ts == "ts_tmp":
            df[ts] = df.groupby(user, sort=False).cumcount()

    if t is None:
        df["t"] = df.groupby(user, sort=False).cumcount()
        t = "t"

    # Basic cleanup types
    df[user] = df[user].astype(str)
    df[qid] = df[qid].astype(str)
    df[reward] = pd.to_numeric(df[reward], errors="coerce").fillna(0).astype(int)

    if elapsed is not None:
        df[elapsed] = pd.to_numeric(df[elapsed], errors="coerce")

    # Sort
    df = df.sort_values([user, ts]).reset_index(drop=True)

    return df, Cols(user=user, t=t, ts=ts, qid=qid, tags=tags, part=part, elapsed=elapsed, reward=reward)

# -----------------------------
# Feature engineering (strategy signals)
# -----------------------------
def add_interaction_signals(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    out = df.copy()

    # elapsed seconds, log
    if cols.elapsed is not None:
        # Heuristic: if values look like ms (typically >= 1000), convert to seconds
        med = float(pd.to_numeric(out[cols.elapsed], errors="coerce").median())
        if med >= 1000:
            out["elapsed_s"] = pd.to_numeric(out[cols.elapsed], errors="coerce") / 1000.0
        else:
            out["elapsed_s"] = pd.to_numeric(out[cols.elapsed], errors="coerce")
        out["elapsed_s"] = clip_series(out["elapsed_s"], 0.01, 0.99)
        out["log_elapsed"] = np.log1p(out["elapsed_s"].clip(lower=0))
    else:
        out["elapsed_s"] = np.nan
        out["log_elapsed"] = 0.0

    # tag parsing
    if cols.tags is not None:
        # tags are like "5;2;182" -> first token
        out["first_tag"] = out[cols.tags].astype(str).str.split(";").str[0]
    else:
        out["first_tag"] = "NA"

    # tag switch (within user)
    out["tag_switch"] = (
        out.groupby(cols.user, sort=False)["first_tag"]
        .transform(lambda s: (s != s.shift(1)).astype(int))
        .fillna(0)
        .astype(int)
    )

    # part switch
    if cols.part is not None:
        out["part_switch"] = (
            out.groupby(cols.user, sort=False)[cols.part]
            .transform(lambda s: (s != s.shift(1)).astype(int))
            .fillna(0)
            .astype(int)
        )
    else:
        out["part_switch"] = 0

    return out

def build_user_features(df: pd.DataFrame, cols: Cols) -> pd.DataFrame:
    feats = []
    for uid, g in df.groupby(cols.user, sort=False):
        r = g[cols.reward].to_numpy()
        n = len(r)
        acc = float(np.mean(r)) if n else np.nan

        # elapsed-based
        el = g["elapsed_s"].to_numpy()
        el = el[np.isfinite(el)]
        el_med = float(np.median(el)) if len(el) else np.nan
        el_mean = float(np.mean(el)) if len(el) else np.nan
        el_std = float(np.std(el)) if len(el) else np.nan
        fast_rate = float(np.mean(el <= 3.0)) if len(el) else np.nan  # guessing-ish

        # learning / adaptation
        k = min(50, n)
        early = float(np.mean(r[:k])) if n else np.nan
        late = float(np.mean(r[-k:])) if n else np.nan
        improve = (late - early) if np.isfinite(early) and np.isfinite(late) else np.nan

        max_wrong = longest_run(r, 0)
        max_corr = longest_run(r, 1)

        tags = g["first_tag"].astype(str).to_numpy()
        if len(tags) >= 2:
            tag_switch_rate = float(np.mean(tags[1:] != tags[:-1]))
        else:
            tag_switch_rate = np.nan
        if len(tags):
            vc = pd.Series(tags).value_counts(normalize=True)
            top_tag_ratio = float(vc.iloc[0]) if len(vc) else np.nan
        else:
            top_tag_ratio = np.nan

        # persistence: streakiness + volatility
        recent3 = g["reward_rolling3"].to_numpy()
        vol3 = float(np.nanstd(recent3)) if np.isfinite(recent3).any() else np.nan

        feats.append({
            "user_id": uid,
            "n": n,
            "accuracy": acc,
            "elapsed_median": el_med,
            "elapsed_mean": el_mean,
            "elapsed_std": el_std,
            "fast_rate_le_3s": fast_rate,
            "early_acc": early,
            "late_acc": late,
            "improve": improve,
            "max_wrong_run": max_wrong,
            "max_correct_run": max_corr,
            "tag_switch_rate": tag_switch_rate,
            "top_tag_ratio": top_tag_ratio,
            "recent_acc_volatility": vol3,
        })

    feat = pd.DataFrame(feats).replace([np.inf, -np.inf], np.nan)
    return feat

# -----------------------------
# Modeling: clustering + baselines
# -----------------------------
def standardize(X: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    med = X.median(numeric_only=True)
    X2 = X.fillna(med).replace([np.inf, -np.inf], np.nan).fillna(med)
    return X2.to_numpy(dtype=float), med

def run_clustering(feat: pd.DataFrame, out_tables: Path, out_figs: Path, seed: int = 42) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.metrics import adjusted_rand_score

    use_cols = [
        "n","accuracy","elapsed_median","fast_rate_le_3s","improve",
        "max_wrong_run","tag_switch_rate","top_tag_ratio","recent_acc_volatility"
    ]
    X = feat[use_cols].copy()
    X_np, _ = standardize(X)
    Xs = StandardScaler().fit_transform(X_np)

    # choose k by quick sweep
    ks = [3,4,5,6]
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        lab = km.fit_predict(Xs)
        sil = silhouette_score(Xs, lab)
        dbi = davies_bouldin_score(Xs, lab)
        rows.append({"k": k, "model": "kmeans", "silhouette": sil, "dbi": dbi})

        gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full")
        glab = gmm.fit_predict(Xs)
        # silhouette requires >=2 labels
        sil2 = silhouette_score(Xs, glab) if len(set(glab)) > 1 else np.nan
        dbi2 = davies_bouldin_score(Xs, glab) if len(set(glab)) > 1 else np.nan
        rows.append({"k": k, "model": "gmm", "silhouette": sil2, "dbi": dbi2})

    score_df = pd.DataFrame(rows).sort_values(["model","k"])
    score_df.to_csv(out_tables / "cluster_model_selection.csv", index=False)

    # pick k=4 default (interpretable)
    k = 4

    km = KMeans(n_clusters=k, random_state=seed, n_init=50)
    feat["cluster_kmeans"] = km.fit_predict(Xs)

    gmm = GaussianMixture(n_components=k, random_state=seed, covariance_type="full")
    feat["cluster_gmm"] = gmm.fit_predict(Xs)

    # stability (kmeans only) across seeds
    labs = []
    for s in [seed, seed+1, seed+2]:
        labs.append(KMeans(n_clusters=k, random_state=s, n_init=20).fit_predict(Xs))
    ari01 = adjusted_rand_score(labs[0], labs[1])
    ari02 = adjusted_rand_score(labs[0], labs[2])
    stab = pd.DataFrame([{"k": k, "ARI(seed,seed+1)": ari01, "ARI(seed,seed+2)": ari02}])
    stab.to_csv(out_tables / "cluster_stability_kmeans.csv", index=False)

    # cluster means for interpretation
    use_cols2 = use_cols
    means_km = feat.groupby("cluster_kmeans")[use_cols2].mean()
    means_gmm = feat.groupby("cluster_gmm")[use_cols2].mean()
    means_km.to_csv(out_tables / "cluster_means_kmeans.csv")
    means_gmm.to_csv(out_tables / "cluster_means_gmm.csv")

    # Visualization: exploration vs concentration
    plt.figure()
    for c in sorted(feat["cluster_kmeans"].unique()):
        sub = feat[feat["cluster_kmeans"] == c]
        plt.scatter(sub["tag_switch_rate"], sub["top_tag_ratio"], alpha=0.7, label=f"kmeans {c}")
    plt.title("Exploration vs Concentration (KMeans clusters)")
    plt.xlabel("tag_switch_rate (higher=more exploration)")
    plt.ylabel("top_tag_ratio (higher=more repetition)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figs / "cluster_scatter_explore_vs_repeat.png", dpi=200)
    plt.close()

    # Visualization: guessing vs accuracy
    plt.figure()
    for c in sorted(feat["cluster_kmeans"].unique()):
        sub = feat[feat["cluster_kmeans"] == c]
        plt.scatter(sub["fast_rate_le_3s"], sub["accuracy"], alpha=0.7, label=f"kmeans {c}")
    plt.title("Guessing signal vs Accuracy (KMeans clusters)")
    plt.xlabel("fast_rate (elapsed<=3s)")
    plt.ylabel("accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_figs / "cluster_scatter_fast_vs_accuracy.png", dpi=200)
    plt.close()

    # Baseline score: GuessingScore
    from sklearn.preprocessing import StandardScaler
    Z = StandardScaler().fit_transform(feat[["fast_rate_le_3s","accuracy"]].fillna(feat[["fast_rate_le_3s","accuracy"]].median()))
    feat["guessing_score"] = Z[:,0] - Z[:,1]  # high=fast & low accuracy

    return feat

# -----------------------------
# Transition modeling
# -----------------------------
def hmm_transition(df_int: pd.DataFrame, cols: Cols, out_tables: Path, out_figs: Path, n_states: int = 4, seed: int = 42):
    """
    HMM on interaction-level signals:
      obs = [reward, log_elapsed, tag_switch]
    """
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception as e:
        return False, f"hmmlearn not available: {e}"

    # Build sequences
    obs_cols = ["reward", "log_elapsed", "tag_switch"]
    X_list = []
    lengths = []
    for _, g in df_int.groupby(cols.user, sort=False):
        Xg = g[obs_cols].to_numpy(dtype=float)
        if len(Xg) < 5:
            continue
        X_list.append(Xg)
        lengths.append(len(Xg))
    if not X_list:
        return False, "Not enough sequences for HMM (need len>=5)."

    X = np.vstack(X_list)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=200,
        random_state=seed,
        verbose=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, lengths)

    # Transition matrix
    trans = model.transmat_
    trans_df = pd.DataFrame(trans, columns=[f"S{j}" for j in range(n_states)], index=[f"S{i}" for i in range(n_states)])
    trans_df.to_csv(out_tables / "hmm_transition_matrix.csv")

    # Emission summary: decode states then aggregate
    # decode per interaction
    states = model.predict(X, lengths)
    Xdf = pd.DataFrame(X, columns=obs_cols)
    Xdf["state"] = states
    emis = Xdf.groupby("state").agg(
        reward_mean=("reward","mean"),
        log_elapsed_mean=("log_elapsed","mean"),
        tag_switch_rate=("tag_switch","mean"),
        n=("reward","size")
    ).sort_index()
    emis.to_csv(out_tables / "hmm_emission_summary.csv")

    # Plot transition heatmap (matplotlib imshow)
    plt.figure()
    plt.imshow(trans, aspect="auto")
    plt.colorbar(label="P(next_state)")
    plt.title("HMM transition matrix")
    plt.xlabel("to state")
    plt.ylabel("from state")
    plt.xticks(range(n_states), [f"S{i}" for i in range(n_states)])
    plt.yticks(range(n_states), [f"S{i}" for i in range(n_states)])
    plt.tight_layout()
    plt.savefig(out_figs / "hmm_transition_heatmap.png", dpi=200)
    plt.close()

    # Plot emissions
    plt.figure()
    plt.bar(emis.index.astype(int), emis["reward_mean"])
    plt.title("HMM state: mean reward")
    plt.xlabel("state")
    plt.ylabel("mean reward")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_figs / "hmm_state_mean_reward.png", dpi=200)
    plt.close()

    plt.figure()
    plt.bar(emis.index.astype(int), emis["tag_switch_rate"])
    plt.title("HMM state: tag switch rate")
    plt.xlabel("state")
    plt.ylabel("tag_switch_rate")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_figs / "hmm_state_tag_switch_rate.png", dpi=200)
    plt.close()

    return True, "ok"

def markov_from_labels(df_int: pd.DataFrame, cols: Cols, label_col: str, out_tables: Path, out_figs: Path):
    """Fallback transition model: compute Markov transitions on a discrete label sequence within each user."""
    labels = df_int[label_col].astype(int)
    n_states = int(labels.max()) + 1

    trans = np.zeros((n_states, n_states), dtype=float)
    for _, g in df_int.groupby(cols.user, sort=False):
        s = g[label_col].astype(int).to_numpy()
        if len(s) < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            trans[a, b] += 1

    # normalize rows
    row_sums = trans.sum(axis=1, keepdims=True)
    trans = np.divide(trans, row_sums, out=np.zeros_like(trans), where=row_sums != 0)

    trans_df = pd.DataFrame(trans, columns=[f"S{j}" for j in range(n_states)], index=[f"S{i}" for i in range(n_states)])
    trans_df.to_csv(out_tables / f"markov_transition_{label_col}.csv")

    plt.figure()
    plt.imshow(trans, aspect="auto")
    plt.colorbar(label="P(next)")
    plt.title(f"Markov transition on {label_col}")
    plt.xlabel("to state")
    plt.ylabel("from state")
    plt.xticks(range(n_states), [f"S{i}" for i in range(n_states)])
    plt.yticks(range(n_states), [f"S{i}" for i in range(n_states)])
    plt.tight_layout()
    plt.savefig(out_figs / f"markov_transition_{label_col}.png", dpi=200)
    plt.close()

# -----------------------------
# Visualizations
# -----------------------------
def make_figures(df_int: pd.DataFrame, feat: pd.DataFrame, cols: Cols, out_figs: Path, out_tables: Path):
    # 1) reward distribution
    plt.figure()
    df_int[cols.reward].value_counts().sort_index().plot(kind="bar")
    plt.title("Reward (correct=1) distribution")
    plt.xlabel("reward")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_figs / "reward_distribution.png", dpi=200)
    plt.close()

    # 2) interactions per user
    cnt = df_int.groupby(cols.user).size()
    plt.figure()
    cnt.clip(upper=cnt.quantile(0.99)).hist(bins=40)
    plt.title("Interactions per user (clipped at 99%)")
    plt.xlabel("# interactions")
    plt.ylabel("count(users)")
    plt.tight_layout()
    plt.savefig(out_figs / "interactions_per_user_hist.png", dpi=200)
    plt.close()

    # 3) round vs mean reward (overall)
    df = df_int.copy()
    df["round"] = df.groupby(cols.user, sort=False).cumcount() + 1
    by_round = df.groupby("round")[cols.reward].mean().head(200)
    plt.figure()
    plt.plot(by_round.index, by_round.values, marker="o")
    plt.title("Mean reward by round (first 200)")
    plt.xlabel("round")
    plt.ylabel("mean reward")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_figs / "round_mean_reward_overall.png", dpi=200)
    plt.close()

    # 4) strategy-wise learning curves (using kmeans cluster)
    if "cluster_kmeans" in feat.columns:
        df = df.merge(feat[["user_id","cluster_kmeans"]].rename(columns={"user_id": cols.user}), on=cols.user, how="left")
        curves = (
            df.groupby(["cluster_kmeans","round"])[cols.reward]
            .mean()
            .reset_index()
        )
        plt.figure()
        for c in sorted(curves["cluster_kmeans"].dropna().unique()):
            sub = curves[curves["cluster_kmeans"] == c].sort_values("round").head(200)
            plt.plot(sub["round"], sub[cols.reward], label=f"cluster {int(c)}")
        plt.title("Learning curves by strategy cluster (KMeans)")
        plt.xlabel("round")
        plt.ylabel("mean reward")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "learning_curves_by_cluster.png", dpi=200)
        plt.close()

    # 5) feature profile bar chart (cluster means)
    if "cluster_kmeans" in feat.columns:
        use_cols = [
            "accuracy","fast_rate_le_3s","tag_switch_rate","top_tag_ratio","improve","elapsed_median"
        ]
        means = feat.groupby("cluster_kmeans")[use_cols].mean()
        means.to_csv(out_tables / "cluster_profile_selected_features.csv")
        means.plot(kind="bar")
        plt.title("Cluster feature profile (selected)")
        plt.xlabel("cluster")
        plt.ylabel("mean value")
        plt.tight_layout()
        plt.savefig(out_figs / "cluster_feature_profile_selected.png", dpi=200)
        plt.close()

    # 6) elapsed distribution by cluster
    if "cluster_kmeans" in feat.columns:
        df = df_int.merge(feat[["user_id","cluster_kmeans"]].rename(columns={"user_id": cols.user}), on=cols.user, how="left")
        plt.figure()
        for c in sorted(df["cluster_kmeans"].dropna().unique()):
            sub = df[df["cluster_kmeans"] == c]["elapsed_s"].dropna()
            sub = sub.clip(upper=sub.quantile(0.99))
            plt.hist(sub, bins=40, alpha=0.5, label=f"cluster {int(c)}")
        plt.title("Elapsed time (s) distribution by cluster (clipped)")
        plt.xlabel("elapsed_s")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_figs / "elapsed_by_cluster_hist.png", dpi=200)
        plt.close()

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="ednet_step0_prepared_sample500.csv.xlsx", help="Path to xlsx")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hmm_states", type=int, default=4)
    args = parser.parse_args()

    np.random.seed(args.seed)

    out_root = Path(args.out)
    out_figs = out_root / "figs"
    out_tables = out_root / "tables"
    out_models = out_root / "models"
    ensure_dir(out_figs); ensure_dir(out_tables); ensure_dir(out_models)

    df, cols = load_data(Path(args.data))

    # Rolling recent accuracy (window 3) for volatility feature
    df["reward_rolling3"] = (
        df.groupby(cols.user, sort=False)[cols.reward]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    df = add_interaction_signals(df, cols)

    # Quick dataset summary
    print("=== DATA SUMMARY ===")
    print("rows:", len(df), "users:", df[cols.user].nunique())
    print("reward rate:", float(df[cols.reward].mean()))
    if cols.elapsed is not None:
        print("elapsed median(raw):", float(pd.to_numeric(df[cols.elapsed], errors="coerce").median()))
        print("elapsed median(s):", float(pd.to_numeric(df["elapsed_s"], errors="coerce").median()))
    print("====================")

    # User-level features + clustering
    feat = build_user_features(df, cols)
    feat = run_clustering(feat, out_tables, out_figs, seed=args.seed)

    feat.to_csv(out_tables / "user_features_with_clusters.csv", index=False)

    # Attach a per-interaction label sequence for fallback Markov transitions
    df_int = df.copy()
    df_int = df_int.rename(columns={cols.user: "user_id"})
    cols2 = Cols(user="user_id", t=cols.t, ts=cols.ts, qid=cols.qid, tags=cols.tags, part=cols.part, elapsed=cols.elapsed, reward=cols.reward)

    df_int = df_int.merge(feat[["user_id","cluster_kmeans","cluster_gmm"]], on="user_id", how="left")

    # Transition modeling
    ok, msg = hmm_transition(df_int, cols2, out_tables, out_figs, n_states=args.hmm_states, seed=args.seed)
    if ok:
        print("[HMM] success")
    else:
        print(f"[HMM] skipped -> {msg}")
        # Fallback: Markov transitions on discrete cluster labels
        markov_from_labels(df_int, cols2, "cluster_kmeans", out_tables, out_figs)
        markov_from_labels(df_int, cols2, "cluster_gmm", out_tables, out_figs)

    # Figures
    make_figures(df_int, feat, cols2, out_figs, out_tables)

    print("\nDone. Outputs written to:", out_root.resolve())

if __name__ == "__main__":
    main()
