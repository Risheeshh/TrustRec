"""TrustRec core: data loading, hybrid recommender, shilling attacks, detector, metrics."""
import io
import os
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]
ML_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ATTACKS = ["random", "average", "bandwagon"]


# ----------------------------------------------------------------------------- data
def _synthetic(seed=0, n_users=943, n_items=1682):
    """MovieLens-like fallback data (used only if the real dataset can't be downloaded)."""
    rng = np.random.default_rng(seed)
    G = np.zeros((n_items, len(GENRES)))
    for i in range(n_items):
        G[i, rng.choice(np.arange(1, len(GENRES)), rng.integers(1, 4), replace=False)] = 1
    pref = rng.normal(0, 0.7, (n_users, len(GENRES)))
    pop = 1 / np.arange(1, n_items + 1) ** 0.8
    pop = rng.permutation(pop / pop.sum())
    R = np.zeros((n_users, n_items))
    for u in range(n_users):
        n = int(np.clip(rng.lognormal(4, 0.8), 20, 600))
        items = rng.choice(n_items, n, replace=False, p=pop)
        R[u, items] = np.clip(np.round(3.3 + G[items] @ pref[u] + rng.normal(0, 0.8, n)), 1, 5)
    titles = np.array([f"Synthetic Movie {i + 1}" for i in range(n_items)])
    return R, titles, G


def load_data(data_dir="data"):
    """Returns (R, titles, G, is_real). Downloads MovieLens 100K on first run."""
    folder = os.path.join(data_dir, "ml-100k")
    if not os.path.exists(os.path.join(folder, "u.data")):
        try:
            raw = urllib.request.urlopen(ML_URL, timeout=20).read()
            zipfile.ZipFile(io.BytesIO(raw)).extractall(data_dir)
        except Exception:
            R, titles, G = _synthetic()
            return R, titles, G, False
    ratings = pd.read_csv(os.path.join(folder, "u.data"), sep="\t",
                          names=["u", "i", "r", "t"])
    items = pd.read_csv(os.path.join(folder, "u.item"), sep="|", header=None, encoding="latin-1")
    R = np.zeros((943, 1682))
    R[ratings.u - 1, ratings.i - 1] = ratings.r
    return R, items[1].values, items.iloc[:, 5:24].values.astype(float), True


def split(R, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    test = (R > 0) & (rng.random(R.shape) < frac)
    return R * ~test, R * test


# ------------------------------------------------------------------ hybrid recommender
def fit(R, k=50):
    """Item-based CF model: shrunk item means + top-k positive item similarities."""
    M = R > 0
    cnt = M.sum(0)
    gm = R.sum() / max(M.sum(), 1)
    mean = (R.sum(0) + 5 * gm) / (cnt + 5)  # shrink
    C = np.where(M, R - mean, 0.0)
    norm = np.linalg.norm(C, axis=0) + 1e-9
    S = (C.T @ C) / np.outer(norm, norm)
    co = M.T.astype(float) @ M
    S = S * co / (co + 10)  # penalise coincindence similarities
    np.fill_diagonal(S, 0)
    S = np.clip(S, 0, None)
    thr = -np.partition(-S, k, axis=0)[k]
    S = np.where(S > thr[None, :], S, 0.0)
    return {"S": S, "mean": mean}


def score(Ru, model, G, alpha=0.7):
    """Hybrid score in [0,1] = alpha * item-CF + (1-alpha) * content-based."""
    M = Ru > 0
    C = np.where(M, Ru - model["mean"], 0.0)
    S = model["S"]
    cf = np.clip(model["mean"] + (C @ S) / (M.astype(float) @ S + 1.0), 1, 5)
    cf_n = (cf - 1) / 4
    P = np.where(M, Ru - 3.0, 0.0) @ G
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    ct = (Pn @ Gn.T + 1) / 2
    return alpha * cf_n + (1 - alpha) * ct, Pn


def explain_recommendations(Ru, model, G, top, alpha=0.7):
    """Return the CF/content evidence behind one user's recommendations."""
    Ru = np.asarray(Ru)
    if Ru.ndim != 1:
        raise ValueError("Ru must contain ratings for exactly one user")

    M = Ru > 0
    C = np.where(M, Ru - model["mean"], 0.0)
    S = model["S"]
    cf = np.clip(model["mean"] + (C @ S) / (M.astype(float) @ S + 1.0), 1, 5)
    cf_n = (cf - 1) / 4

    P = np.where(M, Ru - 3.0, 0.0) @ G
    Pn = P / (np.linalg.norm(P) + 1e-9)
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
    ct = (Pn @ Gn.T + 1) / 2

    evidence = []
    for item in top:
        item = int(item)
        contributions = C * S[:, item]
        positive = np.maximum(contributions, 0)
        anchor = int(np.argmax(positive)) if positive.any() else -1
        evidence.append({
            "item": item,
            "score": float(alpha * cf_n[item] + (1 - alpha) * ct[item]),
            "cf_component": float(alpha * cf_n[item]),
            "content_component": float((1 - alpha) * ct[item]),
            "anchor": anchor,
            "anchor_rating": float(Ru[anchor]) if anchor >= 0 else 0.0,
            "genre": int(np.argmax(Pn * G[item])) if G[item].any() else -1,
        })
    return evidence


def top_n(scores, Rtr, n=10):
    s = np.where(Rtr > 0, -np.inf, scores)
    return np.argsort(-s, axis=1)[:, :n]


def evaluate(scores, Rtr, Rte, target, k=10):
    s = np.where(Rtr > 0, -np.inf, scores)
    order = np.argsort(-s, axis=1)
    rel = Rte >= 4
    hits = np.take_along_axis(rel, order[:, :k], 1).sum(1)
    has = rel.any(1)
    rank = np.argsort(order, axis=1)[:, target]
    elig = Rtr[:, target] == 0
    return {"Precision@10": float((hits[has] / k).mean()),
            "Target hit ratio@10": float((rank[elig] < k).mean()),
            "Target avg rank": float(rank[elig].mean() + 1)}


# ------------------------------------------------------------------------- attacks
def make_attack(R, kind, n_fake, target, rng, filler_frac=0.05):
    n_items = R.shape[1]
    M = R > 0
    icnt = M.sum(0)
    imean = R.sum(0) / np.maximum(icnt, 1)
    allr = R[M]
    gm, gs = allr.mean(), allr.std()
    n_fill = max(int(filler_frac * n_items), 5)
    popular = [i for i in np.argsort(-icnt)[:30] if i != target]
    cand = np.setdiff1d(np.arange(n_items), [target])
    F = np.zeros((n_fake, n_items))
    for f in range(n_fake):
        fill = rng.choice(cand, n_fill, replace=False)
        if kind == "random":
            vals = rng.normal(gm, gs, n_fill)
        elif kind == "average":
            vals = rng.normal(imean[fill], 1.0)
        else:  # bandwagon: max rating on popular items + random filler
            vals = rng.normal(gm, gs, n_fill)
            F[f, popular] = 5
        F[f, fill] = np.clip(np.round(vals), 1, 5)
        F[f, target] = 5
    return F


# ------------------------------------------------------------------------ detector
def features(R):
    M = R > 0
    cnt = M.sum(1)
    n = np.maximum(cnt, 1)
    mean = R.sum(1) / n
    std = np.sqrt(np.maximum((R ** 2).sum(1) / n - mean ** 2, 0))
    icnt = M.sum(0)
    imean = R.sum(0) / np.maximum(icnt, 1)
    dev = np.abs(np.where(M, R - imean, 0))
    rdma = (dev / np.maximum(icnt, 1)).sum(1) / n
    popfrac = (M & (icnt >= np.quantile(icnt, 0.95))).sum(1) / n
    frac5 = (R == 5).sum(1) / n
    return np.c_[cnt, mean, std, rdma, dev.sum(1) / n, popfrac, frac5]


def train_detector(R, seed=0):
    """Random Forest trained on simulated attacks (varied type, size, target)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for kind in ATTACKS:
        for _ in range(2):
            nf = int(R.shape[0] * rng.uniform(0.03, 0.10))
            F = make_attack(R, kind, nf, int(rng.integers(R.shape[1])), rng,
                            filler_frac=rng.uniform(0.02, 0.10))
            X.append(features(np.vstack([R, F])))
            y.append(np.r_[np.zeros(len(R)), np.ones(nf)])
    return RandomForestClassifier(200, class_weight="balanced", random_state=seed,
                                  n_jobs=-1).fit(np.vstack(X), np.concatenate(y))


def run_scenarios(Rtr, Rte, G, base_model, detector, kind, frac, target, alpha, seed):
    """Clean vs attacked vs defended. Returns metrics, score matrices, detection stats."""
    rng = np.random.default_rng(seed)
    n_real = len(Rtr)
    n_fake = max(int(n_real * frac), 1)
    F = make_attack(Rtr, kind, n_fake, target, rng)
    A = np.vstack([Rtr, F])

    s_clean, _ = score(Rtr, base_model, G, alpha)
    s_att, _ = score(Rtr, fit(A), G, alpha)

    flagged = detector.predict_proba(features(A))[:, 1] > 0.5
    keep = ~flagged
    s_def, _ = score(Rtr, fit(A[keep]), G, alpha)

    truth = np.r_[np.zeros(n_real, bool), np.ones(n_fake, bool)]
    tp = int((flagged & truth).sum())
    det = {"Fake profiles injected": n_fake, "Fake caught": tp,
           "Real users wrongly flagged": int((flagged & ~truth).sum()),
           "Recall": tp / n_fake, "Precision": tp / max(int(flagged.sum()), 1)}
    scores = {"Clean": s_clean, "Attacked": s_att, "Defended": s_def}
    metrics = {name: evaluate(s, Rtr, Rte, target) for name, s in scores.items()}
    return metrics, scores, det
