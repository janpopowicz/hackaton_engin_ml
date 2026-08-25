"""Rodziny klasy ``unknown`` — dodatkowa sugestia dla mechanika.

Etykieta diagnostyczna zostaje ``unknown`` (nie wchodzi do submitu).
Tu tylko grupujemy kształt residuum, żeby UI mogło powiedzieć
„to wygląda na rodzinę U2 — wznos 14–20 kHz”.

Przestrzeń: residuum po wyjęciu jittera wzmocnienia (``Pool.gain_removed``),
znormalizowane do wektora jednostkowego — klasteryzujemy kształt, nie
amplitudę ani profil silnika.

Trening: złote unknown z ``val_full.csv`` + cylindry z ``train.csv``
oznaczone TabPFN-em jako unknown (to ta sama klasa co hybryda; nasilenie
nie dotyczy). Słabe residua z traina (szum / nadrozpoznanie) odpadają.

Inferencja: czysty numpy, cosine do zapisanych centrów. Bez sklearn.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from physics_diagnose import FAULTS, FREQ_COLS, build_pool

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
DEFAULT_PATH = ARTIFACTS / "unknown_clusters.pkl"
JSON_PATH = ARTIFACTS / "unknown_clusters.json"
VIZ = ARTIFACTS / "visualizations"

KHZ = np.arange(21)
L2_MIN_TRAIN = 25.0
COSINE_MIN = 0.55
N_CLUSTERS = 3

# Identyfikatory stabilne względem permutacji k-means: dominujące pasmo średniej.
FAMILY_SPEC = {
    "mf": {
        "id": "U1",
        "name": "garb 7–11 kHz",
        "hint": (
            "Dodatnia energia w paśmie środkowym (7–11 kHz) i dołek ok. 5 kHz. "
            "Nie pokrywa się z katalogiem zakoksowany / iglica / pompa / lejący."
        ),
        "highlight_khz": (7, 8, 9, 10, 11),
        "color": "#0f766e",
    },
    "hf": {
        "id": "U2",
        "name": "wznos 14–20 kHz",
        "hint": (
            "Dołek 5–10 kHz i rosnąca energia 14–20 kHz. Trochę przypomina koks, "
            "ale bez typowego odbicia 9→12 kHz."
        ),
        "highlight_khz": (5, 16, 18, 19, 20),
        "color": "#9f1239",
    },
    "lf": {
        "id": "U3",
        "name": "szczyt 1–4 kHz",
        "hint": (
            "Silny garb 1–4 kHz i dołek 8–13 kHz; czasem drugie odbicie ~16 kHz. "
            "Częściowo podobne do iglicy, ale z innym ogonem HF."
        ),
        "highlight_khz": (1, 2, 3, 10, 16),
        "color": "#3730a3",
    },
}

FAMILY_COLORS = {spec["id"]: spec["color"] for spec in FAMILY_SPEC.values()}


def _unit(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True).clip(1e-9)
    return X / n


def _dominant_band(mean_res: np.ndarray) -> str:
    bands = {
        "lf": float(mean_res[:5].mean()),
        "mf": float(mean_res[6:12].mean()),
        "hf": float(mean_res[14:].mean()),
    }
    return max(bands, key=bands.get)


def _family_from_mean(mean_res: np.ndarray) -> dict:
    band = _dominant_band(mean_res)
    spec = FAMILY_SPEC[band]
    peak = int(np.argmax(np.abs(mean_res)))
    return {
        **spec,
        "band": band,
        "peak_khz": peak,
        "peak_mV": round(float(mean_res[peak]), 2),
    }


# --------------------------------------------------------------------------
# zbiór treningowy
# --------------------------------------------------------------------------
def collect_unknown_rows(
    *,
    val_path: Path | None = None,
    train_path: Path | None = None,
    train_labels_path: Path | None = None,
    l2_min_train: float = L2_MIN_TRAIN,
) -> dict[str, np.ndarray]:
    val = pd.read_csv(val_path or BASE / "val_full.csv")
    train = pd.read_csv(train_path or BASE / "train.csv")
    labels = pd.read_csv(train_labels_path or BASE / "predictions_tabpfn_train.csv")
    train = train.merge(labels, on=["engine_id", "cylinder"], how="left")
    if train["label"].isna().any():
        missing = int(train["label"].isna().sum())
        raise ValueError(f"brak etykiet TabPFN dla {missing} wierszy train.csv")

    def take(df: pd.DataFrame, src: str, l2_min: float) -> dict[str, np.ndarray]:
        pool = build_pool(df.reset_index(drop=True))
        P = pool.gain_removed()
        nrm = np.linalg.norm(P, axis=1)
        mask = (df["label"].to_numpy() == "unknown") & (nrm >= l2_min)
        return {
            "P": P[mask],
            "l2": nrm[mask],
            "engine_id": df.loc[mask, "engine_id"].to_numpy(),
            "cylinder": df.loc[mask, "cylinder"].to_numpy().astype(int),
            "n_cylinders": df.loc[mask, "n_cylinders"].to_numpy().astype(int),
            "source": np.array([src] * int(mask.sum())),
        }

    a = take(val, "val_full", 0.0)
    b = take(train, "train_tabpfn", l2_min_train)
    keys = ["P", "l2", "engine_id", "cylinder", "n_cylinders", "source"]
    out = {k: np.concatenate([a[k], b[k]], axis=0) for k in keys}
    out["n_val"] = np.array([len(a["P"])])
    out["n_train"] = np.array([len(b["P"])])
    out["n_train_dropped"] = np.array(
        [int(((train["label"] == "unknown").sum()) - len(b["P"]))]
    )
    return out


def fit(
    *,
    k: int = N_CLUSTERS,
    l2_min_train: float = L2_MIN_TRAIN,
    random_state: int = 0,
) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    rows = collect_unknown_rows(l2_min_train=l2_min_train)
    P = rows["P"].astype(float)
    Up = _unit(P)
    km = KMeans(k, n_init=50, random_state=random_state).fit(Up)

    sil_k = {}
    for kk in range(2, 8):
        lab_k = KMeans(kk, n_init=50, random_state=random_state).fit_predict(Up)
        sil_k[kk] = float(silhouette_score(Up, lab_k, metric="cosine"))

    # szablony znanych usterek (do opisu, nie do decyzji)
    val = pd.read_csv(BASE / "val_full.csv")
    Pv = build_pool(val).gain_removed()
    yv = val["label"].to_numpy()
    templates = {}
    for lab in FAULTS:
        t = Pv[yv == lab].mean(0)
        templates[lab] = t / max(float(np.linalg.norm(t)), 1e-9)

    families = []
    centers = np.zeros((k, P.shape[1]))
    mean_res = np.zeros((k, P.shape[1]))
    used_ids: set[str] = set()
    raw_labels = km.labels_.copy()
    new_labels = np.full(len(raw_labels), -1, dtype=int)
    order = []

    for raw in range(k):
        m = raw_labels == raw
        mean = P[m].mean(0)
        fam = _family_from_mean(mean)
        if fam["id"] in used_ids:
            fam["id"] = f"{fam['id']}{raw}"
        used_ids.add(fam["id"])
        c = Up[m].mean(0)
        c = c / max(float(np.linalg.norm(c)), 1e-9)
        cors = {lab: float(c @ templates[lab]) for lab in FAULTS}
        nearest = max(cors, key=cors.get)
        own = Up[m] @ c
        fam.update(
            {
                "n": int(m.sum()),
                "n_val": int(((rows["source"] == "val_full") & m).sum()),
                "n_train": int(((rows["source"] == "train_tabpfn") & m).sum()),
                "mean_cosine": round(float(own.mean()), 3),
                "min_cosine": round(float(own.min()), 3),
                "nearest_fault": nearest,
                "nearest_fault_cosine": round(cors[nearest], 3),
                "kmeans_id": int(raw),
            }
        )
        order.append((fam["id"], raw, fam, c, mean, m))

    order.sort(key=lambda t: t[0])
    for new_i, (_, raw, fam, c, mean, m) in enumerate(order):
        fam["index"] = new_i
        families.append(fam)
        centers[new_i] = c
        mean_res[new_i] = mean
        new_labels[m] = new_i

    state = {
        "format": 1,
        "k": int(k),
        "space": "gain_removed_unit",
        "cosine_min": float(COSINE_MIN),
        "l2_min_train": float(l2_min_train),
        "centers": centers,
        "mean_residual": mean_res,
        "families": families,
        "silhouette": sil_k,
        "silhouette_chosen": sil_k[k],
        "n_samples": int(len(P)),
        "n_val": int(rows["n_val"][0]),
        "n_train": int(rows["n_train"][0]),
        "n_train_dropped": int(rows["n_train_dropped"][0]),
        "assignments": {
            "engine_id": rows["engine_id"],
            "cylinder": rows["cylinder"],
            "n_cylinders": rows["n_cylinders"],
            "source": rows["source"],
            "l2": rows["l2"],
            "cluster": new_labels,
            "cosine": np.array(
                [float(Up[i] @ centers[new_labels[i]]) for i in range(len(Up))]
            ),
        },
        "templates_unit": templates,
        "P": P,
        "source": rows["source"],
        "cluster": new_labels,
    }
    return state


def save(state: dict, path: Path | None = None) -> Path:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = Path(path or DEFAULT_PATH)
    slim = {
        "format": state["format"],
        "k": state["k"],
        "space": state["space"],
        "cosine_min": state["cosine_min"],
        "l2_min_train": state["l2_min_train"],
        "centers": np.asarray(state["centers"], dtype=float),
        "mean_residual": np.asarray(state["mean_residual"], dtype=float),
        "families": state["families"],
        "silhouette": state["silhouette"],
        "silhouette_chosen": state["silhouette_chosen"],
        "n_samples": state["n_samples"],
        "n_val": state["n_val"],
        "n_train": state["n_train"],
        "n_train_dropped": state["n_train_dropped"],
    }
    path.write_bytes(pickle.dumps(slim, protocol=4))

    json_blob = {
        **{k: slim[k] for k in slim if k not in ("centers", "mean_residual")},
        "centers": slim["centers"].round(6).tolist(),
        "mean_residual": slim["mean_residual"].round(4).tolist(),
        "khz": list(range(21)),
    }
    JSON_PATH.write_text(
        json.dumps(json_blob, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ass = state.get("assignments")
    if ass is not None:
        fam_by_i = {f["index"]: f["id"] for f in state["families"]}
        pd.DataFrame(
            {
                "engine_id": ass["engine_id"],
                "cylinder": ass["cylinder"],
                "source": ass["source"],
                "cluster": [fam_by_i[int(i)] for i in ass["cluster"]],
                "cosine": np.round(ass["cosine"], 3),
                "l2": np.round(ass["l2"], 2),
            }
        ).to_csv(ARTIFACTS / "unknown_cluster_assignments.csv", index=False)
    return path


def load(path: Path | str | None = None) -> dict:
    p = Path(path or DEFAULT_PATH)
    state = pickle.loads(p.read_bytes())
    if int(state.get("format", 0)) != 1:
        raise ValueError(f"nieznany format unknown_clusters: {state.get('format')}")
    state["centers"] = np.asarray(state["centers"], dtype=float)
    state["mean_residual"] = np.asarray(state["mean_residual"], dtype=float)
    return state


def assign_gain_removed(
    P: np.ndarray, state: dict | None = None
) -> list[dict]:
    """Przypisz wiersze residuum (gain-removed) do najbliższej rodziny.

    Zwraca listę dictów; ``id`` jest ``None``, gdy cosine < próg
    (kształt poza trzema rodzinami — nadal ``unknown``).
    """
    if state is None:
        state = load()
    P = np.asarray(P, dtype=float)
    if P.ndim == 1:
        P = P[None, :]
    U = _unit(P)
    C = state["centers"]
    sims = U @ C.T
    best = np.argmax(sims, axis=1)
    families = state["families"]
    thr = float(state["cosine_min"])
    out = []
    for i, k in enumerate(best):
        row_sims = sims[i]
        order = np.argsort(row_sims)[::-1]
        cos = float(row_sims[k])
        second = float(row_sims[order[1]]) if len(order) > 1 else -1.0
        fam = families[int(k)]
        ok = cos >= thr
        out.append(
            {
                "id": fam["id"] if ok else None,
                "name": fam["name"] if ok else "poza katalogiem rodzin",
                "hint": fam["hint"] if ok else (
                    "Kształt nie pasuje do żadnej z trzech rodzin unknown — "
                    "zostawiam gołą etykietę unknown."
                ),
                "color": fam["color"] if ok else "#57534e",
                "highlight_khz": list(fam["highlight_khz"]) if ok else [],
                "index": int(k) if ok else -1,
                "cosine": round(cos, 3),
                "margin": round(cos - second, 3),
                "matched": bool(ok),
            }
        )
    return out


def assign_from_df(df: pd.DataFrame, state: dict | None = None) -> list[dict]:
    pool = build_pool(df.reset_index(drop=True))
    return assign_gain_removed(pool.gain_removed(), state)


def suggestion_line(hit: dict) -> str:
    if hit["matched"]:
        return (
            f"Rodzina kształtu (sugestia, nie zmienia etykiety): "
            f"{hit['id']} — {hit['name']}  "
            f"(cosine {hit['cosine']:.2f}, margines {hit['margin']:.2f})."
        )
    return (
        f"Rodzina kształtu: brak dopasowania do U1/U2/U3 "
        f"(najbliższe cosine {hit['cosine']:.2f})."
    )


# --------------------------------------------------------------------------
# wykresy
# --------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)
    ax.set_xticks(KHZ[::2])


def plot_all(state: dict, out_dir: Path | None = None) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir or VIZ)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    families = state["families"]
    P = state["P"]
    cl = state["cluster"]
    src = state["source"]
    Cmean = state["mean_residual"]
    templates = state["templates_unit"]

    # 1. wybór k
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ks = sorted(state["silhouette"])
    ys = [state["silhouette"][k] for k in ks]
    ax.plot(ks, ys, color="#444", marker="o", lw=1.8)
    ax.scatter(
        [state["k"]],
        [state["silhouette_chosen"]],
        s=140,
        zorder=5,
        color="#9f1239",
        label=f"wybrane k={state['k']}",
    )
    ax.set_xlabel("liczba klastrów k")
    ax.set_ylabel("silhouette (cosine)")
    ax.set_title("Dobór k dla rodzin unknown (residuum gain-removed, unit)")
    ax.set_xticks(ks)
    ax.set_xlim(1.5, 7.5)
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.95)
    ax.text(
        0.02, 0.04,
        "k=6 ma wyższy silhouette, ale tnie te same 3 kształty.\n"
        "k=3 to pierwsze plateau i trzy rozłączne rodziny.",
        transform=ax.transAxes, fontsize=8, color="#444", va="bottom",
    )
    fig.tight_layout()
    p = out_dir / "unknown_klastry_silhouette.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    # 2. centra vs znane usterki
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharex=True, sharey=True)
    fault_ls = {
        "zakoksowany": ("--", "#e67e22"),
        "lejacy": (":", "#2980b9"),
        "pompa": ("-.", "#8e44ad"),
        "iglica": ((0, (4, 2)), "#27ae60"),
    }
    for ax, fam, mean in zip(axes, families, Cmean):
        ax.axhline(0, color="#bbb", lw=1)
        for lab, (ls, col) in fault_ls.items():
            t = templates[lab]
            # skala szablonu do RMS centrum, żeby porównać kształt
            scale = float(np.linalg.norm(mean))
            ax.plot(KHZ, t * scale, ls=ls, color=col, lw=1.15, alpha=0.85, label=lab)
        ax.plot(KHZ, mean, color=fam["color"], lw=2.6, label=fam["id"])
        ax.set_title(
            f"{fam['id']}  ·  {fam['name']}\n"
            f"n={fam['n']}  (val {fam['n_val']}, train {fam['n_train']})",
            fontsize=10,
        )
        _style(ax)
        ax.set_xlabel("częstotliwość [kHz]")
    axes[0].set_ylabel("średnie residuum [mV]")
    axes[0].legend(fontsize=7, loc="upper right", ncol=2, framealpha=0.92)
    fig.suptitle(
        "Centra rodzin unknown vs. szablony znanych usterek (to samo residuum)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    p = out_dir / "unknown_klastry_centra.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)

    # 3. członkowie
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    for ax, fam, idx in zip(axes, families, range(len(families))):
        m = cl == idx
        ax.axhline(0, color="#bbb", lw=1)
        val_m = m & (src == "val_full")
        tr_m = m & (src == "train_tabpfn")
        first_tr = True
        for row in P[tr_m]:
            ax.plot(
                KHZ, row, color=fam["color"], lw=0.7, alpha=0.22,
                label="train (TabPFN)" if first_tr else None,
            )
            first_tr = False
        first_val = True
        for row in P[val_m]:
            ax.plot(
                KHZ, row, color="#111", lw=1.5, alpha=0.9,
                label="val_full (złote)" if first_val else None,
            )
            first_val = False
        ax.plot(KHZ, Cmean[idx], color=fam["color"], lw=2.8, label="centrum")
        ax.set_title(f"{fam['id']}  ·  {fam['name']}", fontsize=11)
        _style(ax)
        ax.set_xlabel("częstotliwość [kHz]")
        ax.legend(fontsize=7, loc="upper right", framealpha=0.92)
    axes[0].set_ylabel("residuum gain-removed [mV]")
    fig.suptitle("Członkowie rodzin unknown — train TabPFN + złote z val_full", fontsize=12)
    fig.tight_layout()
    p = out_dir / "unknown_klastry_czlonkowie.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    # 4. PCA
    from sklearn.decomposition import PCA

    U = _unit(P)
    xy = PCA(2, random_state=0).fit_transform(U)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    for idx, fam in enumerate(families):
        m = cl == idx
        val_m = m & (src == "val_full")
        tr_m = m & (src == "train_tabpfn")
        ax.scatter(
            xy[tr_m, 0], xy[tr_m, 1],
            s=36, alpha=0.75, c=fam["color"],
            label=f"{fam['id']} train n={int(tr_m.sum())}",
            edgecolors="none",
        )
        ax.scatter(
            xy[val_m, 0], xy[val_m, 1],
            s=90, marker="D", c=fam["color"],
            edgecolors="#111", linewidths=0.8,
            label=f"{fam['id']} val n={int(val_m.sum())}",
            zorder=5,
        )
    ax.set_xlabel("PC1 (kształt residuum)")
    ax.set_ylabel("PC2")
    ax.set_title("Unknown — PCA jednostkowych residuów, k=3")
    ax.legend(fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out_dir / "unknown_klastry_pca.png"
    fig.savefig(p, dpi=160)
    plt.close(fig)
    saved.append(p)

    return saved


def main() -> None:
    state = fit()
    path = save(state)
    plots = plot_all(state)
    print(f"zapisano {path}")
    print(f"json     {JSON_PATH}")
    print(
        f"k={state['k']}  silhouette={state['silhouette_chosen']:.3f}  "
        f"N={state['n_samples']} (val {state['n_val']} + train {state['n_train']}, "
        f"odrzucono {state['n_train_dropped']} słabych z traina)"
    )
    for fam in state["families"]:
        print(
            f"  {fam['id']:4s} {fam['name']:18s}  n={fam['n']:3d}  "
            f"val={fam['n_val']}  peak={fam['peak_khz']} kHz  "
            f"cosine={fam['mean_cosine']:.2f}  "
            f"najbliższa usterka {fam['nearest_fault']} "
            f"({fam['nearest_fault_cosine']:+.2f})"
        )
    print("silhouette vs k:", {k: round(v, 3) for k, v in state["silhouette"].items()})
    for p in plots:
        print(f"  wykres {p.name}")


if __name__ == "__main__":
    main()
