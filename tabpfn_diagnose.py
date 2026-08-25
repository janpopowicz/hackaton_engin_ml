"""Second solution: TabPFN teacher distilled into a decision tree.

TabPFN learns the diagnostic mapping from the labeled `val.csv`.
It then labels the unlabeled workshop archive (`train.csv`). A shallow
sklearn tree is distilled on named acoustic features so that:

* inference is CPU-only and millisecond-fast (hackathon CPU criterion),
* each verdict is an if/then path a mechanic can read (explainability).

Honest score is `final_valid.csv` (held-out engines, never used in fit).
val/final_valid get the same ~5% spectrum holes as test.csv before fit
and scoring; train.csv/test.csv already have them.
Severity stays physics-based (signature magnitude), not a second model.

Colab T4:  Runtime → GPU, then `python tabpfn_diagnose.py --cv`
CPU only:  `python tabpfn_diagnose.py`  (skip --cv if you just need the app)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
FREQ_COLS = [f"mV_{i}" for i in range(21)]
LABELS = ["ok", "zakoksowany", "lejacy", "pompa", "iglica", "unknown"]
FAULTS = ["zakoksowany", "lejacy", "pompa", "iglica"]
SEV_ORDER = ["male", "srednie", "duze"]

SIG_KEYS = ["dip9", "peak12", "dip3", "dip18", "hf", "mf", "lf", "l2", "l1", "rough"]
FEATURE_NAMES = (
    [f"res_{i}kHz" for i in range(21)]
    + SIG_KEYS
    + [f"cos_{lab}" for lab in FAULTS]
    + ["n_cylinders"]
)

FEATURE_PL = {
    "dip9": "dołek przy 9 kHz (koks)",
    "peak12": "odbicie 9 → 12 kHz",
    "dip3": "dołek przy 3 kHz (pompa)",
    "dip18": "dołek przy 18 kHz",
    "hf": "pasmo wysokie 14–20 kHz",
    "mf": "pasmo środkowe 6–11 kHz",
    "lf": "pasmo niskie 0–4 kHz",
    "l2": "odchyłka L2 od profilu silnika",
    "l1": "odchyłka L1 od profilu silnika",
    "rough": "chropowatość widma residualnego",
    "cos_zakoksowany": "podobieństwo do wzorca: zakoksowany",
    "cos_lejacy": "podobieństwo do wzorca: lejący",
    "cos_pompa": "podobieństwo do wzorca: pompa",
    "cos_iglica": "podobieństwo do wzorca: iglica",
    "n_cylinders": "liczba cylindrów jednostki",
}
for _i in range(21):
    FEATURE_PL[f"res_{_i}kHz"] = f"residual {_i} kHz vs. baseline silnika"

FEATURE_BANDS: dict[str, tuple[int, ...]] = {
    "dip9": (8, 9, 10),
    "peak12": (9, 12),
    "dip3": (2, 3, 4),
    "dip18": (17, 18, 19),
    "hf": tuple(range(14, 21)),
    "mf": tuple(range(6, 12)),
    "lf": tuple(range(0, 5)),
    "cos_zakoksowany": (9, 12),
    "cos_lejacy": tuple(range(6, 21)),
    "cos_pompa": (2, 3, 4),
    "cos_iglica": tuple(range(6, 12)),
}
for _i in range(21):
    FEATURE_BANDS[f"res_{_i}kHz"] = (_i,)

_SEV_DEFAULTS = {
    "zakoksowany": (24.5, 35.5),
    "lejacy": (80.0, 118.0),
    "pompa": (39.0, 68.0),
    "iglica": (36.0, 60.0),
}

CONF_MIN = 0.70
TREE_DEPTH = 6
OK_L2_MAX = 32.0
# val.csv / final_valid.csv are complete lab spectra. test.csv already has
# ~5% missing bins. Punch the same pattern into labeled data so CV/holdout
# match the submit distribution.
GAP_RATE = 0.05
GAP_SEED = 0
LABELED_COLS = ["engine_id", "cylinder", "n_cylinders", *FREQ_COLS, "label", "severity"]


def read_labeled_csv(path: Path) -> pd.DataFrame:
    """Load val / final_valid even if the file was saved without a header."""
    header = pd.read_csv(path, nrows=0)
    if "engine_id" in header.columns:
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=LABELED_COLS)


def _row_gap_seed(seed: int, engine_id: str, cylinder: int) -> int:
    payload = f"{int(seed)}:{engine_id}:{int(cylinder)}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def punch_spectrum_gaps(
    df: pd.DataFrame,
    *,
    rate: float = GAP_RATE,
    seed: int = GAP_SEED,
    verbose: bool = False,
) -> pd.DataFrame:
    """Drop ~rate of spectrum bins, matching workshop/test missingness.

    Independent Bernoulli per 1 kHz bin (~5%) reproduces train/test: ~65% of
    rows have a hole, runs are almost all length 1, rarely 2–3. Sets that
    already contain NaNs (train.csv, test.csv) are left unchanged.
    Masks are keyed by engine_id+cylinder so CV/LOEO/predict stay aligned.
    """
    out = df.copy()
    spec = out[FREQ_COLS].to_numpy(dtype=float, copy=True)
    if np.isnan(spec).any():
        if verbose:
            n_nan = int(np.isnan(spec).sum())
            print(
                f"  luki pomiarowe: już {n_nan}/{spec.size} "
                f"({100 * n_nan / spec.size:.2f}%) — nie dubluję"
            )
        return out
    if rate <= 0:
        return out

    n_rows, n_bins = spec.shape
    eids = out["engine_id"].astype(str).to_numpy()
    cyls = np.asarray(out["cylinder"], dtype=np.int64)
    drop = np.zeros((n_rows, n_bins), dtype=bool)
    min_keep = 2
    for i in range(n_rows):
        rng = np.random.default_rng(_row_gap_seed(seed, eids[i], int(cyls[i])))
        mask = rng.random(n_bins) < rate
        if int((~mask).sum()) < min_keep:
            mask[:] = False
            n_drop = min(n_bins - min_keep, max(1, int(round(rate * n_bins))))
            mask[rng.choice(n_bins, size=n_drop, replace=False)] = True
        drop[i] = mask
    spec[drop] = np.nan
    out[FREQ_COLS] = spec
    if verbose:
        n_nan = int(drop.sum())
        n_hit = int(drop.any(axis=1).sum())
        print(
            f"  luki pomiarowe: {n_nan}/{spec.size} ({100 * n_nan / spec.size:.2f}%)  "
            f"wiersze z luką: {n_hit}/{n_rows}"
        )
    return out


def interpolate_spectrum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[FREQ_COLS] = (
        out[FREQ_COLS].interpolate(axis=1, limit_direction="both").clip(lower=0.0)
    )
    return out


def engine_baseline(spectra: np.ndarray, keep_frac: float = 0.7) -> np.ndarray:
    med = np.median(spectra, axis=0)
    dist = np.linalg.norm(spectra - med, axis=1)
    k = max(2, int(np.ceil(len(spectra) * keep_frac)))
    keep = np.argsort(dist)[:k]
    return np.median(spectra[keep], axis=0)


def compute_residuals(spectra: np.ndarray, engine_ids: np.ndarray) -> np.ndarray:
    residual = np.zeros_like(spectra, dtype=float)
    for eid in np.unique(engine_ids):
        idx = np.where(engine_ids == eid)[0]
        residual[idx] = spectra[idx] - engine_baseline(spectra[idx])
    return residual


def signature_table(residual: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "l2": np.linalg.norm(residual, axis=1),
        "l1": np.abs(residual).sum(axis=1),
        "dip9": residual[:, 9] - 0.5 * (residual[:, 8] + residual[:, 10]),
        "peak12": residual[:, 12] - residual[:, 9],
        "dip3": residual[:, 3] - 0.5 * (residual[:, 2] + residual[:, 4]),
        "dip18": residual[:, 18] - 0.5 * (residual[:, 17] + residual[:, 19]),
        "hf": residual[:, 14:].mean(axis=1),
        "mf": residual[:, 6:12].mean(axis=1),
        "lf": residual[:, :5].mean(axis=1),
        "rough": np.abs(np.diff(residual, axis=1)).sum(axis=1),
    }


def build_templates(residual: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for lab in FAULTS:
        mask = labels == lab
        templates[lab] = (
            residual[mask].mean(axis=0) if mask.any() else np.zeros(residual.shape[1])
        )
    return templates


def cosine_to_templates(
    residual: np.ndarray, templates: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    norms = np.clip(np.linalg.norm(residual, axis=1, keepdims=True), 1e-9, None)
    unit = residual / norms
    out = {}
    for lab, vec in templates.items():
        tnorm = max(float(np.linalg.norm(vec)), 1e-9)
        out[lab] = unit @ (vec / tnorm)
    return out


def feature_frame(
    residual: np.ndarray,
    sig: dict[str, np.ndarray],
    cos: dict[str, np.ndarray],
    n_cylinders: np.ndarray,
) -> pd.DataFrame:
    data = {f"res_{i}kHz": residual[:, i] for i in range(21)}
    for k in SIG_KEYS:
        data[k] = sig[k]
    for lab in FAULTS:
        data[f"cos_{lab}"] = cos[lab]
    data["n_cylinders"] = n_cylinders.astype(float)
    return pd.DataFrame(data, columns=FEATURE_NAMES)


def prepare_xy(
    df: pd.DataFrame,
    templates: dict[str, np.ndarray] | None = None,
    labels_for_templates: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    df = punch_spectrum_gaps(df)
    df = interpolate_spectrum(df)
    spectra = df[FREQ_COLS].to_numpy(float)
    engine_ids = df["engine_id"].to_numpy()
    residual = compute_residuals(spectra, engine_ids)
    sig = signature_table(residual)
    if templates is None:
        if labels_for_templates is None:
            raise ValueError("Need templates or labels to build them")
        templates = build_templates(residual, labels_for_templates)
    cos = cosine_to_templates(residual, templates)
    X = feature_frame(residual, sig, cos, df["n_cylinders"].to_numpy())
    return X, residual, sig, templates


def severity_magnitude(sig: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "zakoksowany": sig["peak12"],
        "lejacy": sig["l2"],
        "pompa": sig["l2"],
        "iglica": sig["l2"],
    }


def fit_severity_thresholds(
    labels: np.ndarray,
    severity: np.ndarray,
    mag: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    thr: dict[str, tuple[float, float]] = {}
    for lab in FAULTS:
        groups = {
            sev: mag[lab][(labels == lab) & (severity == sev)] for sev in SEV_ORDER
        }
        t1_def, t2_def = _SEV_DEFAULTS[lab]
        t1 = (
            0.5 * (float(groups["male"].max()) + float(groups["srednie"].min()))
            if len(groups["male"]) and len(groups["srednie"])
            else t1_def
        )
        t2 = (
            0.5 * (float(groups["srednie"].max()) + float(groups["duze"].min()))
            if len(groups["srednie"]) and len(groups["duze"])
            else t2_def
        )
        if t2 <= t1:
            t1, t2 = t1_def, t2_def
        thr[lab] = (t1, t2)
    return thr


def apply_severity(
    pred_label: np.ndarray,
    mag: dict[str, np.ndarray],
    thresholds: dict[str, tuple[float, float]],
) -> np.ndarray:
    out = np.full(len(pred_label), "nie_dotyczy", dtype=object)
    for i, lab in enumerate(pred_label):
        if lab not in FAULTS:
            continue
        t1, t2 = thresholds[lab]
        v = mag[lab][i]
        out[i] = "male" if v < t1 else ("srednie" if v < t2 else "duze")
    return out


def hackathon_score(y_true, y_pred, s_true, s_pred) -> tuple[float, float, float]:
    macro = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    mask = np.isin(y_true, FAULTS)
    sev_acc = accuracy_score(s_true[mask], s_pred[mask]) if mask.any() else 1.0
    return 0.75 * macro + 0.25 * sev_acc, float(macro), float(sev_acc)


def print_eval(name: str, y_true, y_pred, s_true, s_pred) -> tuple[float, float, float]:
    raw, macro, sev_acc = hackathon_score(y_true, y_pred, s_true, s_pred)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    print(f"macro-F1={macro:.4f}  severity_acc={sev_acc:.4f}  Raw_Score={raw:.4f}")
    return raw, macro, sev_acc


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_teacher(device: str, n_estimators: int) -> TabPFNClassifier:
    return TabPFNClassifier.create_default_for_version(
        ModelVersion.V2,
        device=device,
        n_estimators=n_estimators,
        random_state=0,
        ignore_pretraining_limits=True,
    )


def guard_unknown(pred: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """OK vote with a large residual is an unseen anomaly, not a healthy cylinder."""
    out = pred.copy()
    out[(out == "ok") & (l2 > OK_L2_MAX)] = "unknown"
    return out


def tree_path(
    tree: DecisionTreeClassifier, feature_names: list[str], x: np.ndarray
) -> list[dict]:
    t = tree.tree_
    node = 0
    steps: list[dict] = []
    while t.children_left[node] != -1:
        feat_i = int(t.feature[node])
        thr = float(t.threshold[node])
        val = float(x[feat_i])
        go_left = val <= thr
        steps.append(
            {
                "feature": feature_names[feat_i],
                "value": val,
                "op": "≤" if go_left else ">",
                "threshold": thr,
            }
        )
        node = t.children_left[node] if go_left else t.children_right[node]
    return steps


def format_path_pl(steps: list[dict], label: str) -> list[str]:
    lines = []
    for i, s in enumerate(steps, 1):
        name = FEATURE_PL.get(s["feature"], s["feature"])
        lines.append(f"{i}. {name}: {s['value']:.2f} {s['op']} {s['threshold']:.2f}")
    lines.append(f"→ werdykt drzewa: {label}")
    return lines


class TabPFNTreeDiagnoser:
    """Fit TabPFN, distill a tree, predict and explain on CPU."""

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.templates: dict[str, np.ndarray] | None = None
        self.tree: DecisionTreeClassifier | None = None
        self.sev_thr: dict[str, tuple[float, float]] | None = None
        self.feature_names = list(FEATURE_NAMES)
        self.teacher_classes_: np.ndarray | None = None
        self.teacher_ = None
        self.X_val_: pd.DataFrame | None = None
        self.y_val_: np.ndarray | None = None
        self.s_val_: np.ndarray | None = None
        self.sig_val_: dict[str, np.ndarray] | None = None
        self.y_val_hat_: np.ndarray | None = None
        self.meta: dict = {}

    def _xy_from_df(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        assert self.templates is not None
        X, _residual, sig, _ = prepare_xy(df, templates=self.templates)
        return X, sig

    def fit(
        self,
        val: pd.DataFrame,
        train: pd.DataFrame | None = None,
        *,
        device: str | None = None,
        n_estimators: int | None = None,
        do_cv: bool = False,
        conf_min: float = CONF_MIN,
    ) -> TabPFNTreeDiagnoser:
        device = device or pick_device()
        n_estimators = n_estimators or (8 if device == "cuda" else 4)
        print(f"TabPFN teacher  device={device}  n_estimators={n_estimators}")

        y_val = val["label"].to_numpy()
        s_val = val["severity"].to_numpy()
        X_val, _residual_val, sig_val, self.templates = prepare_xy(
            val, labels_for_templates=y_val
        )
        self.sev_thr = fit_severity_thresholds(
            y_val, s_val, severity_magnitude(sig_val)
        )

        if do_cv:
            self.meta["cv"] = self._group_cv(
                val, y_val, s_val, device, max(2, n_estimators // 2)
            )

        teacher = make_teacher(device, n_estimators)
        teacher.fit(X_val.to_numpy(float), y_val)
        self.teacher_ = teacher
        self.teacher_classes_ = teacher.classes_
        y_val_hat = teacher.predict(X_val.to_numpy(float))
        print("TabPFN in-sample on val.csv (sanity, not the holdout score):")
        print(classification_report(y_val, y_val_hat, labels=LABELS, digits=3, zero_division=0))

        self.X_val_ = X_val
        self.y_val_ = y_val
        self.s_val_ = s_val
        self.sig_val_ = sig_val
        self.y_val_hat_ = y_val_hat
        self.meta.update({"device": device, "n_estimators": n_estimators})
        return self.distill(train, conf_min=conf_min)

    def distill(
        self,
        train: pd.DataFrame | None = None,
        *,
        conf_min: float = CONF_MIN,
    ) -> TabPFNTreeDiagnoser:
        """Rebuild the student from val labels [+ TabPFN pseudo-labels on train].

        TabPFN is not refit. Call after ``fit`` (or after a previous ``distill``)
        to compare val-only vs val ∪ train without paying for the teacher twice.
        """
        assert self.teacher_ is not None
        assert self.templates is not None
        assert self.X_val_ is not None and self.y_val_ is not None
        assert self.sig_val_ is not None and self.s_val_ is not None
        assert self.y_val_hat_ is not None and self.sev_thr is not None

        X_val = self.X_val_
        y_val = self.y_val_
        X_parts = [X_val.to_numpy(float)]
        y_parts = [y_val]
        w_parts = [np.full(len(y_val), 2.0)]

        n_pseudo = 0
        n_student = len(y_val)
        if train is not None and len(train):
            X_tr, _res_tr, _sig_tr, _ = prepare_xy(train, templates=self.templates)
            proba = self.teacher_.predict_proba(X_tr.to_numpy(float))
            pred_tr = self.teacher_.classes_[proba.argmax(axis=1)]
            conf = proba.max(axis=1)
            keep = conf >= conf_min
            n_pseudo = int(keep.sum())
            n_student = len(y_val) + n_pseudo
            print(
                f"Pseudo-labels from train.csv: {n_pseudo}/{len(train)} "
                f"with max P ≥ {conf_min:.2f}"
            )
            if n_pseudo:
                print(pd.Series(pred_tr[keep]).value_counts().to_string())
                X_parts.append(X_tr.to_numpy(float)[keep])
                y_parts.append(pred_tr[keep])
                w_parts.append(conf[keep])
            self.meta["train_pseudo"] = {
                "kept": n_pseudo,
                "total": int(len(train)),
                "conf_min": conf_min,
                "label_counts": (
                    pd.Series(pred_tr[keep]).value_counts().to_dict() if n_pseudo else {}
                ),
            }
        else:
            self.meta.pop("train_pseudo", None)

        X_s = np.vstack(X_parts)
        y_s = np.concatenate(y_parts)
        w_s = np.concatenate(w_parts)
        print(f"Student tree distilled on {n_student} rows (val={len(y_val)}, pseudo={n_pseudo})")

        self.tree = DecisionTreeClassifier(
            max_depth=TREE_DEPTH,
            min_samples_leaf=4,
            random_state=self.random_state,
        )
        self.tree.fit(X_s, y_s, sample_weight=w_s)

        y_tree_val = guard_unknown(
            self.tree.predict(X_val.to_numpy(float)), self.sig_val_["l2"]
        )
        mag = severity_magnitude(self.sig_val_)
        s_tree_val = apply_severity(y_tree_val, mag, self.sev_thr)
        raw, macro, sev_acc = hackathon_score(y_val, y_tree_val, self.s_val_, s_tree_val)
        fidelity = float((y_tree_val == self.y_val_hat_).mean())
        print("\nDistilled tree on val.csv (resubstitution, not holdout):")
        print(classification_report(y_val, y_tree_val, labels=LABELS, digits=3, zero_division=0))
        print(
            f"tree vs val labels  macro-F1={macro:.4f}  severity_acc={sev_acc:.4f}  "
            f"Raw_Score={raw:.4f}  fidelity vs TabPFN={fidelity:.3f}"
        )
        self.meta.update(
            {
                "tree_depth": TREE_DEPTH,
                "n_pseudo": n_pseudo,
                "n_student": n_student,
                "conf_min": conf_min,
                "val_raw_score": raw,
                "val_macro_f1": macro,
                "val_sev_acc": sev_acc,
                "fidelity_vs_tabpfn": fidelity,
            }
        )
        return self

    def _group_cv(
        self,
        val: pd.DataFrame,
        y_val: np.ndarray,
        s_val: np.ndarray,
        device: str,
        n_estimators: int,
    ) -> dict:
        """5-fold GroupKFold by engine. Templates rebuilt on training engines only."""
        print("\nGroupKFold(5) by engine — TabPFN teacher ...")
        val = punch_spectrum_gaps(val)
        engines = val["engine_id"].to_numpy()
        residual = compute_residuals(
            interpolate_spectrum(val)[FREQ_COLS].to_numpy(float), engines
        )
        sig = signature_table(residual)
        n_cyl = val["n_cylinders"].to_numpy()
        gkf = GroupKFold(n_splits=5)
        pred_y = np.empty(len(val), dtype=object)
        pred_s = np.empty(len(val), dtype=object)
        dummy = np.zeros(len(val))
        for fold, (tr, te) in enumerate(gkf.split(dummy, y_val, groups=engines), 1):
            templates = build_templates(residual[tr], y_val[tr])
            X_tr = feature_frame(
                residual[tr],
                {k: v[tr] for k, v in sig.items()},
                cosine_to_templates(residual[tr], templates),
                n_cyl[tr],
            )
            X_te = feature_frame(
                residual[te],
                {k: v[te] for k, v in sig.items()},
                cosine_to_templates(residual[te], templates),
                n_cyl[te],
            )
            clf = make_teacher(device, n_estimators)
            clf.fit(X_tr.to_numpy(float), y_val[tr])
            yhat = guard_unknown(clf.predict(X_te.to_numpy(float)), sig["l2"][te])
            mag = severity_magnitude(sig)
            thr = fit_severity_thresholds(
                y_val[tr], s_val[tr], {k: v[tr] for k, v in mag.items()}
            )
            shat = apply_severity(yhat, {k: v[te] for k, v in mag.items()}, thr)
            pred_y[te] = yhat
            pred_s[te] = shat
            print(f"  fold {fold}/5  held-out engines={len(np.unique(engines[te]))}")
        raw, macro, sev_acc = hackathon_score(y_val, pred_y, s_val, pred_s)
        print(classification_report(y_val, pred_y, labels=LABELS, digits=3, zero_division=0))
        print(f"CV  macro-F1={macro:.4f}  severity_acc={sev_acc:.4f}  Raw_Score={raw:.4f}")
        return {"raw": raw, "macro_f1": macro, "sev_acc": sev_acc}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.tree is not None and self.sev_thr is not None
        X, sig = self._xy_from_df(df)
        labels = guard_unknown(self.tree.predict(X.to_numpy(float)), sig["l2"])
        sev = apply_severity(labels, severity_magnitude(sig), self.sev_thr)
        return pd.DataFrame(
            {
                "engine_id": df["engine_id"].to_numpy(),
                "cylinder": df["cylinder"].to_numpy(),
                "label": labels,
                "severity": sev,
            }
        )

    def predict_teacher(self, df: pd.DataFrame, teacher: TabPFNClassifier) -> pd.DataFrame:
        assert self.sev_thr is not None
        X, sig = self._xy_from_df(df)
        labels = guard_unknown(teacher.predict(X.to_numpy(float)), sig["l2"])
        sev = apply_severity(labels, severity_magnitude(sig), self.sev_thr)
        return pd.DataFrame(
            {
                "engine_id": df["engine_id"].to_numpy(),
                "cylinder": df["cylinder"].to_numpy(),
                "label": labels,
                "severity": sev,
            }
        )

    def explain_row(self, df: pd.DataFrame, index: int) -> dict:
        assert self.tree is not None and self.sev_thr is not None
        X, sig = self._xy_from_df(df)
        x = X.to_numpy(float)[index]
        raw_label = str(self.tree.predict(x.reshape(1, -1))[0])
        label = str(guard_unknown(np.array([raw_label]), np.array([sig["l2"][index]]))[0])
        steps = tree_path(self.tree, self.feature_names, x)
        bands: set[int] = set()
        for s in steps:
            bands.update(FEATURE_BANDS.get(s["feature"], ()))
        sev = str(
            apply_severity(
                np.array([label]),
                {k: np.array([v[index]]) for k, v in severity_magnitude(sig).items()},
                self.sev_thr,
            )[0]
        )
        return {
            "label": label,
            "severity": sev,
            "steps": steps,
            "lines": format_path_pl(steps, label),
            "bands_khz": sorted(bands),
            "l2": float(sig["l2"][index]),
            "features": {name: float(x[i]) for i, name in enumerate(self.feature_names)},
        }

    def rules_text(self) -> str:
        assert self.tree is not None
        named = [FEATURE_PL.get(n, n) for n in self.feature_names]
        return export_text(self.tree, feature_names=named, decimals=2)

    def save(self, path: Path | None = None) -> Path:
        ARTIFACTS.mkdir(exist_ok=True)
        path = path or ARTIFACTS / "diagnoser_tree.joblib"
        joblib.dump(
            {
                "tree": self.tree,
                "templates": self.templates,
                "sev_thr": self.sev_thr,
                "feature_names": self.feature_names,
                "meta": self.meta,
            },
            path,
        )
        rules_path = ARTIFACTS / "tree_rules.txt"
        rules_path.write_text(self.rules_text(), encoding="utf-8")
        (ARTIFACTS / "meta.json").write_text(
            json.dumps(self.meta, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {path} and {rules_path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> TabPFNTreeDiagnoser:
        path = path or ARTIFACTS / "diagnoser_tree.joblib"
        blob = joblib.load(path)
        obj = cls()
        obj.tree = blob["tree"]
        obj.templates = blob["templates"]
        obj.sev_thr = blob["sev_thr"]
        obj.feature_names = blob["feature_names"]
        obj.meta = blob.get("meta", {})
        return obj


def leave_one_engine_out(
    labeled: pd.DataFrame,
    train: pd.DataFrame | None = None,
    *,
    device: str | None = None,
    n_estimators: int | None = None,
    conf_min: float = CONF_MIN,
) -> dict[str, dict[str, float]]:
    """Leave-one-engine-out on a labeled pool concatenated in memory.

    Does not read/write ``val.csv`` / ``final_valid.csv``. Per held-out engine:
    TabPFN teacher, tree on fold labels only, tree on fold labels + pseudo-train.
    Templates, teacher and severity cuts never see the held-out engine.
    """
    device = device or pick_device()
    n_estimators = n_estimators or (8 if device == "cuda" else 4)
    labeled = punch_spectrum_gaps(labeled.reset_index(drop=True), verbose=True)
    labeled = interpolate_spectrum(labeled)
    engines = labeled["engine_id"].to_numpy()
    y = labeled["label"].to_numpy()
    s = labeled["severity"].to_numpy()
    residual = compute_residuals(labeled[FREQ_COLS].to_numpy(float), engines)
    sig = signature_table(residual)
    n_cyl = labeled["n_cylinders"].to_numpy()
    mag_all = severity_magnitude(sig)
    unique = np.unique(engines)
    n = len(labeled)
    names = ("tabpfn", "tree_val", "tree_val_train")
    pred_y = {k: np.empty(n, dtype=object) for k in names}
    pred_s = {k: np.empty(n, dtype=object) for k in names}

    print(
        f"LOEO: {len(unique)} silników, {n} cylindrów  "
        f"device={device}  n_estimators={n_estimators}"
    )
    for i, eid in enumerate(unique, 1):
        tr = engines != eid
        te = engines == eid
        templates = build_templates(residual[tr], y[tr])
        X_tr = feature_frame(
            residual[tr],
            {k: v[tr] for k, v in sig.items()},
            cosine_to_templates(residual[tr], templates),
            n_cyl[tr],
        )
        X_te = feature_frame(
            residual[te],
            {k: v[te] for k, v in sig.items()},
            cosine_to_templates(residual[te], templates),
            n_cyl[te],
        )
        teacher = make_teacher(device, n_estimators)
        teacher.fit(X_tr.to_numpy(float), y[tr])
        y_tab = guard_unknown(teacher.predict(X_te.to_numpy(float)), sig["l2"][te])

        tree_val = DecisionTreeClassifier(
            max_depth=TREE_DEPTH, min_samples_leaf=4, random_state=0
        )
        w_val = np.full(int(tr.sum()), 2.0)
        tree_val.fit(X_tr.to_numpy(float), y[tr], sample_weight=w_val)
        y_tree_val = guard_unknown(tree_val.predict(X_te.to_numpy(float)), sig["l2"][te])

        X_parts = [X_tr.to_numpy(float)]
        y_parts = [y[tr]]
        w_parts = [w_val]
        if train is not None and len(train):
            X_unlab, _res, _sig, _ = prepare_xy(train, templates=templates)
            proba = teacher.predict_proba(X_unlab.to_numpy(float))
            pred_unlab = teacher.classes_[proba.argmax(axis=1)]
            conf = proba.max(axis=1)
            keep = conf >= conf_min
            if keep.any():
                X_parts.append(X_unlab.to_numpy(float)[keep])
                y_parts.append(pred_unlab[keep])
                w_parts.append(conf[keep])
        tree_big = DecisionTreeClassifier(
            max_depth=TREE_DEPTH, min_samples_leaf=4, random_state=0
        )
        tree_big.fit(
            np.vstack(X_parts),
            np.concatenate(y_parts),
            sample_weight=np.concatenate(w_parts),
        )
        y_tree_big = guard_unknown(tree_big.predict(X_te.to_numpy(float)), sig["l2"][te])

        thr = fit_severity_thresholds(
            y[tr], s[tr], {k: v[tr] for k, v in mag_all.items()}
        )
        mag_te = {k: v[te] for k, v in mag_all.items()}
        pred_y["tabpfn"][te] = y_tab
        pred_y["tree_val"][te] = y_tree_val
        pred_y["tree_val_train"][te] = y_tree_big
        pred_s["tabpfn"][te] = apply_severity(y_tab, mag_te, thr)
        pred_s["tree_val"][te] = apply_severity(y_tree_val, mag_te, thr)
        pred_s["tree_val_train"][te] = apply_severity(y_tree_big, mag_te, thr)
        if i % 5 == 0 or i == len(unique):
            print(f"  {i}/{len(unique)}  held-out={eid}")

    titles = {
        "tabpfn": "LOEO — TabPFN teacher",
        "tree_val_train": "LOEO — drzewo ← labeled + pseudo-train",
        "tree_val": "LOEO — drzewo ← tylko labeled",
    }
    results: dict[str, dict[str, float]] = {}
    for name in ("tabpfn", "tree_val_train", "tree_val"):
        raw, macro, sev = print_eval(titles[name], y, pred_y[name], s, pred_s[name])
        results[name] = {"raw": float(raw), "macro_f1": float(macro), "sev_acc": float(sev)}
        err = pred_y[name] != y
        if err.any():
            print(f"  błędy ({int(err.sum())}):")
            for j in np.where(err)[0]:
                print(
                    f"    {labeled.iloc[j]['engine_id']} c{int(labeled.iloc[j]['cylinder'])}  "
                    f"true={y[j]}  pred={pred_y[name][j]}"
                )
    return results


def _assert_submit(sub: pd.DataFrame, test: pd.DataFrame) -> None:
    assert len(sub) == len(test)
    assert (sub["engine_id"].to_numpy() == test["engine_id"].to_numpy()).all()
    assert (sub["cylinder"].to_numpy() == test["cylinder"].to_numpy()).all()
    bad_ok = sub["label"].isin(["ok", "unknown"]) & (sub["severity"] != "nie_dotyczy")
    bad_fault = sub["label"].isin(FAULTS) & ~sub["severity"].isin(SEV_ORDER)
    assert not bad_ok.any() and not bad_fault.any()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv",
        action="store_true",
        help="5-fold GroupKFold on TabPFN (slow on CPU, fine on T4)",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="distill the tree on val only (skip unlabeled train.csv)",
    )
    parser.add_argument("--n-estimators", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    val = read_labeled_csv(BASE_DIR / "val.csv")
    holdout = read_labeled_csv(BASE_DIR / "final_valid.csv")
    test = pd.read_csv(BASE_DIR / "test.csv")
    train = None if args.no_train else pd.read_csv(BASE_DIR / "train.csv")
    print("symulacja luk warsztatowych na val / final_valid (train/test już je mają):")
    val = punch_spectrum_gaps(val, verbose=True)
    holdout = punch_spectrum_gaps(holdout, verbose=True)
    if train is not None:
        punch_spectrum_gaps(train, verbose=True)

    overlap = set(val["engine_id"]) & set(holdout["engine_id"])
    assert not overlap, f"wyciek silników val ∩ final_valid: {sorted(overlap)}"
    print(
        f"train val.csv: {val['engine_id'].nunique()} silników, {len(val)} cylindrów"
    )
    print(
        f"test  final_valid.csv: {holdout['engine_id'].nunique()} silników, {len(holdout)} cylindrów"
    )

    model = TabPFNTreeDiagnoser().fit(
        val,
        train,
        device=args.device,
        n_estimators=args.n_estimators,
        do_cv=args.cv,
    )
    model.save()

    assert model.teacher_ is not None
    sub_ho_tree = model.predict(holdout)
    sub_ho_tabpfn = model.predict_teacher(holdout, model.teacher_)
    y_ho = holdout["label"].to_numpy()
    s_ho = holdout["severity"].to_numpy()
    print_eval("final_valid.csv — TabPFN teacher", y_ho, sub_ho_tabpfn["label"].to_numpy(), s_ho, sub_ho_tabpfn["severity"].to_numpy())
    print_eval("final_valid.csv — distilled tree", y_ho, sub_ho_tree["label"].to_numpy(), s_ho, sub_ho_tree["severity"].to_numpy())
    agree_ho = float((sub_ho_tree["label"] == sub_ho_tabpfn["label"]).mean())
    print(f"Tree vs TabPFN agreement on final_valid: {agree_ho:.3f}")
    model.meta["final_valid"] = {
        "n_engines": int(holdout["engine_id"].nunique()),
        "n_rows": int(len(holdout)),
        "tree_raw": float(hackathon_score(y_ho, sub_ho_tree["label"], s_ho, sub_ho_tree["severity"])[0]),
        "teacher_raw": float(hackathon_score(y_ho, sub_ho_tabpfn["label"], s_ho, sub_ho_tabpfn["severity"])[0]),
        "agreement": agree_ho,
    }
    model.save()

    sub_tree = model.predict(test)
    sub_tabpfn = model.predict_teacher(test, model.teacher_)
    _assert_submit(sub_tree, test)
    _assert_submit(sub_tabpfn, test)

    tree_csv = BASE_DIR / "predictions_tree.csv"
    tabpfn_csv = BASE_DIR / "predictions_tabpfn.csv"
    sub_tree.to_csv(tree_csv, index=False)
    sub_tabpfn.to_csv(tabpfn_csv, index=False)
    agree = float((sub_tree["label"] == sub_tabpfn["label"]).mean())
    print(f"\nWrote {tabpfn_csv.name} and {tree_csv.name}")
    print(f"Tree vs TabPFN agreement on unlabeled test: {agree:.3f}")
    print("TabPFN label counts:\n", sub_tabpfn["label"].value_counts().to_string())
    print("Tree label counts:\n", sub_tree["label"].value_counts().to_string())
    print("Submit format OK.")


if __name__ == "__main__":
    main()
