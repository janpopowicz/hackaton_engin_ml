"""Diesel injection acoustic diagnostics: labels + severity for test.csv.

Approach
--------
1. Interpolate missing spectrum bins along frequency.
2. Subtract a robust per-engine healthy baseline (median of the most
   typical cylinders) so each cylinder is scored relative to its unit.
3. RandomForest on raw + residual + signature features, trained on val.csv.
4. Physics-style post-rules for the four known fault signatures
   (coking notch, leaking broadband drop, pump 3 kHz dip, needle mid-band).
5. Severity from signature magnitude (not a second black-box model).

Train on labeled `val.csv`. Honest score is `final_valid.csv` (held-out
engines that never enter fit). `test.csv` is the unlabeled submit set.
Optional `--loeo` still runs leave-one-engine-out on val.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

BASE_DIR = Path(__file__).resolve().parent
FREQ_COLS = [f"mV_{i}" for i in range(21)]
LABELS = ["ok", "zakoksowany", "lejacy", "pompa", "iglica", "unknown"]
FAULTS = ["zakoksowany", "lejacy", "pompa", "iglica"]
SEV_ORDER = ["male", "srednie", "duze"]

# Residual L2 above this, with an "ok" RF vote, is treated as unknown.
OK_L2_MAX = 32.0


LABELED_COLS = ["engine_id", "cylinder", "n_cylinders", *FREQ_COLS, "label", "severity"]


def read_labeled_csv(path: Path) -> pd.DataFrame:
    """Load val / final_valid even if the file was saved without a header."""
    header = pd.read_csv(path, nrows=0)
    if "engine_id" in header.columns:
        return pd.read_csv(path)
    return pd.read_csv(path, header=None, names=LABELED_COLS)


def interpolate_spectrum(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[FREQ_COLS] = (
        out[FREQ_COLS]
        .interpolate(axis=1, limit_direction="both")
        .clip(lower=0.0)
    )
    return out


def engine_baseline(spectra: np.ndarray, keep_frac: float = 0.7) -> np.ndarray:
    """Median spectrum of cylinders closest to the engine-wide median."""
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
    l2 = np.linalg.norm(residual, axis=1)
    return {
        "l2": l2,
        "l1": np.abs(residual).sum(axis=1),
        "dip9": residual[:, 9] - 0.5 * (residual[:, 8] + residual[:, 10]),
        "peak12": residual[:, 12] - residual[:, 9],
        "dip3": residual[:, 3] - 0.5 * (residual[:, 2] + residual[:, 4]),
        "dip18": residual[:, 18] - 0.5 * (residual[:, 17] + residual[:, 19]),
        "hf": residual[:, 14:].mean(axis=1),
        "mf": residual[:, 6:12].mean(axis=1),
        "lf": residual[:, :5].mean(axis=1),
        "rough": np.abs(np.diff(residual, axis=1)).sum(axis=1),
        "energy": residual.sum(axis=1),
    }


def cosine_to_templates(residual: np.ndarray, templates: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    norms = np.clip(np.linalg.norm(residual, axis=1, keepdims=True), 1e-9, None)
    unit = residual / norms
    out = {}
    for lab, vec in templates.items():
        tnorm = max(float(np.linalg.norm(vec)), 1e-9)
        out[lab] = unit @ (vec / tnorm)
    return out


def build_templates(residual: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    templates = {}
    for lab in FAULTS:
        mask = labels == lab
        templates[lab] = residual[mask].mean(axis=0) if mask.any() else np.zeros(residual.shape[1])
    return templates


def feature_matrix(
    spectra: np.ndarray,
    residual: np.ndarray,
    sig: dict[str, np.ndarray],
) -> np.ndarray:
    unit = residual / np.clip(np.linalg.norm(residual, axis=1, keepdims=True), 1e-6, None)
    extra = np.column_stack(
        [
            sig["dip9"],
            sig["peak12"],
            sig["dip3"],
            sig["dip18"],
            sig["hf"],
            sig["mf"],
            sig["lf"],
            sig["l2"],
            sig["l1"],
            residual[:, 9],
            residual[:, 3],
            residual[:, 12],
            residual[:, 18],
            residual[:, 19],
            sig["rough"],
            spectra.sum(axis=1),
        ]
    )
    return np.hstack([spectra, residual, unit, extra])


def postprocess_labels(
    rf_pred: np.ndarray,
    sig: dict[str, np.ndarray],
    cos: dict[str, np.ndarray],
) -> np.ndarray:
    """Override RF with hard acoustic signatures that are linearly separable."""
    out = rf_pred.copy()
    n = len(out)
    for i in range(n):
        lab = out[i]
        dip9, peak12, l2 = sig["dip9"][i], sig["peak12"][i], sig["l2"][i]
        hf, mf = sig["hf"][i], sig["mf"][i]

        # Unique 9 kHz notch + 12 kHz rebound → coking, regardless of RF.
        if dip9 < -7.5 and peak12 > 16.0 and cos["zakoksowany"][i] > 0.80:
            lab = "zakoksowany"
        elif l2 > 60 and hf < -15 and mf < -18:
            lab = "lejacy"
        elif cos["pompa"][i] > 0.95 and l2 > 20 and dip9 > -5.0 and not (l2 > 60 and hf < -15):
            lab = "pompa"
        elif lab == "zakoksowany" and (dip9 > -6.0 or peak12 < 14.0 or cos["zakoksowany"][i] < 0.70):
            lab = "unknown"
        elif lab == "pompa" and cos["pompa"][i] < 0.55:
            lab = "unknown"
        elif lab == "ok" and cos["iglica"][i] > 0.95 and l2 > 22:
            lab = "iglica"
        elif lab == "ok" and l2 > OK_L2_MAX:
            lab = "unknown"
        elif lab == "lejacy" and l2 < 55:
            lab = "unknown"
        out[i] = lab
    return out


# Fallback cuts if a fold is missing a severity level.
_SEV_DEFAULTS = {
    "zakoksowany": (24.5, 35.5),  # peak12
    "lejacy": (80.0, 118.0),      # l2
    "pompa": (39.0, 68.0),        # l2
    "iglica": (36.0, 60.0),       # l2
}


def fit_severity_thresholds(
    labels: np.ndarray,
    severity: np.ndarray,
    mag: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    """Cut at the midpoint of the gap between adjacent severity groups."""
    thr: dict[str, tuple[float, float]] = {}
    for lab in FAULTS:
        groups = {
            sev: mag[lab][(labels == lab) & (severity == sev)]
            for sev in SEV_ORDER
        }
        t1_def, t2_def = _SEV_DEFAULTS[lab]
        if len(groups["male"]) and len(groups["srednie"]):
            t1 = 0.5 * (float(groups["male"].max()) + float(groups["srednie"].min()))
        else:
            t1 = t1_def
        if len(groups["srednie"]) and len(groups["duze"]):
            t2 = 0.5 * (float(groups["srednie"].max()) + float(groups["duze"].min()))
        else:
            t2 = t2_def
        if t2 <= t1:
            t1, t2 = t1_def, t2_def
        thr[lab] = (t1, t2)
    return thr


def severity_magnitude(sig: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Coking severity tracks the 9→12 kHz rebound, not overall energy.
    return {
        "zakoksowany": sig["peak12"],
        "lejacy": sig["l2"],
        "pompa": sig["l2"],
        "iglica": sig["l2"],
    }


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
        if v < t1:
            out[i] = "male"
        elif v < t2:
            out[i] = "srednie"
        else:
            out[i] = "duze"
    return out


def hackathon_score(y_true, y_pred, s_true, s_pred) -> tuple[float, float, float]:
    macro = f1_score(y_true, y_pred, average="macro", labels=LABELS)
    mask = np.isin(y_true, FAULTS)
    sev_acc = accuracy_score(s_true[mask], s_pred[mask]) if mask.any() else 1.0
    return 0.75 * macro + 0.25 * sev_acc, float(macro), float(sev_acc)


class Diagnoser:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.templates: dict[str, np.ndarray] | None = None
        self.rf: RandomForestClassifier | None = None
        self.sev_thr: dict[str, tuple[float, float]] | None = None

    def _rf(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=500,
            random_state=self.random_state,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
        )

    def fit(self, df: pd.DataFrame) -> "Diagnoser":
        df = interpolate_spectrum(df)
        spectra = df[FREQ_COLS].to_numpy(float)
        engine_ids = df["engine_id"].to_numpy()
        labels = df["label"].to_numpy()
        severity = df["severity"].to_numpy()
        residual = compute_residuals(spectra, engine_ids)
        self.templates = build_templates(residual, labels)
        sig = signature_table(residual)
        feats = feature_matrix(spectra, residual, sig)
        self.rf = self._rf()
        self.rf.fit(feats, labels)
        self.sev_thr = fit_severity_thresholds(labels, severity, severity_magnitude(sig))
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.rf is not None and self.templates is not None and self.sev_thr is not None
        df = interpolate_spectrum(df)
        spectra = df[FREQ_COLS].to_numpy(float)
        engine_ids = df["engine_id"].to_numpy()
        residual = compute_residuals(spectra, engine_ids)
        sig = signature_table(residual)
        cos = cosine_to_templates(residual, self.templates)
        feats = feature_matrix(spectra, residual, sig)
        rf_pred = self.rf.predict(feats)
        labels = postprocess_labels(rf_pred, sig, cos)
        mag = severity_magnitude(sig)
        sev = apply_severity(labels, mag, self.sev_thr)
        return pd.DataFrame(
            {
                "engine_id": df["engine_id"].to_numpy(),
                "cylinder": df["cylinder"].to_numpy(),
                "label": labels,
                "severity": sev,
            }
        )


def leave_one_engine_out(val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """True LOEO: templates, RF and severity cutoffs never see the held-out engine."""
    val = interpolate_spectrum(val)
    spectra = val[FREQ_COLS].to_numpy(float)
    engine_ids = val["engine_id"].to_numpy()
    labels = val["label"].to_numpy()
    severity = val["severity"].to_numpy()
    residual = compute_residuals(spectra, engine_ids)
    sig = signature_table(residual)
    engines = np.unique(engine_ids)

    pred_y = np.empty(len(val), dtype=object)
    pred_s = np.empty(len(val), dtype=object)
    for eid in engines:
        tr = engine_ids != eid
        te = engine_ids == eid
        templates = build_templates(residual[tr], labels[tr])
        cos_te = cosine_to_templates(residual[te], templates)
        feats_tr = feature_matrix(spectra[tr], residual[tr], {k: v[tr] for k, v in sig.items()})
        feats_te = feature_matrix(spectra[te], residual[te], {k: v[te] for k, v in sig.items()})
        rf = RandomForestClassifier(
            n_estimators=400,
            random_state=0,
            class_weight="balanced",
            min_samples_leaf=2,
            n_jobs=1,
        )
        rf.fit(feats_tr, labels[tr])
        rf_pred = rf.predict(feats_te)
        sig_te = {k: v[te] for k, v in sig.items()}
        yhat = postprocess_labels(rf_pred, sig_te, cos_te)
        thr = fit_severity_thresholds(labels[tr], severity[tr], {k: v[tr] for k, v in severity_magnitude(sig).items()})
        shat = apply_severity(yhat, {k: v[te] for k, v in severity_magnitude(sig).items()}, thr)
        pred_y[te] = yhat
        pred_s[te] = shat
    return pred_y, pred_s


def print_eval(name: str, y_true, y_pred, s_true, s_pred) -> tuple[float, float, float]:
    raw, macro, sev_acc = hackathon_score(y_true, y_pred, s_true, s_pred)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, labels=LABELS, digits=3, zero_division=0))
    print("macierz pomyłek  wiersz=true  kolumna=pred")
    print(
        pd.DataFrame(
            confusion_matrix(y_true, y_pred, labels=LABELS),
            index=LABELS,
            columns=LABELS,
        ).to_string()
    )
    print(f"macro-F1(label)={macro:.4f}  severity_acc={sev_acc:.4f}  Raw_Score={raw:.4f}")
    return raw, macro, sev_acc


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
        "--loeo",
        action="store_true",
        help="dodatkowo leave-one-engine-out na val.csv (wolniejsze)",
    )
    args = parser.parse_args()

    val = read_labeled_csv(BASE_DIR / "val.csv")
    holdout = read_labeled_csv(BASE_DIR / "final_valid.csv")
    test = pd.read_csv(BASE_DIR / "test.csv")

    print(
        f"train val.csv: {val['engine_id'].nunique()} silników, {len(val)} cylindrów"
    )
    print(
        f"test  final_valid.csv: {holdout['engine_id'].nunique()} silników, {len(holdout)} cylindrów"
    )
    overlap = set(val["engine_id"]) & set(holdout["engine_id"])
    assert not overlap, f"wyciek silników val ∩ final_valid: {sorted(overlap)}"

    if args.loeo:
        print("\nLeave-one-engine-out on val.csv ...")
        pred_y, pred_s = leave_one_engine_out(val)
        print_eval("LOEO val.csv", val["label"].to_numpy(), pred_y, val["severity"].to_numpy(), pred_s)

    print("\nFitting on val.csv ...")
    model = Diagnoser().fit(val)

    print("Evaluating on final_valid.csv (held-out engines, never used in fit) ...")
    sub_ho = model.predict(holdout)
    print_eval(
        "final_valid.csv",
        holdout["label"].to_numpy(),
        sub_ho["label"].to_numpy(),
        holdout["severity"].to_numpy(),
        sub_ho["severity"].to_numpy(),
    )

    print("\nPredicting unlabeled test.csv ...")
    sub = model.predict(test)
    out_path = BASE_DIR / "predictions.csv"
    sub.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  rows={len(sub)}")
    print(sub["label"].value_counts().to_string())
    print(pd.crosstab(sub["label"], sub["severity"]).to_string())
    _assert_submit(sub, test)
    print("Submit format OK.")


if __name__ == "__main__":
    main()
