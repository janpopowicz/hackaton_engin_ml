"""Uczciwe LOEO na całej puli etykiet + bootstrap na przedziały ufności.

Porównywane podejścia (każde dostaje ten sam podział i te same luki pomiarowe):

* ``glrt``        -- generatywny model widmowy, szablony z danych NIEOZNACZONYCH
* ``tabpfn``      -- nauczyciel z rozwiązania 2
* ``rf`` / ``tree`` -- odniesienie na tych samych cechach co TabPFN

Wszystko, co dotyka etykiet (nazwy szablonów, progi nasilenia, wzorce cosinus,
progi decyzyjne) jest dopasowywane wyłącznie na silnikach treningowych fałdy.
Silnik trzymany z boku nie wpływa na nic poza własnym profilem bazowym --
a ten na teście też jest dostępny, bo ``test.csv`` zawiera pełne silniki.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tabpfn_diagnose as td
from physics_diagnose import FAULTS, LABELS, SpectralGLRT, build_pool, pool_from_frames

BASE = Path(__file__).resolve().parent


def raw_score(y, yp, s, sp) -> tuple[float, float, float]:
    from sklearn.metrics import accuracy_score, f1_score

    macro = f1_score(y, yp, average="macro", labels=LABELS, zero_division=0)
    m = np.isin(y, FAULTS)
    sev = accuracy_score(s[m], sp[m]) if m.any() else 1.0
    return 0.75 * macro + 0.25 * sev, float(macro), float(sev)


def bootstrap_ci(y, yp, s, sp, engines, n_boot=2000, seed=0):
    """Bootstrap po SILNIKACH -- cylindry jednego silnika nie są niezależne."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(engines)
    idx_by_engine = {e: np.where(engines == e)[0] for e in uniq}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_engine[e] for e in pick])
        try:
            out.append(raw_score(y[idx], yp[idx], s[idx], sp[idx])[0])
        except Exception:
            pass
    out = np.array(out)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def paired_test(y, a_pred, b_pred, s, a_sev, b_sev, engines, n_boot=2000, seed=0):
    """Bootstrap sparowany: P(model A >= model B) na tych samych fałdach."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(engines)
    idx_by_engine = {e: np.where(engines == e)[0] for e in uniq}
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_engine[e] for e in pick])
        ra = raw_score(y[idx], a_pred[idx], s[idx], a_sev[idx])[0]
        rb = raw_score(y[idx], b_pred[idx], s[idx], b_sev[idx])[0]
        diffs.append(ra - rb)
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="val_full.csv")
    ap.add_argument("--no-tabpfn", action="store_true")
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--rank", type=int, default=3, help="wymiar podprzestrzeni usterki")
    args = ap.parse_args()

    labeled = td.read_labeled_csv(BASE / args.labeled).reset_index(drop=True)
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")

    # te same luki pomiarowe co w test.csv -- inaczej porównanie jest nieuczciwe
    labeled = td.punch_spectrum_gaps(labeled, verbose=True)
    y = labeled["label"].to_numpy()
    s = labeled["severity"].to_numpy()
    engines = labeled["engine_id"].to_numpy()
    uniq = np.unique(engines)
    n = len(labeled)

    pool = pool_from_frames([train, test])
    lp = build_pool(labeled)
    print(
        f"\nPula nieoznaczona do douczania szablonów: "
        f"{len(train) + len(test)} wierszy, 0 etykiet"
    )

    # --- cechy dla modeli tabelarycznych (jak w tabpfn_diagnose)
    gapped = td.interpolate_spectrum(labeled)
    resid = td.compute_residuals(gapped[td.FREQ_COLS].to_numpy(float), engines)
    sig = td.signature_table(resid)
    n_cyl = labeled["n_cylinders"].to_numpy()
    mag_all = td.severity_magnitude(sig)

    names = ["glrt", "tree", "rf"] + ([] if args.no_tabpfn else ["tabpfn", "hybryda"])
    pred_y = {k: np.empty(n, dtype=object) for k in names}
    pred_s = {k: np.empty(n, dtype=object) for k in names}

    print(f"\nLOEO: {len(uniq)} silników, {n} cylindrów")
    for i, eid in enumerate(uniq, 1):
        tr_m = engines != eid
        te_m = engines == eid

        # ---------------- model generatywny ----------------
        glrt = SpectralGLRT(rank=args.rank).fit(
            labeled[tr_m], pool, labeled_pool=lp.take(tr_m)
        )
        out = glrt.predict(labeled[te_m], lp.take(te_m))
        pred_y["glrt"][te_m] = out["label"].to_numpy()
        pred_s["glrt"][te_m] = out["severity"].to_numpy()

        # ---------------- modele tabelaryczne ----------------
        tpl = td.build_templates(resid[tr_m], y[tr_m])
        X_tr = td.feature_frame(
            resid[tr_m], {k: v[tr_m] for k, v in sig.items()},
            td.cosine_to_templates(resid[tr_m], tpl), n_cyl[tr_m],
        ).to_numpy(float)
        X_te = td.feature_frame(
            resid[te_m], {k: v[te_m] for k, v in sig.items()},
            td.cosine_to_templates(resid[te_m], tpl), n_cyl[te_m],
        ).to_numpy(float)

        thr = td.fit_severity_thresholds(
            y[tr_m], s[tr_m], {k: v[tr_m] for k, v in mag_all.items()}
        )
        mag_te = {k: v[te_m] for k, v in mag_all.items()}

        clfs = {
            "tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=4, random_state=0),
            "rf": RandomForestClassifier(400, random_state=0, n_jobs=-1),
        }
        if not args.no_tabpfn:
            clfs["tabpfn"] = td.make_teacher("cpu", args.n_estimators)
        for nm, clf in clfs.items():
            clf.fit(X_tr, y[tr_m])
            yh = td.guard_unknown(np.asarray(clf.predict(X_te)), sig["l2"][te_m])
            pred_y[nm][te_m] = yh
            pred_s[nm][te_m] = td.apply_severity(yh, mag_te, thr)

        # hybryda: klasa od TabPFN, nasilenie z amplitudy modelu widmowego
        if "hybryda" in names:
            lbl = pred_y["tabpfn"][te_m]
            pred_y["hybryda"][te_m] = lbl
            pred_s["hybryda"][te_m] = glrt.severity_for(lbl, pool=lp.take(te_m))

        if i % 5 == 0 or i == len(uniq):
            print(f"  {i}/{len(uniq)}")

    # --------------------------- raport ---------------------------
    from sklearn.metrics import classification_report

    results = {}
    print("\n" + "=" * 78)
    print(f"{'model':10s} {'macro-F1':>9s} {'sev acc':>9s} {'raw':>8s}   {'95% CI (bootstrap po silnikach)':>32s}")
    print("=" * 78)
    for nm in names:
        raw, macro, sev = raw_score(y, pred_y[nm], s, pred_s[nm])
        lo, hi = bootstrap_ci(y, pred_y[nm], s, pred_s[nm], engines)
        results[nm] = {"raw": raw, "macro_f1": macro, "sev_acc": sev, "ci": [lo, hi]}
        print(f"{nm:10s} {macro:9.4f} {sev:9.4f} {raw:8.4f}   [{lo:.4f}, {hi:.4f}]")

    best = max(names, key=lambda k: results[k]["raw"])
    print(f"\nPorównania sparowane względem '{best}':")
    for nm in names:
        if nm == best:
            continue
        d, lo, hi, p = paired_test(
            y, pred_y[best], pred_y[nm], s, pred_s[best], pred_s[nm], engines
        )
        print(f"  {best} - {nm:8s}: dRaw={d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(lepszy)={p:.3f}")

    for nm in names:
        print(f"\n--- {nm} ---")
        print(classification_report(y, pred_y[nm], labels=LABELS, digits=3, zero_division=0))
        err = pred_y[nm] != y
        sev_err = np.isin(y, FAULTS) & (pred_s[nm] != s) & ~err
        print(f"  błędy klasy: {err.sum()},  poprawna klasa ale złe nasilenie: {sev_err.sum()}")
        for j in np.where(err)[0]:
            print(f"    {engines[j]} c{labeled.cylinder[j]:<2d} true={y[j]:12s} pred={pred_y[nm][j]}")
        for j in np.where(sev_err)[0]:
            print(f"    {engines[j]} c{labeled.cylinder[j]:<2d} {y[j]:12s} sev true={s[j]:8s} pred={pred_s[nm][j]}")

    (BASE / "artifacts").mkdir(exist_ok=True)
    (BASE / "artifacts" / "loeo_comparison.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print("\nZapisano artifacts/loeo_comparison.json")


if __name__ == "__main__":
    main()
