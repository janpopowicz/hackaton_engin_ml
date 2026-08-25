"""Produkcyjne API modelu widmowego (GLRT) — to wrzucasz na stronę.

Skopiuj do repo aplikacji:

    glrt_serve.py
    physics_diagnose.py
    artifacts/spectral_glrt.pkl

Zależności na inferencji: ``numpy``, ``pandas``. Bez torch, bez TabPFN, bez sklearn.

Drop-in zamiast drzewa::

    from glrt_serve import load, predict

    model = load()                  # raz, przy starcie procesu
    payload = predict(engine_df)    # jeden silnik albo wiele

``engine_df`` ma kolumny ``engine_id``, ``cylinder`` oraz ``mV_0``…``mV_20``.
Luki pomiarowe (NaN) zostaw jak są — model je maskuje, nie interpoluje.

``predict`` zwraca JSON-owalny dict: werdykt + liczby fizyczne + krzywe do
wykresu + zdania po polsku, które można wyświetlić mechanikowi.

Eksport artefaktu z tego repo (nie potrzebny na prodzie)::

    python glrt_serve.py --export
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from physics_diagnose import FREQ_COLS, SpectralGLRT
from unknown_clusters import (
    assign_gain_removed,
    load as load_unknown_clusters,
    suggestion_line,
)

BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "artifacts" / "spectral_glrt.pkl"

SEV_PL = {
    "male": "małe",
    "srednie": "średnie",
    "duze": "duże",
    "nie_dotyczy": "nie dotyczy",
}

# pasma, które warto podświetlić przy danym szablonie (kHz)
TEMPLATE_BANDS: dict[str, tuple[int, ...]] = {
    "zakoksowany": (8, 9, 10, 12),
    "lejacy": tuple(range(6, 21)),
    "pompa": (2, 3, 4),
    "iglica": tuple(range(6, 12)),
    "unknown": (),
}


def load(path: str | Path | None = None) -> SpectralGLRT:
    return SpectralGLRT.load(path or DEFAULT_MODEL)


_UNK_STATE: dict | None = None
_UNK_LOADED = False


def _load_unknown_families() -> dict | None:
    global _UNK_STATE, _UNK_LOADED
    if not _UNK_LOADED:
        _UNK_LOADED = True
        try:
            _UNK_STATE = load_unknown_clusters()
        except FileNotFoundError:
            _UNK_STATE = None
    return _UNK_STATE


def _f(x) -> float:
    v = float(x)
    return v if np.isfinite(v) else None


def _vec(arr) -> list:
    out = []
    for v in np.asarray(arr, dtype=float).ravel():
        out.append(None if not np.isfinite(v) else round(float(v), 4))
    return out


def _highlight(label: str, residual: np.ndarray) -> list[int]:
    bands = list(TEMPLATE_BANDS.get(label, ()))
    if bands:
        return bands
    r = np.nan_to_num(residual, nan=0.0)
    if not r.size:
        return []
    return [int(i) for i in np.argsort(np.abs(r))[-3:][::-1]]


def _decision_lines(model: SpectralGLRT, row: dict) -> list[str]:
    z = row["istotnosc_sigma"]
    gof = row["chi_dopasowania"]
    amp = row["amplituda_mV"]
    cand = row["szablon"]
    lab = row["label"]
    sev = row["severity"]
    reason = row["reason"]
    z_det = model.z_detect_
    z_unk = model.z_unknown_
    gof_max = model.gof_max_

    lines = [
        f"Odchylenie od profilu silnika: {z:.1f}σ  "
        f"(próg usterki {z_det:.1f}σ, próg unknown {z_unk:.1f}σ).",
        f"Najbliższy kształt w katalogu: {cand}  "
        f"(χ dopasowania {gof:.2f}, limit {gof_max:.2f}).",
    ]
    if reason == "too_weak" or (lab == "ok" and reason != "unknown"):
        if reason == "too_weak":
            lines.append("Nie odstaje wystarczająco od reszty jednostki — cylinder sprawny.")
        elif reason == "bad_shape":
            lines.append(
                "Kształt nie pasuje do znanej usterki, a odchylenie nie dosięga "
                "progu unknown — traktuję jako sprawny."
            )
        elif reason == "wrong_sign":
            lines.append("Dopasowanie ma zły znak (to nie wkład usterki) — cylinder sprawny.")
        elif reason == "amp_low":
            lines.append("Amplituda poniżej minimum katalogowego — za słabe na usterkę.")
        else:
            lines.append("Cylinder sprawny.")
    elif lab == "unknown":
        lines.append(
            "Odstaje wyraźnie, ale żaden szablon z katalogu nie opisuje kształtu "
            "— inna anomalia (unknown)."
        )
        unk = row.get("unknown_family")
        if unk:
            lines.append(suggestion_line(unk))
    else:
        t1, t2 = model.sev_thr_[lab]
        lines.append(
            f"Amplituda sygnatury {amp:.1f} mV → nasilenie {SEV_PL.get(sev, sev)} "
            f"(progi {t1:.0f} / {t2:.0f} mV)."
        )
    lines.append(f"Werdykt: {lab} / {SEV_PL.get(sev, sev)}.")
    return lines


def predict(df: pd.DataFrame, model: SpectralGLRT | None = None) -> dict:
    """Werdykt + explainability gotowe pod JSON / frontend.

    Zwraca::

        {
          "cylinders": [ {engine_id, cylinder, label, severity,
                          amplituda_mV, istotnosc_sigma, chi_dopasowania,
                          szablon, decision, highlight_khz,
                          spectrum_mV, residual_mV, profile_mV, fitted_fault_mV} ],
          "engines":   [ {engine_id, n_cylinders, n_faults, worst_cylinder, worst_label} ],
        }
    """
    if model is None:
        model = load()
    df = df.reset_index(drop=True)
    missing = [c for c in ("engine_id", "cylinder", *FREQ_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"brak kolumn: {missing}")

    d = model.diagnose(df)
    n = len(df)
    unk_state = _load_unknown_families()
    unk_hits = None
    if unk_state is not None:
        from physics_diagnose import build_pool

        unk_hits = assign_gain_removed(build_pool(df).gain_removed(), unk_state)
    cylinders = []
    for i in range(n):
        lab = str(d["label"][i])
        sev = str(d["severity"][i])
        tpl = str(model.tpl_label_[int(d["best"][i])])
        unk = unk_hits[i] if (unk_hits is not None and lab == "unknown") else None
        highlight = _highlight(lab if lab != "ok" else tpl, d["residual"][i])
        if unk is not None and unk["matched"] and unk["highlight_khz"]:
            highlight = list(unk["highlight_khz"])
        rec = {
            "engine_id": str(d["engine_id"][i]),
            "cylinder": int(d["cylinder"][i]),
            "label": lab,
            "severity": sev,
            "severity_pl": SEV_PL.get(sev, sev),
            "amplituda_mV": round(_f(d["amp_sev"][i]) or 0.0, 2),
            "istotnosc_sigma": round(_f(d["z_anom"][i]) or 0.0, 1),
            "chi_dopasowania": round(_f(d["gof"][i]) or 0.0, 2),
            "szablon": tpl,
            "reason": str(d["reason"][i]),
            "unknown_family": unk,
            "highlight_khz": highlight,
            "khz": list(range(21)),
            "spectrum_mV": _vec(d["spectrum"][i]),
            "residual_mV": _vec(d["residual"][i]),
            "profile_mV": _vec(d["profile"][i]),
            "fitted_fault_mV": _vec(d["fitted_fault"][i]),
        }
        rec["decision"] = _decision_lines(model, rec)
        rec.pop("reason")
        cylinders.append(rec)

    engines = []
    by = pd.DataFrame(cylinders)
    for eid, g in by.groupby("engine_id", sort=False):
        faults = g[g["label"] != "ok"]
        if len(faults):
            worst = faults.sort_values("istotnosc_sigma", ascending=False).iloc[0]
            worst_cyl, worst_lab = int(worst["cylinder"]), str(worst["label"])
        else:
            worst_cyl, worst_lab = None, "ok"
        engines.append(
            {
                "engine_id": str(eid),
                "n_cylinders": int(len(g)),
                "n_faults": int((g["label"] != "ok").sum()),
                "worst_cylinder": worst_cyl,
                "worst_label": worst_lab,
            }
        )
    return {"cylinders": cylinders, "engines": engines}


def predict_table(df: pd.DataFrame, model: SpectralGLRT | None = None) -> pd.DataFrame:
    """Płaska ramka jak ``tree.predict`` + kolumny explain (bez wektorów)."""
    payload = predict(df, model=model)
    rows = []
    for c in payload["cylinders"]:
        rows.append(
            {
                "engine_id": c["engine_id"],
                "cylinder": c["cylinder"],
                "label": c["label"],
                "severity": c["severity"],
                "amplituda_mV": c["amplituda_mV"],
                "istotnosc_sigma": c["istotnosc_sigma"],
                "chi_dopasowania": c["chi_dopasowania"],
                "szablon": c["szablon"],
                "unknown_rodzina": (c["unknown_family"] or {}).get("id")
                if c.get("unknown_family")
                else None,
                "decision": " ".join(c["decision"]),
            }
        )
    return pd.DataFrame(rows)


def export(path: str | Path | None = None, rank: int = 3) -> Path:
    """Fit na val_full + nieoznaczonym train/test i zapisz artefakt."""
    from tabpfn_diagnose import punch_spectrum_gaps, read_labeled_csv

    path = Path(path or DEFAULT_MODEL)
    labeled = punch_spectrum_gaps(
        read_labeled_csv(BASE / "val_full.csv").reset_index(drop=True)
    )
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")
    from physics_diagnose import build_pool, pool_from_frames

    model = SpectralGLRT(rank=rank).fit(
        labeled, pool_from_frames([train, test]), labeled_pool=build_pool(labeled)
    )
    model.save(path)
    print(f"zapisano {path}  sigma={model.sigma_:.2f} mV  szablonów={len(model.tpl_label_)}")
    print("progi nasilenia:", {k: (round(a, 1), round(b, 1)) for k, (a, b) in model.sev_thr_.items()})
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true", help="przelicz i zapisz artifacts/spectral_glrt.pkl")
    ap.add_argument("--demo-engine", default=None, help="np. test_0000 — wypisz predict() na stdout")
    args = ap.parse_args()
    if args.export:
        export()
    if args.demo_engine:
        import json

        test = pd.read_csv(BASE / "test.csv")
        eng = test[test["engine_id"] == args.demo_engine]
        if eng.empty:
            raise SystemExit(f"brak silnika {args.demo_engine}")
        payload = predict(eng)
        print(json.dumps(payload["engines"], ensure_ascii=False, indent=2))
        slim = [{k: c[k] for k in ("cylinder", "label", "severity", "amplituda_mV",
                                   "istotnosc_sigma", "chi_dopasowania", "szablon",
                                   "unknown_family", "decision")}
                for c in payload["cylinders"]]
        print(json.dumps(slim, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
