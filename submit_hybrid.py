"""Submisja hybrydowa: klasa z TabPFN, nasilenie z amplitudy modelu widmowego.

Podział zadania idzie po tym, w czym każdy model jest mierzalnie lepszy w LOEO
(patrz ``wyniki.txt``):

* **klasa** -- TabPFN, macro-F1 0.991. Model widmowy ma 0.990, różnica jest
  w granicach jednego cylindra, więc nie ma powodu zmieniać tego, co działa.
* **nasilenie** -- model widmowy, trafność 0.965 wobec 0.895 dla progów na
  normie residuum. Tu przewaga ma przyczynę, nie jest przypadkiem próbki:
  ``||r||`` zawiera szum i jitter wzmocnienia (``E||r||^2 = a^2 + n*sigma^2``),
  więc przy nasileniu ``male`` mierzy głównie zakłócenie. Rzut na kierunek
  klasy ``<r, t>/<t, t>`` jest estymatorem nieobciążonym.

0.965 to zmierzony sufit tego zadania: dwa cylindry w całym zbiorze leżą
dokładnie na granicy przedziałów amplitudy i są nierozstrzygalne.

Uruchomienie:

    python tabpfn_diagnose.py     # wytwarza predictions_tabpfn.csv
    python submit_hybrid.py       # składa predictions_hybrid.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import tabpfn_diagnose as td
from physics_diagnose import (
    FAULTS,
    SEV_ORDER,
    SpectralGLRT,
    build_pool,
    pool_from_frames,
)

BASE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-from", default="predictions_tabpfn.csv")
    ap.add_argument("--out", default="predictions_hybrid.csv")
    ap.add_argument("--rank", type=int, default=3)
    args = ap.parse_args()

    labeled = td.punch_spectrum_gaps(
        td.read_labeled_csv(BASE / "val_full.csv").reset_index(drop=True)
    )
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")

    model = SpectralGLRT(rank=args.rank).fit(
        labeled, pool_from_frames([train, test]), labeled_pool=build_pool(labeled)
    )
    print(f"sigma={model.sigma_:.2f} mV  szablonów={len(model.bases_)}")
    print("progi nasilenia (mV):", {k: (round(v[0], 1), round(v[1], 1)) for k, v in model.sev_thr_.items()})

    src = pd.read_csv(BASE / args.labels_from)
    merged = test[["engine_id", "cylinder"]].merge(src, on=["engine_id", "cylinder"], how="left")
    assert len(merged) == len(test) and merged["label"].notna().all(), "brak klas dla części cylindrów"

    labels = merged["label"].to_numpy()
    out = pd.DataFrame(
        {
            "engine_id": test["engine_id"].to_numpy(),
            "cylinder": test["cylinder"].to_numpy(),
            "label": labels,
            "severity": model.severity_for(labels, test),
        }
    )

    bad_ok = out["label"].isin(["ok", "unknown"]) & (out["severity"] != "nie_dotyczy")
    bad_fault = out["label"].isin(FAULTS) & ~out["severity"].isin(SEV_ORDER)
    assert not bad_ok.any() and not bad_fault.any(), "niespójna para (label, severity)"

    out.to_csv(BASE / args.out, index=False)
    changed = (out["severity"].to_numpy() != merged["severity"].to_numpy()).sum()
    print(f"\nZapisano {args.out}  ({len(out)} wierszy)")
    print(f"nasilenie zmienione względem {args.labels_from}: {changed} cylindrów")
    print(out["label"].value_counts().to_string())
    print(out["severity"].value_counts().to_string())


if __name__ == "__main__":
    main()
