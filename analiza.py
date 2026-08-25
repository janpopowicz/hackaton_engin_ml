"""Dowody empiryczne pod rozwiązanie 3 + audyt zaufania do walidacji.

Uruchom: ``python analiza.py``

Skrypt odpowiada na trzy pytania:

A. Czy wynikom z LOEO można wierzyć -- czy warunki fałdy odpowiadają testowi?
B. Ile z przewagi jednego modelu nad drugim to sygnał, a ile szum próbki?
C. Jaka jest struktura generatora i gdzie leży sufit możliwy do osiągnięcia?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import tabpfn_diagnose as td
from physics_diagnose import (
    FAULTS,
    FREQ_COLS,
    SEV_ORDER,
    build_pool,
    pool_from_frames,
    sigma_from_pool,
)

RULE = "=" * 74


def head(txt: str) -> None:
    print(f"\n{RULE}\n{txt}\n{RULE}")


# --------------------------------------------------------------------------
def a_czy_walidacja_odpowiada_testowi(labeled, train, test) -> None:
    head("A. Czy zbiór z etykietami odpowiada zbiorowi testowemu?")

    gapped = td.punch_spectrum_gaps(labeled)
    print("Braki pomiarowe (udział pustych prążków, udział wierszy z luką):")
    for name, d in [("val_full (surowe)", labeled), ("val_full (+luki)", gapped),
                    ("train", train), ("test", test)]:
        nan = d[FREQ_COLS].isna().to_numpy()
        print(f"  {name:20s} {100 * nan.mean():5.2f}%   {100 * nan.any(1).mean():5.1f}%")

    print("\nRozkłady po symulacji luk (mediana / p90 / p99 normy residuum):")
    for name, d in [("val_full", gapped), ("train", train), ("test", test)]:
        p = build_pool(d.reset_index(drop=True))
        l2 = np.linalg.norm(np.nan_to_num(p.res) * p.obs, axis=1)
        q = np.percentile(l2, [50, 90, 99])
        print(f"  {name:10s} {q[0]:6.2f} {q[1]:7.2f} {q[2]:7.2f}   sigma={sigma_from_pool(p):.3f}")

    print("\nKlasyfikator domenowy (czy da się odróżnić val od testu?):")
    def feats(d):
        p = build_pool(d.reset_index(drop=True))
        R = np.nan_to_num(p.res) * p.obs
        return np.column_stack([R, p.gain_removed(), np.linalg.norm(R, axis=1)])
    Xv, Xt = feats(gapped), feats(test)
    X = np.vstack([Xv, Xt])
    yy = np.r_[np.zeros(len(Xv)), np.ones(len(Xt))]
    gg = np.r_[gapped.engine_id.to_numpy(), test.engine_id.to_numpy()]
    auc = cross_val_score(
        RandomForestClassifier(400, random_state=0, n_jobs=-1), X, yy,
        groups=gg, cv=StratifiedGroupKFold(5, shuffle=True, random_state=0),
        scoring="roc_auc",
    )
    print(f"  AUC = {auc.mean():.3f} +- {auc.std():.3f}   (0.50 = zbiory nieodróżnialne)")
    print(
        "\nWniosek: po dorzuceniu luk pomiarowych val_full jest statystycznie\n"
        "nieodróżnialny od test.csv. Warunki LOEO odpowiadają testowi -- pod\n"
        "warunkiem, że w test.csv rozkład klas jest podobny, czego nie widzimy."
    )


# --------------------------------------------------------------------------
def b_ile_z_przewagi_to_szum(labeled) -> None:
    head("B. Rozmiar próbki a wiarygodność różnic między modelami")

    y = labeled["label"].to_numpy()
    print(f"Wiersze z etykietą: {len(labeled)}, silniki: {labeled.engine_id.nunique()}")
    print("Liczność klas (macro-F1 waży każdą tak samo):")
    for lab, n in labeled.label.value_counts().items():
        print(f"  {lab:12s} {n:4d}")
    print("\nLiczność (klasa, nasilenie) -- na tym liczy się 25% metryki:")
    fr = labeled[labeled.label.isin(FAULTS)]
    print(pd.crosstab(fr.label, fr.severity).to_string())

    print("\nWpływ JEDNEGO przeklasyfikowanego cylindra na Raw_Score:")
    from sklearn.metrics import f1_score
    base = f1_score(y, y, average="macro", labels=td.LABELS, zero_division=0)
    for lab in ["pompa", "unknown", "lejacy", "ok"]:
        idx = np.where(y == lab)[0]
        if not len(idx):
            continue
        yp = y.copy()
        yp[idx[0]] = "ok" if lab != "ok" else "pompa"
        f = f1_score(y, yp, average="macro", labels=td.LABELS, zero_division=0)
        print(f"  jeden błąd na klasie {lab:12s}: macro-F1 {base:.4f} -> {f:.4f}"
              f"  (Raw_Score -{0.75 * (base - f):.4f})")
    print(
        "\nWniosek: klasa pompa ma 9 przykładów, unknown 12. Jeden błąd to\n"
        "ok. 0.01-0.02 Raw_Score. Różnica 0.9581 vs 0.9405 z wyniki.txt to\n"
        "różnica 2-3 cylindrów, czyli mieści się w szumie próbki."
    )


# --------------------------------------------------------------------------
def c_audyt_leakage(labeled, train) -> None:
    head("C. Audyt leakage w obecnym pipeline (tabpfn_diagnose.py)")

    y = labeled["label"].to_numpy()
    s = labeled["severity"].to_numpy()
    gapped = td.punch_spectrum_gaps(labeled)
    resid = td.compute_residuals(
        td.interpolate_spectrum(gapped)[FREQ_COLS].to_numpy(float),
        gapped.engine_id.to_numpy(),
    )
    sig = td.signature_table(resid)
    mag = td.severity_magnitude(sig)

    print("1) Stałe globalne dobrane na CAŁYM zbiorze etykiet, używane w LOEO:")
    l2 = sig["l2"]
    print(f"   OK_L2_MAX = {td.OK_L2_MAX}: sprawnych powyżej progu "
          f"{100 * (l2[y == 'ok'] > td.OK_L2_MAX).mean():.1f}%, "
          f"uszkodzonych {100 * (l2[np.isin(y, FAULTS)] > td.OK_L2_MAX).mean():.1f}%")
    print("   -> to jest granica decyzyjna dopasowana do wszystkich 40 silników,")
    print("      a mimo to stosowana do silnika trzymanego z boku.")

    print(f"\n2) _SEV_DEFAULTS = {td._SEV_DEFAULTS}")
    print("   Fallback odpala się, gdy w fałdzie brakuje jakiegoś poziomu nasilenia.")
    engines = gapped.engine_id.to_numpy()
    hits = 0
    for eid in np.unique(engines):
        tr = engines != eid
        for lab in FAULTS:
            grp = {sv: ((y[tr] == lab) & (s[tr] == sv)).sum() for sv in SEV_ORDER}
            if min(grp.values()) == 0:
                hits += 1
    print(f"   Fałd x klasa z brakującym poziomem: {hits} / {len(np.unique(engines)) * 4}"
          f"  -> tyle razy w LOEO wchodzi stała z pełnego zbioru.")

    print("\n3) Wybór pasm i sygnatur (dip9, peak12, dip3, dip18) powstał z oglądania")
    print("   widm wszystkich 40 silników. LOEO tego wyboru nie powtarza, więc nie")
    print("   mierzy kosztu jego dobrania.")
    print("\n4) TREE_DEPTH, CONF_MIN, keep_frac, min_samples_leaf oraz sam wybór")
    print("   modelu (5 wierszy w wyniki.txt) były oceniane na tej samej metryce.")
    print("   Przy 69 usterkach wybór najlepszego z 5 podbija wynik sam z siebie.")
    print(
        "\nWniosek: nie ma klasycznego wycieku etykiet (silnik testowy nie wchodzi\n"
        "do treningu), ale są trzy stałe i cały wybór modelu dopasowane na tej\n"
        "samej puli. LOEO jest więc lekko optymistyczne, a różnice rzędu 0.02\n"
        "nie mają pokrycia w danych."
    )


# --------------------------------------------------------------------------
def d_struktura_generatora(labeled, train, test) -> None:
    head("D. Struktura generatora: co naprawdę tworzy widmo")

    y = labeled["label"].to_numpy()
    s = labeled["severity"].to_numpy()
    lp = build_pool(labeled)
    R = np.nan_to_num(lp.res) * lp.obs
    P = lp.gain_removed()

    print("1) Jitter wzmocnienia per cylinder (residuum wzdłuż profilu silnika):")
    ok = y == "ok"
    print(f"   sprawne, norma residuum:            p95={np.percentile(np.linalg.norm(R[ok], axis=1), 95):6.2f}"
          f"  max={np.linalg.norm(R[ok], axis=1).max():6.2f}")
    print(f"   sprawne, po usunięciu kierunku profilu: p95={np.percentile(np.linalg.norm(P[ok], axis=1), 95):6.2f}"
          f"  max={np.linalg.norm(P[ok], axis=1).max():6.2f}")
    print(f"   sigma szumu po korekcie: {sigma_from_pool(lp):.3f} mV")
    print("   -> większość rozrzutu sprawnych cylindrów to wzmocnienie, nie szum.")

    print("\n2) Każda klasa usterki jest rank-1 (wariancja w kolejnych składnikach):")
    for lab in FAULTS + ["unknown"]:
        m = y == lab
        _, sv, _ = np.linalg.svd(P[m], full_matrices=False)
        ev = sv**2 / (sv**2).sum()
        print(f"   {lab:12s} n={m.sum():2d}  {ev[:3].round(3)}  suma2={ev[:2].sum():.3f}")
    print("   -> usterki tak; unknown nie (0.58) i dlatego jest osobną kategorią.")

    print("\n3) Nasilenie to mnożnik amplitudy, nie osobny wzorzec:")
    for lab in FAULTS:
        m = y == lab
        t = R[m].mean(0)
        t /= np.linalg.norm(t)
        a = (R[m] @ t) / (lp.obs[m].astype(float) @ (t**2)).clip(1e-9)
        base = np.median(a[s[m] == "male"]) if (s[m] == "male").any() else np.nan
        txt = {sv: np.round(np.sort(a[s[m] == sv]) / base, 2).tolist()
               for sv in SEV_ORDER if (s[m] == sv).any()}
        print(f"   {lab:12s} baza={base:6.1f} mV  krotności: {txt}")
    print("   -> mnożniki ~1 / ~1.45 / ~2.0. Przy dobrej amplitudzie nasilenie")
    print("      jest zadaniem jednowymiarowym z wyraźnymi przerwami.")

    print("\n4) Szablony da się odtworzyć BEZ etykiet, z train.csv + test.csv:")
    pool = pool_from_frames([train, test])
    from sklearn.cluster import KMeans

    from physics_diagnose import _fit_null

    sg = sigma_from_pool(pool)
    sse0, dof0 = _fit_null(pool)
    anom = (sse0 / sg**2 - dof0) / np.sqrt(2 * dof0) >= 12.0
    Pa = pool.gain_removed()[anom]
    unit = Pa / np.linalg.norm(Pa, axis=1, keepdims=True).clip(1e-9)
    km = KMeans(8, n_init=25, random_state=0).fit(unit)
    Tref = {}
    for lab in FAULTS:
        t = P[y == lab].mean(0)
        Tref[lab] = t / np.linalg.norm(t)
    print(f"   anomalii nieoznaczonych: {anom.sum()} (etykietowanych usterek: {np.isin(y, FAULTS).sum()})")
    for k in range(8):
        mem = unit[km.labels_ == k]
        if len(mem) < 8:
            continue
        c = mem.mean(0)
        c /= np.linalg.norm(c)
        cors = {lab: float(c @ t) for lab, t in Tref.items()}
        b = max(cors, key=lambda x: cors[x])
        tag = b if cors[b] > 0.97 else "(rodzina unknown)"
        print(f"   klaster {k} n={len(mem):4d}  r_max={cors[b]:+.3f} -> {tag}")
    print("   -> 4 klastry pokrywają się z klasami usterek (r > 0.97), reszta to")
    print("      rodziny tworzące unknown. Etykiety są potrzebne tylko do nazwania.")


# --------------------------------------------------------------------------
def e_sufit(labeled) -> None:
    head("E. Sufit osiągalny: symulacja z odtworzonego generatora")

    y = labeled["label"].to_numpy()
    s = labeled["severity"].to_numpy()
    lp = build_pool(labeled)
    P = lp.gain_removed()
    sigma = sigma_from_pool(lp)
    T = np.array([(lambda t: t / np.linalg.norm(t))(P[y == lab].mean(0)) for lab in FAULTS])

    amps = {}
    for i, lab in enumerate(FAULTS):
        m = y == lab
        amps[lab] = P[m] @ T[i]

    rng = np.random.default_rng(0)
    N = 60_000
    p = np.array([(y == lab).sum() for lab in FAULTS], float)
    p /= p.sum()
    ci = rng.choice(4, N, p=p)
    a = np.array([rng.choice(amps[FAULTS[c]]) for c in ci])
    a *= rng.uniform(0.92, 1.08, N)
    Rs = a[:, None] * T[ci] + rng.normal(0, sigma, (N, 21))
    pred = (Rs @ T.T).argmax(1)
    print(f"sigma = {sigma:.2f} mV, amplitudy losowane z rozkładu obserwowanego")
    print(f"Trafność rozróżnienia 4 usterek przy znanych szablonach: {(pred == ci).mean():.4f}")
    for i, lab in enumerate(FAULTS):
        m = ci == i
        print(f"  {lab:12s} {(pred[m] == i).mean():.4f}")
    print("\nTrafność w zależności od amplitudy (czyli od nasilenia):")
    for lo, hi in [(0, 30), (30, 45), (45, 70), (70, 200)]:
        m = (a >= lo) & (a < hi)
        if m.sum():
            print(f"  a in [{lo:3d},{hi:3d}) n={m.sum():6d}  {(pred[m] == ci[m]).mean():.4f}")
    print(
        "\nWniosek: przy poprawnym modelu szumu zadanie jest niemal\n"
        "deterministyczne. Sufit macro-F1 jest bliski 1.0, więc 0.98 nie jest\n"
        "granicą problemu, tylko granicą dotychczasowego zestawu cech."
    )


def main() -> None:
    labeled = td.read_labeled_csv(BASE / "val_full.csv").reset_index(drop=True)
    train = pd.read_csv(BASE / "train.csv")
    test = pd.read_csv(BASE / "test.csv")
    a_czy_walidacja_odpowiada_testowi(labeled, train, test)
    b_ile_z_przewagi_to_szum(labeled)
    c_audyt_leakage(labeled, train)
    d_struktura_generatora(labeled, train, test)
    e_sufit(labeled)


if __name__ == "__main__":
    main()
