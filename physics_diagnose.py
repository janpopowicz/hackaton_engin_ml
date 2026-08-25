"""Rozwiązanie 3: generatywny model widma (GLRT na podprzestrzeniach usterek).

Model odtworzony z danych:

    widmo[cyl] = (1 + g) * profil_silnika + a * szablon(klasa) + szum(sigma)

Fakty empiryczne, na których stoi konstrukcja (weryfikuje je ``analiza.py``):

1. Każdy cylinder ma **jitter wzmocnienia** ~5%: składnik residuum
   proporcjonalny do profilu silnika. To nie usterka, tylko zakłócenie --
   i to ono, nie szum pomiarowy, dawało większość rozrzutu sprawnych
   cylindrów. Po wyjęciu tego kierunku norma residuum sprawnego cylindra
   spada z ~33 do ~13, czyli efektywna sigma z ~3.0 na ~1.35.
2. Po usunięciu jittera każda klasa usterki jest praktycznie **rank-1**
   (0.84-0.94 wariancji), a rank-2 domyka 0.94-0.96. Usterka to jeden
   kształt przeskalowany amplitudą.
3. Amplituda zależy tylko od nasilenia, z mnożnikiem ~1 / ~1.45 / ~2.0 dla
   ``male`` / ``srednie`` / ``duze``, praktycznie bez zakładkowania -- czyli
   nasilenie jest zadaniem jednowymiarowym, o ile amplitudę mierzy się dobrze.

Stąd cztery decyzje, których nie ma w wariancie TabPFN / drzewo:

* **jitter wzmocnienia jako regresor zakłócający.** Każdy cylinder dopasowuje
  jednocześnie profil silnika i podprzestrzeń usterki, a testowany jest tylko
  wkład usterki. Bez tego "sprawny z wzmocnieniem +5%" jest nieodróżnialny
  od "usterka male".
* **szablony douczane na danych nieoznaczonych.** Etykiet jest 69 usterek,
  a nieoznaczonych anomalii w ``train.csv`` + ``test.csv`` ~500. Podprzestrzenie
  startują ze średnich klas i są przeliczane iteracyjnie (EM) na przypisanych
  anomaliach; odtwarzają wzorce z etykiet z r > 0.98.
* **luki maskowane, nie interpolowane.** Brakujący prążek trafia często
  dokładnie w pasmo diagnostyczne (dołek 9 kHz to sygnatura zakoksowania);
  interpolacja liniowa go zaciera. Dopasowanie ważone maską pomija
  nieobserwowane prążki.
* **progi z rozkładu, nie z siatki.** Próg detekcji bierze się z rozrzutu
  sprawnych cylindrów w danych treningowych, a próg "nieznanego kształtu"
  z jakości dopasowania znanych usterek. Żadna stała nie jest dobierana
  pod wynik walidacji.

Decyzja per cylinder: test chi-kwadrat "czy cokolwiek odstaje", potem wybór
podprzestrzeni o najlepszym dopasowaniu, potem kontrola jakości dopasowania.
Wszystkie wielkości są fizyczne i raportowalne: nazwa szablonu, amplituda
w mV, istotność w sigmach, chi-kwadrat dopasowania.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

FREQ_COLS = [f"mV_{i}" for i in range(21)]
LABELS = ["ok", "zakoksowany", "lejacy", "pompa", "iglica", "unknown"]
FAULTS = ["zakoksowany", "lejacy", "pompa", "iglica"]
SEV_ORDER = ["male", "srednie", "duze"]
UNKNOWN = "unknown"
SEV_MULT = (1.0, 1.45, 2.0)  # mnożniki amplitudy male / srednie / duze


# --------------------------------------------------------------------------
# profil silnika, residua, kierunek jittera
# --------------------------------------------------------------------------
def _masked_engine_baseline(
    spec: np.ndarray, obs: np.ndarray, n_iter: int = 3
) -> np.ndarray:
    """Profil zdrowego silnika z jego własnych cylindrów.

    Start: mediana po cylindrach, odporna na kilka usterek w jednostce.
    Potem iteracje odrzucania cylindrów o dużym residuum i uśredniania
    pozostałych -- średnia po zdrowych ma mniejszy błąd niż mediana po
    wszystkich, a błąd profilu wchodzi wprost w residuum.
    """
    n_bins = spec.shape[1]
    base = np.zeros(n_bins)
    for j in range(n_bins):
        col = spec[obs[:, j], j]
        base[j] = np.median(col) if len(col) else 0.0
    for _ in range(n_iter):
        dev = np.where(obs, spec - base, 0.0)
        rms = np.sqrt((dev**2).sum(1) / obs.sum(1).clip(1))
        keep = rms <= max(np.median(rms) * 2.0, 1e-9)
        if keep.sum() < 3:
            break
        w = (obs & keep[:, None]).astype(float)
        tot = w.sum(0)
        new = np.where(tot > 0, (w * np.nan_to_num(spec)).sum(0) / tot.clip(1), base)
        if np.allclose(new, base, atol=1e-9):
            break
        base = new
    return base


class Pool:
    """Residua, maska obserwacji i kierunek jittera dla zbioru cylindrów."""

    __slots__ = ("res", "obs", "prof")

    def __init__(self, res: np.ndarray, obs: np.ndarray, prof: np.ndarray):
        self.res, self.obs, self.prof = res, obs, prof

    def __len__(self) -> int:
        return len(self.res)

    def take(self, mask: np.ndarray) -> Pool:
        return Pool(self.res[mask], self.obs[mask], self.prof[mask])

    @staticmethod
    def stack(pools: list[Pool]) -> Pool:
        return Pool(
            np.vstack([p.res for p in pools]),
            np.vstack([p.obs for p in pools]),
            np.vstack([p.prof for p in pools]),
        )

    def gain_removed(self) -> np.ndarray:
        """Residua z wyjętym kierunkiem jittera wzmocnienia."""
        R = np.nan_to_num(self.res) * self.obs
        U = self.prof * self.obs
        g = (R * U).sum(1) / (U**2).sum(1).clip(1e-9)
        return R - g[:, None] * U


def build_pool(df: pd.DataFrame) -> Pool:
    """Residua względem profilu własnego silnika + kierunek jittera per wiersz."""
    spec = df[FREQ_COLS].to_numpy(float)
    obs = ~np.isnan(spec)
    res = np.zeros_like(spec)
    prof = np.zeros_like(spec)
    eids = df["engine_id"].to_numpy()
    for eid in pd.unique(eids):
        idx = np.where(eids == eid)[0]
        base = _masked_engine_baseline(spec[idx], obs[idx])
        res[idx] = np.where(obs[idx], spec[idx] - base, 0.0)
        prof[idx] = base / max(float(np.linalg.norm(base)), 1e-9)
    return Pool(res, obs, prof)


def pool_from_frames(dfs: list[pd.DataFrame]) -> Pool:
    """Wspólna pula z kilku ramek. Policz raz i podawaj do każdej fałdy."""
    return Pool.stack(
        [build_pool(d.reset_index(drop=True)) for d in dfs if d is not None and len(d)]
    )


# --------------------------------------------------------------------------
# maskowane najmniejsze kwadraty: [profil silnika | podprzestrzeń usterki]
# --------------------------------------------------------------------------
def _fit_null(pool: Pool) -> tuple[np.ndarray, np.ndarray]:
    """Hipoteza "sprawny": tylko swobodne wzmocnienie. Zwraca SSE i stopnie swobody."""
    R = np.nan_to_num(pool.res) * pool.obs
    U = pool.prof * pool.obs
    Srr = (R**2).sum(1)
    Suu = (U**2).sum(1).clip(1e-9)
    Sru = (R * U).sum(1)
    return (Srr - Sru**2 / Suu).clip(0.0, None), (pool.obs.sum(1) - 1).clip(1)


def _fit_subspace(pool: Pool, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hipoteza "usterka": wzmocnienie + kombinacja wektorów ``basis``.

    ``basis`` ma kształt (r, n_bins). Rozwiązywane są równania normalne
    rozmiaru (1+r) osobno dla każdego wiersza, bo maska obserwacji i profil
    silnika różnią się między wierszami. Zwraca SSE oraz współczynniki bazy.
    """
    n, r = len(pool), len(basis)
    R = np.nan_to_num(pool.res) * pool.obs
    U = pool.prof * pool.obs
    M = pool.obs.astype(float)

    A = np.zeros((n, 1 + r, 1 + r))
    b = np.zeros((n, 1 + r))
    A[:, 0, 0] = (U**2).sum(1)
    b[:, 0] = (R * U).sum(1)
    UB = U @ basis.T
    A[:, 0, 1:] = UB
    A[:, 1:, 0] = UB
    b[:, 1:] = R @ basis.T
    for p in range(r):
        for q in range(p, r):
            v = M @ (basis[p] * basis[q])
            A[:, 1 + p, 1 + q] = v
            A[:, 1 + q, 1 + p] = v
    A[:, np.arange(1 + r), np.arange(1 + r)] += 1e-9
    coef = np.linalg.solve(A, b[:, :, None])[:, :, 0]
    sse = ((R**2).sum(1) - (coef * b).sum(1)).clip(0.0, None)
    return sse, coef[:, 1:]


def sigma_from_pool(pool: Pool) -> float:
    """Sigma szumu po odjęciu jittera, z najcichszych 60% cylindrów."""
    sse0, dof = _fit_null(pool)
    var = sse0 / dof
    return float(np.sqrt(np.median(var[var <= np.percentile(var, 60)]) / 0.62))


def _subspace(rows: np.ndarray, rank: int, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ortonormalna baza rank-``rank`` chmury residuów + kierunek główny.

    Kierunek główny (pierwszy wektor własny, znak zgodny z ``ref``) służy do
    odczytu amplitudy, cała baza -- do dopasowania kształtu.
    """
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    rank = min(rank, len(vt))
    t = vt[0]
    if t @ ref < 0:
        t = -t
    return vt[:rank], t / max(float(np.linalg.norm(t)), 1e-9)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class SpectralGLRT:
    """Klasyfikator ilorazu wiarygodności na podprzestrzeniach usterek.

    Progi nie są strojone pod metrykę: ``z_detect`` bierze się z rozrzutu
    sprawnych cylindrów fałdy treningowej, ``gof_max`` z jakości dopasowania
    znanych usterek tej samej fałdy.
    """

    def __init__(self, *, rank: int = 2, n_unknown: int = 4, z_floor: float = 6.0):
        self.rank = rank
        self.n_unknown = n_unknown
        self.z_floor = z_floor
        self.bases_: list[np.ndarray] = []
        self.dirs_: list[np.ndarray] = []
        self.tpl_label_: list[str] = []
        self.sev_dirs_: dict[str, np.ndarray] = {}
        self.amp_min_: dict[str, float] = {}
        self.sigma_: float = 1.35
        self.z_detect_: float = 9.0
        self.z_unknown_: float = 30.0
        self.gof_max_: float = 3.0
        self.sev_thr_: dict[str, tuple[float, float]] = {}

    # ---- amplituda pod nasilenie ------------------------------------------
    def _sev_amplitude(self, pool: Pool, labels: np.ndarray) -> np.ndarray:
        """Amplituda sygnatury w surowym residuum, bez regresora wzmocnienia.

        Regresor jittera jest niezbędny do detekcji, ale szablony ``lejacy``,
        ``pompa`` i ``iglica`` są z kierunkiem profilu skorelowane w ~0.9,
        więc wspólne dopasowanie rozdziela sygnał między wzmocnienie i usterkę
        w sposób źle uwarunkowany. Nasilenie odczytujemy więc z rzutu na
        surowy kierunek klasy, gdzie skala generatora jest zachowana.
        """
        R = np.nan_to_num(pool.res) * pool.obs
        M = pool.obs.astype(float)
        out = np.zeros(len(pool))
        for lab, t in self.sev_dirs_.items():
            m = labels == lab
            if not m.any():
                continue
            out[m] = (R[m] @ t) / (M[m] @ (t**2)).clip(1e-9)
        return out

    # ---- statystyki decyzyjne --------------------------------------------
    def _stats(self, pool: Pool) -> dict[str, np.ndarray]:
        sse0, dof0 = _fit_null(pool)
        z_anom = (sse0 / self.sigma_**2 - dof0) / np.sqrt(2 * dof0)
        n_obs = pool.obs.sum(1).clip(1)

        sse = np.empty((len(pool), len(self.bases_)))
        amp = np.empty((len(pool), len(self.bases_)))
        for k, (B, t) in enumerate(zip(self.bases_, self.dirs_)):
            s, coef = _fit_subspace(pool, B)
            sse[:, k] = s
            # znak wkładu usterki: rzut dopasowanego sygnału na kierunek klasy
            amp[:, k] = (coef @ B) @ t
        best = sse.argmin(1)
        rows = np.arange(len(pool))
        dof = (n_obs - 1 - self.rank).clip(1)
        return {
            "sse": sse, "amp": amp, "best": best, "z_anom": z_anom,
            "amp_best": amp[rows, best],
            "gof": np.sqrt(sse[rows, best] / dof) / self.sigma_,
        }

    # ---- douczanie podprzestrzeni na danych nieoznaczonych ----------------
    def _refine(self, init: dict[str, np.ndarray], pool: Pool, n_iter: int = 6) -> None:
        """EM: przypisz anomalie nieoznaczone do klas, przelicz podprzestrzenie.

        Anomalie, których nie opisuje żadna znana klasa, są klastrowane
        w rodziny ``unknown``. Dzięki temu "inna anomalia" ma model pozytywny,
        a nie regułę resztkową typu "duże residuum, a klasyfikator mówi ok".
        """
        self.sigma_ = sigma_from_pool(pool)
        names = list(init.keys())
        self.bases_ = [init[k][None, :] for k in names]
        self.dirs_ = [init[k] for k in names]
        self.tpl_label_ = list(names)
        refs = [init[k].copy() for k in names]

        for it in range(n_iter):
            st = self._stats(pool)
            anom = st["z_anom"] >= self.z_detect_
            fit_ok = st["gof"] <= self.gof_max_
            changed = False
            for k in range(len(names)):
                m = anom & fit_ok & (st["best"] == k)
                if m.sum() < 5:
                    continue
                B, t = _subspace(pool.take(m).gain_removed(), self.rank, refs[k])
                if self.bases_[k].shape != B.shape or not np.allclose(self.bases_[k], B, atol=1e-6):
                    changed = True
                self.bases_[k], self.dirs_[k] = B, t
            if not changed and it:
                break

        # kierunki do odczytu amplitudy, też douczone na danych nieoznaczonych.
        # Z etykiet mamy 9-18 usterek na klasę, tu setki -- a kierunek wchodzi
        # wprost w amplitudę, czyli w nasilenie.
        st = self._stats(pool)
        R = np.nan_to_num(pool.res) * pool.obs
        good = (st["z_anom"] >= self.z_detect_) & (st["gof"] <= self.gof_max_)
        self.sev_dirs_ = {}
        for k, lab in enumerate(self.tpl_label_):
            if lab not in FAULTS:
                continue
            m = good & (st["best"] == k) & (st["amp_best"] > 0)
            if m.sum() >= 20:
                t = R[m].mean(0)
                self.sev_dirs_[lab] = t / max(float(np.linalg.norm(t)), 1e-9)

        # rodziny "unknown": anomalie bez dobrego dopasowania do znanych klas
        odd = (st["z_anom"] >= self.z_detect_) & (st["gof"] > self.gof_max_)
        if odd.sum() >= 5 * self.n_unknown:
            P = pool.take(odd).gain_removed()
            unit = P / np.linalg.norm(P, axis=1, keepdims=True).clip(1e-9)
            km = KMeans(self.n_unknown, n_init=25, random_state=0).fit(unit)
            for k in range(self.n_unknown):
                mem = unit[km.labels_ == k]
                if len(mem) < 5:
                    continue
                B, t = _subspace(mem, self.rank, mem.mean(0))
                self.bases_.append(B)
                self.dirs_.append(t)
                self.tpl_label_.append(UNKNOWN)

    # ---- fit --------------------------------------------------------------
    def fit(
        self,
        labeled: pd.DataFrame,
        unlabeled: Pool | list[pd.DataFrame] | None = None,
        *,
        labeled_pool: Pool | None = None,
    ) -> SpectralGLRT:
        labeled = labeled.reset_index(drop=True)
        lp = labeled_pool if labeled_pool is not None else build_pool(labeled)
        y = labeled["label"].to_numpy()
        s = labeled["severity"].to_numpy()
        self.sev_dirs_ = {}

        self.sigma_ = sigma_from_pool(lp)
        self._set_z_thresholds(lp, y)

        # inicjalizacja: średni kształt residuum klasy, bez wkładu wzmocnienia
        P = lp.gain_removed()
        init: dict[str, np.ndarray] = {}
        for lab in FAULTS:
            m = y == lab
            if m.any():
                t = P[m].mean(0)
                init[lab] = t / max(float(np.linalg.norm(t)), 1e-9)

        # próg "nieznanego kształtu" z jakości dopasowania znanych usterek
        self.bases_ = [init[k][None, :] for k in init]
        self.dirs_ = list(init.values())
        self.tpl_label_ = list(init.keys())
        fault_rows = np.isin(y, FAULTS)
        if fault_rows.any():
            self.gof_max_ = float(
                np.percentile(self._stats(lp.take(fault_rows))["gof"], 97) * 1.15
            )

        pool = (
            unlabeled
            if isinstance(unlabeled, Pool)
            else (pool_from_frames(list(unlabeled)) if unlabeled else None)
        )
        if pool is not None:
            self._refine(init, pool)
            # EM przelicza sigmę na puli nieoznaczonej, więc progi idą po nim
            self._set_z_thresholds(lp, y)
            if fault_rows.any():
                self.gof_max_ = float(
                    np.percentile(self._stats(lp.take(fault_rows))["gof"], 97) * 1.15
                )

        # kierunki do odczytu amplitudy: surowe średnie residuum klasy.
        # Jeśli EM wyznaczył je na puli nieoznaczonej, tamte są dokładniejsze.
        R = np.nan_to_num(lp.res) * lp.obs
        for lab in FAULTS:
            m = y == lab
            if m.any() and lab not in self.sev_dirs_:
                t = R[m].mean(0)
                self.sev_dirs_[lab] = t / max(float(np.linalg.norm(t)), 1e-9)

        # progi nasilenia oraz minimalna amplituda uznawana za usterkę
        amp_sev = self._sev_amplitude(lp, y)
        self.sev_thr_, self.amp_min_ = {}, {}
        for lab in FAULTS:
            m = y == lab
            if not m.any():
                self.sev_thr_[lab] = (32.0, 55.0)
                self.amp_min_[lab] = 15.0
                continue
            self.sev_thr_[lab] = self._cuts(amp_sev[m], s[m])
            self.amp_min_[lab] = float(np.percentile(amp_sev[m], 2)) * 0.75

        return self

    def _set_z_thresholds(self, lp: Pool, y: np.ndarray) -> None:
        """Progi detekcji z rozkładów w danych treningowych fałdy.

        ``z_detect_`` z rozrzutu sprawnych cylindrów, ``z_unknown_`` w środku
        (geometrycznym) przerwy między sprawnymi a najsłabszą anomalią spoza
        katalogu usterek. Obie wielkości liczone bez oglądania fałdy testowej.
        """
        sse0, dof0 = _fit_null(lp)
        z = (sse0 / self.sigma_**2 - dof0) / np.sqrt(2 * dof0)
        healthy = z[y == "ok"]
        self.z_detect_ = max(
            self.z_floor,
            float(np.percentile(healthy, 99.5)) if len(healthy) else 9.0,
        )
        unk = z[y == UNKNOWN]
        hi = float(np.min(unk)) if len(unk) else 4.0 * self.z_detect_
        self.z_unknown_ = max(
            self.z_detect_, float(np.sqrt(max(hi, 1e-9) * self.z_detect_))
        )

    @staticmethod
    def _cuts(amp: np.ndarray, sev: np.ndarray) -> tuple[float, float]:
        """Progi w połowie przerwy między poziomami nasilenia.

        Gdy w fałdzie brakuje jakiegoś poziomu, próg odtwarzamy z poziomów
        obecnych przez znane mnożniki generatora, zamiast wpisywać stałą
        wyliczoną na całym zbiorze etykiet.
        """
        g = {k: np.sort(amp[sev == k]) for k in SEV_ORDER}
        anchors = np.array(
            [g[k].mean() / mult if len(g[k]) else np.nan for k, mult in zip(SEV_ORDER, SEV_MULT)]
        )
        ref = float(np.nanmean(anchors)) if np.isfinite(anchors).any() else 28.0
        t1 = (
            0.5 * (g["male"].max() + g["srednie"].min())
            if len(g["male"]) and len(g["srednie"])
            else 1.22 * ref
        )
        t2 = (
            0.5 * (g["srednie"].max() + g["duze"].min())
            if len(g["srednie"]) and len(g["duze"])
            else 1.73 * ref
        )
        if not (0 < t1 < t2):
            t1, t2 = 1.22 * ref, 1.73 * ref
        return float(t1), float(t2)

    # ---- decyzja ----------------------------------------------------------
    def _decide(self, pool: Pool) -> dict[str, np.ndarray]:
        """Trzy warunki, w tej kolejności: coś odstaje, coś pasuje, pasuje sensownie.

        Sprawny cylinder z nietypowym szumem przechodzi pierwszy warunek, ale
        odpada na trzecim -- wkład usterki musi mieć znak i rząd wielkości
        zgodny z usterkami widzianymi w treningu. Bez tego reguła myli
        "szum ponad normę" z "usterka male".
        """
        st = self._stats(pool)
        cand = np.array([self.tpl_label_[k] for k in st["best"]], dtype=object)
        amp_sev = self._sev_amplitude(pool, cand)
        floor = np.array(
            [self.amp_min_.get(c, np.inf) if c in FAULTS else -np.inf for c in cand]
        )

        # "inna anomalia" wymaga wyraźniejszego odstawania niż znana usterka:
        # kształt spoza katalogu poznajemy dopiero, gdy energia nie da się
        # wytłumaczyć ani wzmocnieniem, ani żadną sygnaturą
        need_z = np.where(cand == UNKNOWN, self.z_unknown_, self.z_detect_)
        good_fit = (
            (st["z_anom"] >= need_z)
            & (st["gof"] <= self.gof_max_)
            & (st["amp_best"] > 0)
            & (amp_sev >= floor)
        )
        label = np.where(good_fit, cand, "ok").astype(object)
        label[~good_fit & (st["z_anom"] >= self.z_unknown_)] = UNKNOWN

        sev = np.full(len(pool), "nie_dotyczy", dtype=object)
        amp_final = self._sev_amplitude(pool, label)
        for i, lab in enumerate(label):
            if lab in FAULTS:
                t1, t2 = self.sev_thr_[lab]
                v = amp_final[i]
                sev[i] = "male" if v < t1 else ("srednie" if v < t2 else "duze")
        st["label"] = label
        st["severity"] = sev
        st["amp_sev"] = amp_final
        return st

    def severity_for(
        self, labels: np.ndarray, df: pd.DataFrame | None = None, pool: Pool | None = None
    ) -> np.ndarray:
        """Nasilenie dla klas ustalonych z zewnątrz (np. przez inny model).

        Pozwala złożyć submisję, w której klasę daje najlepszy klasyfikator,
        a nasilenie -- amplituda sygnatury, która jest do tego lepszym
        narzędziem niż norma residuum.
        """
        if pool is None:
            if df is None:
                raise ValueError("podaj df albo pool")
            pool = build_pool(df.reset_index(drop=True))
        amp = self._sev_amplitude(pool, labels)
        out = np.full(len(labels), "nie_dotyczy", dtype=object)
        for i, lab in enumerate(labels):
            if lab in FAULTS:
                t1, t2 = self.sev_thr_[lab]
                out[i] = "male" if amp[i] < t1 else ("srednie" if amp[i] < t2 else "duze")
        return out

    def predict(self, df: pd.DataFrame, pool: Pool | None = None) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        out = self._decide(pool if pool is not None else build_pool(df))
        return pd.DataFrame(
            {
                "engine_id": df["engine_id"].to_numpy(),
                "cylinder": df["cylinder"].to_numpy(),
                "label": out["label"],
                "severity": out["severity"],
            }
        )

    def explain(self, df: pd.DataFrame, pool: Pool | None = None) -> pd.DataFrame:
        """Predykcja wraz z wielkościami, na których oparto decyzję."""
        df = df.reset_index(drop=True)
        out = self._decide(pool if pool is not None else build_pool(df))
        return pd.DataFrame(
            {
                "engine_id": df["engine_id"].to_numpy(),
                "cylinder": df["cylinder"].to_numpy(),
                "label": out["label"],
                "severity": out["severity"],
                "amplituda_mV": out["amp_sev"].round(2),
                "istotnosc_sigma": out["z_anom"].round(1),
                "chi_dopasowania": out["gof"].round(2),
                "szablon": [self.tpl_label_[k] for k in out["best"]],
            }
        )
