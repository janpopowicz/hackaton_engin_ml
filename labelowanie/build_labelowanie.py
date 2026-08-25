"""Porównanie TabPFN vs model widmowy (GLRT) i wykresy rozjazdów do ręcznego labelowania."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physics_diagnose import FREQ_COLS, SpectralGLRT, build_pool, _masked_engine_baseline

BASE = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
PLOTS = OUT / "wykresy"
KHZ = np.arange(21)

FAULT_COLOR = {
    "ok": "#27ae60",
    "zakoksowany": "#e67e22",
    "lejacy": "#c0392b",
    "pompa": "#8e44ad",
    "iglica": "#2980b9",
    "unknown": "#7f8c8d",
}
SEV_PL = {"male": "małe", "srednie": "średnie", "duze": "duże", "nie_dotyczy": "nie dotyczy"}


def interpolate_row(spec: np.ndarray) -> np.ndarray:
    x = np.arange(len(spec), dtype=float)
    m = np.isfinite(spec)
    if m.all():
        return spec
    if m.sum() < 2:
        return np.nan_to_num(spec, nan=0.0)
    out = spec.copy()
    out[~m] = np.interp(x[~m], x[m], spec[m])
    return out


def engine_curves(engine: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = engine[FREQ_COLS].to_numpy(float)
    obs = ~np.isnan(spec)
    base = _masked_engine_baseline(spec, obs)
    res = np.where(obs, spec - base, np.nan)
    return spec, base, res


def class_templates(val: pd.DataFrame) -> dict[str, np.ndarray]:
    """Średnie residuum klasy vs profil silnika, z etykietowanego val_full."""
    out: dict[str, np.ndarray] = {}
    for eid, g in val.groupby("engine_id"):
        spec, base, res = engine_curves(g.reset_index(drop=True))
        for i, lab in enumerate(g["label"].to_numpy()):
            out.setdefault(lab, []).append(res[i])
    means = {}
    for lab, rows in out.items():
        stacked = np.vstack(rows)
        means[lab] = np.nanmean(stacked, axis=0)
    return means


def verdict(lab: str, sev: str) -> str:
    if lab in ("ok", "unknown"):
        return lab
    return f"{lab} / {SEV_PL.get(sev, sev)}"


def plot_case(
    engine: pd.DataFrame,
    cyl: int,
    templates: dict[str, np.ndarray],
    tab_lab: str,
    tab_sev: str,
    glrt_lab: str,
    glrt_sev: str,
    glrt_row: pd.Series | None,
    kind: str,
    path: Path,
) -> dict:
    spec, base, res = engine_curves(engine)
    cyls = engine["cylinder"].to_numpy(int)
    i = int(np.where(cyls == cyl)[0][0])
    n_cyl = int(engine["n_cylinders"].iloc[0])
    eid = str(engine["engine_id"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.4))
    ax_sp, ax_res, ax_tab, ax_glrt = axes.ravel()

    for j, c in enumerate(cyls):
        y = interpolate_row(spec[j])
        if c == cyl:
            continue
        ax_sp.plot(KHZ, y, color="#c5c9ce", lw=1.0, alpha=0.85)
    ax_sp.plot(KHZ, interpolate_row(base), color="#718093", ls="--", lw=1.6, label="profil silnika")
    ax_sp.plot(
        KHZ,
        interpolate_row(spec[i]),
        color="#111111",
        lw=2.6,
        label=f"cyl. {cyl}",
    )
    nan_bins = np.where(~np.isfinite(spec[i]))[0]
    if len(nan_bins):
        ax_sp.scatter(nan_bins, interpolate_row(spec[i])[nan_bins], s=28, c="#e74c3c", zorder=5, label="luka NaN")
    ax_sp.set_title(f"{eid}  cyl. {cyl}/{n_cyl}  — widmo", fontweight="bold")
    ax_sp.set_ylabel("amplituda [mV]")
    ax_sp.set_xticks(KHZ)
    ax_sp.grid(True, alpha=0.28)
    ax_sp.legend(loc="upper right", fontsize=8, framealpha=0.92)

    r = res[i]
    r_plot = interpolate_row(r)
    ax_res.axhline(0, color="#7f8c8d", lw=1.0)
    ax_res.fill_between(KHZ, r_plot, 0, where=r_plot >= 0, color="#e74c3c", alpha=0.22)
    ax_res.fill_between(KHZ, r_plot, 0, where=r_plot < 0, color="#2980b9", alpha=0.22)
    ax_res.plot(KHZ, r_plot, color="#111111", lw=2.2)
    if tab_lab in templates:
        ax_res.plot(KHZ, templates[tab_lab], color=FAULT_COLOR[tab_lab], ls="--", lw=1.6, label=f"szablon TabPFN: {tab_lab}")
    if glrt_lab in templates and glrt_lab != tab_lab:
        ax_res.plot(KHZ, templates[glrt_lab], color=FAULT_COLOR[glrt_lab], ls=":", lw=2.0, label=f"szablon widmo: {glrt_lab}")
    ax_res.set_title("residuum vs profil silnika", fontweight="bold")
    ax_res.set_ylabel("ΔmV")
    ax_res.set_xticks(KHZ)
    ax_res.grid(True, alpha=0.28)
    ax_res.legend(loc="best", fontsize=8, framealpha=0.92)

    def panel_template(ax, lab, who):
        ax.axhline(0, color="#7f8c8d", lw=1.0)
        ax.plot(KHZ, r_plot, color="#111111", lw=2.0, label="ten cylinder")
        if lab in templates:
            t = templates[lab]
            ax.plot(KHZ, t, color=FAULT_COLOR[lab], lw=2.2, label=f"średnia {lab} (val)")
            ax.fill_between(KHZ, t, 0, color=FAULT_COLOR[lab], alpha=0.12)
        ax.set_title(f"{who}: {verdict(lab, tab_sev if who == 'TabPFN' else glrt_sev)}", fontweight="bold", color=FAULT_COLOR.get(lab, "#111"))
        ax.set_xticks(KHZ)
        ax.grid(True, alpha=0.28)
        ax.legend(loc="best", fontsize=8, framealpha=0.92)
        ax.set_ylabel("ΔmV")

    panel_template(ax_tab, tab_lab, "TabPFN")
    panel_template(ax_glrt, glrt_lab, "widmo GLRT")
    ax_tab.set_xlabel("częstotliwość [kHz]")
    ax_glrt.set_xlabel("częstotliwość [kHz]")

    note = "ROZJAZD KLASY" if kind == "label" else "ta sama klasa, różne nasilenie"
    extra = ""
    if glrt_row is not None:
        extra = (
            f"   GLRT: σ={glrt_row['istotnosc_sigma']:.1f}  "
            f"χ={glrt_row['chi_dopasowania']:.2f}  "
            f"a={glrt_row['amplituda_mV']:.1f} mV  "
            f"szablon={glrt_row['szablon']}"
        )
    fig.suptitle(f"{note}  ·  TabPFN {verdict(tab_lab, tab_sev)}   vs   widmo {verdict(glrt_lab, glrt_sev)}{extra}", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)

    return {
        "engine_id": eid,
        "cylinder": int(cyl),
        "n_cylinders": n_cyl,
        "kind": kind,
        "tabpfn_label": tab_lab,
        "tabpfn_severity": tab_sev,
        "widmo_label": glrt_lab,
        "widmo_severity": glrt_sev,
        "spectrum": [None if not np.isfinite(v) else round(float(v), 3) for v in spec[i]],
        "residual": [None if not np.isfinite(v) else round(float(v), 3) for v in r],
        "profile": [round(float(v), 3) for v in base],
        "nan_bins": [int(b) for b in nan_bins],
        "glrt_sigma": None if glrt_row is None else float(glrt_row["istotnosc_sigma"]),
        "glrt_chi": None if glrt_row is None else float(glrt_row["chi_dopasowania"]),
        "glrt_amp": None if glrt_row is None else float(glrt_row["amplituda_mV"]),
        "glrt_szablon": None if glrt_row is None else str(glrt_row["szablon"]),
        "plot": path.name,
    }


def plot_overview(cases: list[dict], title: str, path: Path) -> None:
    n = len(cases)
    if n == 0:
        return
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows), squeeze=False)
    for k, ax in enumerate(axes.ravel()):
        if k >= n:
            ax.axis("off")
            continue
        c = cases[k]
        y = np.array([0.0 if v is None else v for v in c["residual"]], float)
        ax.axhline(0, color="#7f8c8d", lw=0.8)
        ax.fill_between(KHZ, y, 0, where=y >= 0, color="#e74c3c", alpha=0.22)
        ax.fill_between(KHZ, y, 0, where=y < 0, color="#2980b9", alpha=0.22)
        ax.plot(KHZ, y, color="#111", lw=1.6)
        ax.set_title(
            f"{c['engine_id']} c{c['cylinder']}\n"
            f"T:{c['tabpfn_label']}  W:{c['widmo_label']}",
            fontsize=8,
        )
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.grid(True, alpha=0.25)
        if k % cols == 0:
            ax.set_ylabel("ΔmV", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_html(cases: list[dict], stats: dict, path: Path) -> None:
    cards = []
    for i, c in enumerate(cases, 1):
        kind = "rozjazd klasy" if c["kind"] == "label" else "rozjazd nasilenia"
        meta = ""
        if c["glrt_sigma"] is not None:
            meta = (
                f"<p class='meta'>GLRT: istotność {c['glrt_sigma']:.1f}σ · "
                f"χ={c['glrt_chi']:.2f} · amplituda {c['glrt_amp']:.1f} mV · "
                f"szablon {c['glrt_szablon']}</p>"
            )
        cards.append(
            f"""
            <article id="c{i}">
              <h2>{i}. {c['engine_id']} cylinder {c['cylinder']}
                <span class="kind">{kind}</span></h2>
              <p class="verdicts">
                <b>TabPFN:</b> {verdict(c['tabpfn_label'], c['tabpfn_severity'])}
                &nbsp;→&nbsp;
                <b>widmo:</b> {verdict(c['widmo_label'], c['widmo_severity'])}
              </p>
              {meta}
              <img src="wykresy/{c['plot']}" alt="{c['engine_id']} cyl {c['cylinder']}">
            </article>
            """
        )
    html = f"""<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>Do ręcznego labelowania — TabPFN vs widmo</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px auto; max-width: 1100px; color: #222; }}
  h1 {{ margin-bottom: 8px; }}
  .stats {{ background: #f4f5f7; padding: 12px 16px; border-radius: 8px; }}
  article {{ margin: 36px 0 56px; }}
  img {{ width: 100%; border: 1px solid #ddd; }}
  .kind {{ font-size: 0.7em; font-weight: 600; color: #c0392b; margin-left: 8px; }}
  .meta {{ color: #555; font-size: 0.95em; }}
  nav a {{ margin-right: 12px; }}
</style>
</head>
<body>
<h1>Rozjazdy TabPFN vs model widmowy</h1>
<div class="stats">
  <p>Zbiór: <code>test.csv</code>, {stats['n']} cylindrów.</p>
  <p><b>Klasa zgadza się w {stats['agree_label_pct']:.2f}%</b> ({stats['agree_label_n']}/{stats['n']}).</p>
  <p>Klasa + nasilenie: {stats['agree_both_pct']:.2f}% ({stats['agree_both_n']}/{stats['n']}).</p>
  <p>Do ręcznego labelowania: {stats['n_label']} rozjazdów klasy + {stats['n_sev']} rozjazdów nasilenia.</p>
  <p>CSV: <code>do_recznego_labelowania.csv</code> — kolumny <code>human_label</code>, <code>human_severity</code>, <code>notes</code>.</p>
</div>
<nav>
  <p>Skok do przypadku:
  {''.join(f'<a href="#c{i}">{i}</a>' for i in range(1, len(cases)+1))}
  </p>
</nav>
{''.join(cards)}
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)

    tab = pd.read_csv(BASE / "predictions_tabpfn.csv")
    glrt_pred = pd.read_csv(BASE / "predictions_glrt.csv")
    test = pd.read_csv(BASE / "test.csv")
    val = pd.read_csv(BASE / "val_full.csv")

    tab.to_csv(OUT / "predictions_tabpfn.csv", index=False)
    glrt_pred.to_csv(OUT / "predictions_widmo.csv", index=False)

    m = tab.merge(glrt_pred, on=["engine_id", "cylinder"], suffixes=("_tabpfn", "_widmo"))
    m = m.merge(test[["engine_id", "cylinder", "n_cylinders"]], on=["engine_id", "cylinder"])
    m["agree_label"] = m["label_tabpfn"] == m["label_widmo"]
    m["agree_severity"] = m["severity_tabpfn"] == m["severity_widmo"]
    m["agree_both"] = m["agree_label"] & m["agree_severity"]
    m.to_csv(OUT / "porownanie.csv", index=False)

    n = len(m)
    stats = {
        "n": int(n),
        "agree_label_n": int(m["agree_label"].sum()),
        "agree_label_pct": float(100 * m["agree_label"].mean()),
        "agree_sev_n": int(m["agree_severity"].sum()),
        "agree_sev_pct": float(100 * m["agree_severity"].mean()),
        "agree_both_n": int(m["agree_both"].sum()),
        "agree_both_pct": float(100 * m["agree_both"].mean()),
    }

    model = SpectralGLRT.load(BASE / "artifacts" / "spectral_glrt.pkl")
    explain = model.explain(test)
    explain = explain.rename(columns=lambda c: f"glrt_{c}" if c not in ("engine_id", "cylinder") else c)

    templates = class_templates(val)

    dlab = m[~m["agree_label"]].copy()
    dsev = m[m["agree_label"] & ~m["agree_severity"]].copy()
    stats["n_label"] = int(len(dlab))
    stats["n_sev"] = int(len(dsev))

    rows_human = []
    cases: list[dict] = []

    def add_group(frame: pd.DataFrame, kind: str, prefix: str) -> None:
        for k, r in enumerate(frame.itertuples(index=False), 1):
            engine = test[test["engine_id"] == r.engine_id].reset_index(drop=True)
            ginfo = explain[(explain["engine_id"] == r.engine_id) & (explain["cylinder"] == r.cylinder)]
            grow = None
            if len(ginfo):
                row = ginfo.iloc[0]
                grow = pd.Series(
                    {
                        "istotnosc_sigma": row["glrt_istotnosc_sigma"],
                        "chi_dopasowania": row["glrt_chi_dopasowania"],
                        "amplituda_mV": row["glrt_amplituda_mV"],
                        "szablon": row["glrt_szablon"],
                    }
                )
            fname = f"{prefix}_{k:02d}_{r.engine_id}_cyl{int(r.cylinder)}.png"
            case = plot_case(
                engine,
                int(r.cylinder),
                templates,
                r.label_tabpfn,
                r.severity_tabpfn,
                r.label_widmo,
                r.severity_widmo,
                grow,
                kind,
                PLOTS / fname,
            )
            cases.append(case)
            rows_human.append(
                {
                    "engine_id": r.engine_id,
                    "cylinder": int(r.cylinder),
                    "n_cylinders": int(r.n_cylinders),
                    "typ_rozjazdu": "klasa" if kind == "label" else "nasilenie",
                    "tabpfn_label": r.label_tabpfn,
                    "tabpfn_severity": r.severity_tabpfn,
                    "widmo_label": r.label_widmo,
                    "widmo_severity": r.severity_widmo,
                    "glrt_istotnosc_sigma": None if grow is None else grow["istotnosc_sigma"],
                    "glrt_chi_dopasowania": None if grow is None else grow["chi_dopasowania"],
                    "glrt_amplituda_mV": None if grow is None else grow["amplituda_mV"],
                    "glrt_szablon": None if grow is None else grow["szablon"],
                    "human_label": "",
                    "human_severity": "",
                    "notes": "",
                    "wykres": f"wykresy/{fname}",
                }
            )

    add_group(dlab, "label", "klasa")
    add_group(dsev, "severity", "nasilenie")

    human = pd.DataFrame(rows_human)
    human.to_csv(OUT / "do_recznego_labelowania.csv", index=False)

    plot_overview(
        [c for c in cases if c["kind"] == "label"],
        "Rozjazdy klasy — residuum vs profil silnika",
        PLOTS / "overview_klasa.png",
    )
    plot_overview(
        [c for c in cases if c["kind"] == "severity"],
        "Rozjazdy nasilenia (klasa zgodna) — residuum vs profil silnika",
        PLOTS / "overview_nasilenie.png",
    )

    write_html(cases, stats, OUT / "index.html")

    # compact dump for the canvas
    dump = {
        "stats": stats,
        "label_counts_tabpfn": tab["label"].value_counts().to_dict(),
        "label_counts_widmo": glrt_pred["label"].value_counts().to_dict(),
        "crosstab": pd.crosstab(m["label_tabpfn"], m["label_widmo"]).to_dict(),
        "sev_pairs": (
            dsev.assign(p=dsev["severity_tabpfn"] + "→" + dsev["severity_widmo"] + " | " + dsev["label_tabpfn"])["p"]
            .value_counts()
            .to_dict()
        ),
        "cases": cases,
        "templates": {k: [round(float(x), 3) for x in v] for k, v in templates.items()},
        "sev_thr": {k: [float(a), float(b)] for k, (a, b) in model.sev_thr_.items()},
    }
    (OUT / "cases.json").write_text(json.dumps(dump, ensure_ascii=False), encoding="utf-8")

    print("=== pokrycie ===")
    print(f"klasa:          {stats['agree_label_pct']:.2f}%  ({stats['agree_label_n']}/{n})")
    print(f"nasilenie:      {stats['agree_sev_pct']:.2f}%  ({stats['agree_sev_n']}/{n})")
    print(f"klasa+nasilenie:{stats['agree_both_pct']:.2f}%  ({stats['agree_both_n']}/{n})")
    print(f"do labelowania: {len(human)} wierszy")
    print(f"zapisano {OUT}")


if __name__ == "__main__":
    main()
