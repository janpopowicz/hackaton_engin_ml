"""Workshop UI for the TabPFN → decision-tree diesel diagnoser.

Run after training:
    streamlit run app_tabpfn.py

The app loads the distilled tree (CPU, no GPU, no TabPFN weights).
Each cylinder shows the if/then path that produced the verdict.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tabpfn_diagnose import (
    FEATURE_PL,
    FREQ_COLS,
    TabPFNTreeDiagnoser,
    interpolate_spectrum,
    prepare_xy,
    punch_spectrum_gaps,
)
from unknown_clusters import assign_from_df, load as load_unknown_clusters

BASE = Path(__file__).resolve().parent
KHZ = np.arange(21)

FAULT_COLOR = {
    "ok": "#2ecc71",
    "zakoksowany": "#e67e22",
    "lejacy": "#e74c3c",
    "pompa": "#9b59b6",
    "iglica": "#3498db",
    "unknown": "#7f8c8d",
}

SEV_PL = {
    "male": "małe",
    "srednie": "średnie",
    "duze": "duże",
    "nie_dotyczy": "nie dotyczy",
}


@st.cache_resource
def load_model() -> TabPFNTreeDiagnoser:
    path = BASE / "artifacts" / "diagnoser_tree.joblib"
    if not path.exists():
        raise FileNotFoundError(
            "Brak artifacts/diagnoser_tree.joblib — najpierw: python tabpfn_diagnose.py"
        )
    return TabPFNTreeDiagnoser.load(path)


@st.cache_resource
def load_unknown_families():
    path = BASE / "artifacts" / "unknown_clusters.pkl"
    if not path.exists():
        return None
    return load_unknown_clusters(path)


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE / name)


def spectrum_figure(
    engine: pd.DataFrame,
    highlight_khz: list[int],
    selected_cyl: int,
) -> go.Figure:
    engine = interpolate_spectrum(punch_spectrum_gaps(engine))
    fig = go.Figure()
    for _, row in engine.iterrows():
        cyl = int(row["cylinder"])
        spec = row[FREQ_COLS].to_numpy(float)
        is_sel = cyl == selected_cyl
        fig.add_trace(
            go.Scatter(
                x=KHZ,
                y=spec,
                mode="lines",
                name=f"cyl. {cyl}",
                line=dict(width=3 if is_sel else 1.2),
                opacity=1.0 if is_sel else 0.35,
            )
        )
    for k in highlight_khz:
        fig.add_vrect(x0=k - 0.4, x1=k + 0.4, fillcolor="#f1c40f", opacity=0.15, line_width=0)
    fig.update_layout(
        xaxis_title="częstotliwość [kHz]",
        yaxis_title="amplituda [mV]",
        height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def residual_figure(residual: np.ndarray, highlight_khz: list[int], cyl_pos: int) -> go.Figure:
    fig = go.Figure()
    for i in range(len(residual)):
        fig.add_trace(
            go.Scatter(
                x=KHZ,
                y=residual[i],
                mode="lines",
                showlegend=False,
                line=dict(width=3 if i == cyl_pos else 1),
                opacity=1.0 if i == cyl_pos else 0.25,
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#95a5a6")
    for k in highlight_khz:
        fig.add_vrect(x0=k - 0.4, x1=k + 0.4, fillcolor="#f1c40f", opacity=0.15, line_width=0)
    fig.update_layout(
        xaxis_title="częstotliwość [kHz]",
        yaxis_title="residual vs. baseline silnika [mV]",
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Aesteel — diagnostyka TabPFN", layout="wide")
    st.title("Diagnostyka wtrysku Diesla")
    st.caption(
        "Nauczyciel: TabPFN (foundation model). "
        "Student: drzewo decyzyjne na sygnaturach akustycznych — to drzewo liczy werdykt i go tłumaczy."
    )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Silnik")
        source = st.radio(
            "Źródło",
            ["val.csv (trening)", "final_valid.csv (holdout)", "test.csv", "wgraj CSV"],
            index=1,
        )
        uploaded = None
        if source.startswith("wgraj"):
            uploaded = st.file_uploader("CSV z kolumnami mV_0…mV_20", type="csv")
            if uploaded is None:
                st.stop()
            df = pd.read_csv(uploaded)
        elif source.startswith("val"):
            df = load_csv("val.csv")
        elif source.startswith("final_valid"):
            df = load_csv("final_valid.csv")
        else:
            df = load_csv("test.csv")

        engines = sorted(df["engine_id"].unique())
        engine_id = st.selectbox("Jednostka", engines)
        engine = df[df["engine_id"] == engine_id].sort_values("cylinder").reset_index(drop=True)
        cyl = st.selectbox("Cylinder", engine["cylinder"].tolist())
        st.markdown("---")
        st.markdown("**Dlaczego drzewo?**")
        st.write(
            "TabPFN jest dokładny, ale nieczytelny. "
            "Destylacja zamienia go w zestaw progów na dołkach 3/9 kHz i odbiciu 12 kHz — "
            "to samo, co mechanik widzi na analizatorze."
        )
        if model.meta:
            st.markdown("---")
            st.json(model.meta)

    pred = model.predict(engine)
    merged = engine[["cylinder"]].copy()
    merged["label"] = pred["label"].to_numpy()
    merged["severity"] = pred["severity"].to_numpy()
    if "label" in engine.columns:
        merged["label_true"] = engine["label"].to_numpy()
        merged["severity_true"] = engine["severity"].to_numpy()

    n_fault = int((merged["label"] != "ok").sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Cylindry", len(merged))
    c2.metric("Wskazane usterki", n_fault)
    worst = merged.loc[merged["label"] != "ok"]
    c3.metric("Najcięższa", "brak" if worst.empty else f"{worst.iloc[0]['label']} / {SEV_PL.get(worst.iloc[0]['severity'], worst.iloc[0]['severity'])}")

    row_pos = int(np.where(engine["cylinder"].to_numpy() == cyl)[0][0])
    expl = model.explain_row(engine, row_pos)
    bands = expl["bands_khz"]

    unk_state = load_unknown_families()
    unk_hits = assign_from_df(engine, unk_state) if unk_state is not None else None
    unk = unk_hits[row_pos] if (unk_hits is not None and expl["label"] == "unknown") else None
    if unk is not None and unk["matched"]:
        bands = sorted(set(bands) | set(unk["highlight_khz"]))

    left, right = st.columns([1.3, 1])
    with left:
        st.subheader(f"Widmo — {engine_id}")
        st.plotly_chart(spectrum_figure(engine, bands, int(cyl)), use_container_width=True)
        X, residual, sig, _ = prepare_xy(engine, templates=model.templates)
        st.subheader("Residual względem zdrowego profilu silnika")
        st.plotly_chart(residual_figure(residual, bands, row_pos), use_container_width=True)

    with right:
        color = FAULT_COLOR.get(expl["label"], "#333")
        st.subheader("Werdykt drzewa")
        st.markdown(
            f"<div style='padding:1rem;border-radius:12px;background:{color}22;"
            f"border:1px solid {color}'>"
            f"<h2 style='margin:0;color:{color}'>{expl['label']}</h2>"
            f"<p style='margin:0.3rem 0 0'>nasilenie: <b>{SEV_PL.get(expl['severity'], expl['severity'])}</b>"
            f" &nbsp;·&nbsp; L2 residualu: {expl['l2']:.1f}</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Ścieżka decyzji")
        st.write("Żółte pasma na wykresie to progi, które drzewo faktycznie użyło:")
        for line in expl["lines"]:
            st.write(line)
        if expl["steps"]:
            used = ", ".join(
                FEATURE_PL.get(s["feature"], s["feature"]) for s in expl["steps"]
            )
            st.caption(f"Cechy na ścieżce: {used}")
        if unk is not None:
            ucol = unk["color"]
            st.markdown("#### Sugestia rodziny unknown")
            st.markdown(
                f"<div style='padding:0.85rem;border-radius:12px;background:{ucol}14;"
                f"border:1px solid {ucol}'>"
                f"<p style='margin:0;font-size:1.15rem;color:{ucol};font-weight:600'>"
                f"{unk['id'] or '—'} — {unk['name']}</p>"
                f"<p style='margin:0.35rem 0 0;font-size:0.92rem'>{unk['hint']}</p>"
                f"<p style='margin:0.35rem 0 0;font-size:0.85rem;opacity:0.8'>"
                f"podobieństwo kształtu (cosine) {unk['cosine']:.2f}"
                f" &nbsp;·&nbsp; margines {unk['margin']:.2f}"
                f" &nbsp;·&nbsp; nie zmienia etykiety <b>unknown</b></p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Tabela cylindrów")
    show = merged.copy()
    show["severity"] = show["severity"].map(lambda s: SEV_PL.get(s, s))
    if unk_hits is not None:
        show["rodzina_unknown"] = [
            (h["id"] + " · " + h["name"]) if lab == "unknown" and h["matched"] else ""
            for lab, h in zip(show["label"].to_numpy(), unk_hits)
        ]
    st.dataframe(show, use_container_width=True, hide_index=True)

    with st.expander("Pełne drzewo (reguły po destylacji TabPFN)"):
        st.code(model.rules_text(), language="text")


if __name__ == "__main__":
    main()
