import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# 1. Wczytanie danych
val_df = pd.read_csv(BASE_DIR / "val.csv")
train_df = pd.read_csv(BASE_DIR / "train.csv")
test_df = pd.read_csv(BASE_DIR / "test.csv")

freq_cols = [f"mV_{i}" for i in range(21)]
freq_axis = np.arange(21)


def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpoluje ewentualne dziury (NaN) wzdluz pasma czestotliwosci."""
    df_clean = df.copy()
    df_clean[freq_cols] = df_clean[freq_cols].interpolate(
        axis=1, limit_direction="both"
    )
    return df_clean


print("Przetwarzanie danych...")
val_clean = clean_missing(val_df)
test_clean = clean_missing(test_df)

fault_colors = {
    "zakoksowany": "#e67e22",  # Pomaranczowy
    "lejacy": "#0984e3",       # Blekitny / Niebieski
    "pompa": "#8e44ad",        # Fioletowy
    "iglica": "#27ae60",       # Zielony
    "unknown": "#d63031",      # Czerwony
}

# ==========================================
# WYKRES 1: Pelne przebiegi 4 glownych usterek
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
faults = ["zakoksowany", "lejacy", "pompa", "iglica"]

# Obliczenie sredniego profilu dla sprawnych cylindrow (OK)
ok_subset = val_clean[val_clean["label"] == "ok"]
ok_mean = ok_subset[freq_cols].mean(axis=0)

for ax, fault in zip(axes.ravel(), faults):
    # Rysujemy losowa probke sprawnych jako szare tlo
    for _, row in ok_subset.sample(n=min(30, len(ok_subset)), random_state=42).iterrows():
        ax.plot(freq_axis, row[freq_cols].values, color="#dcdde1", alpha=0.5, linewidth=0.8)
    
    # Sredni profil sprawny
    ax.plot(freq_axis, ok_mean, color="#718093", linestyle="--", linewidth=1.5, label="Średni sprawny (OK)")

    # Poszczegolne stopnie nasilenia usterki
    sev_styles = {
        "male": (":", 1.8),
        "srednie": ("-.", 2.2),
        "duze": ("-", 2.6)
    }
    
    fault_subset = val_clean[val_clean["label"] == fault]
    for sev, (ls, lw) in sev_styles.items():
        sev_rows = fault_subset[fault_subset["severity"] == sev]
        if not sev_rows.empty:
            mean_curve = sev_rows[freq_cols].mean(axis=0)
            ax.plot(
                freq_axis, mean_curve,
                color=fault_colors[fault],
                linestyle=ls,
                linewidth=lw,
                label=f"{fault} ({sev})"
            )

    ax.set_title(f"Sygnatura: {fault.upper()}", fontweight="bold", fontsize=12)
    ax.set_xticks(freq_axis)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Amplituda akustyczna [mV]")

axes[1, 0].set_xlabel("Częstotliwość [kHz]")
axes[1, 1].set_xlabel("Częstotliwość [kHz]")

plt.suptitle("Porównanie pełnych widm akustycznych usterek na tle sprawnych cylindrów", fontsize=14, y=0.99)
plt.tight_layout()

out1 = BASE_DIR / "1_przebiegi_usterek.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"Zapisano: {out1.name}")

# ==========================================
# WYKRES 2: Pelne przebiegi wszystkich cylindrow jednego silnika
# ==========================================
example_engine = "val_0033"
engine_data = val_clean[val_clean.engine_id == example_engine]

plt.figure(figsize=(12, 6.5))

# Sprawne cylindry w tle
ok_drawn = False
for _, row in engine_data[engine_data.label == "ok"].iterrows():
    plt.plot(
        freq_axis, row[freq_cols].values,
        color="#bdc3c7", alpha=0.8, linewidth=1.2,
        label="Pozostałe cylindry sprawne (OK)" if not ok_drawn else None
    )
    ok_drawn = True

# Uszkodzone cylindry z dedykowanymi kolorami usterki
for _, row in engine_data[engine_data.label != "ok"].iterrows():
    cyl = int(row["cylinder"])
    label = row["label"]
    severity = row["severity"]
    color = fault_colors.get(label, "#e74c3c")
    plt.plot(
        freq_axis, row[freq_cols].values,
        color=color, linewidth=2.8,
        label=f"Cylinder {cyl}: {label} ({severity})"
    )

n_cyl = engine_data["n_cylinders"].iloc[0]
plt.title(f"Pomiary akustyczne silnika {example_engine} ({n_cyl} cylindrów)", fontsize=13, fontweight="bold")
plt.xlabel("Częstotliwość [kHz]")
plt.ylabel("Amplituda akustyczna [mV]")
plt.xticks(freq_axis)
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right", fontsize=10)
plt.tight_layout()

out2 = BASE_DIR / "2_przebiegi_silnika.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"Zapisano: {out2.name}")

# ==========================================
# 3. Zapis bazowego submitu (weryfikacja formatu)
# ==========================================
sub = test_clean[["engine_id", "cylinder"]].copy()
sub["label"] = "ok"
sub["severity"] = "nie_dotyczy"

out_sub = BASE_DIR / "baseline_predictions.csv"
sub.to_csv(out_sub, index=False)
print(f"Zapisano przykladowy submit: {out_sub.name}")
print("Wszystko gotowe.")
