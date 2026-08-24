# Rozwiązanie 2: TabPFN → drzewo decyzyjne

Dwa etapy, celowo rozdzielone pod kryteria hackathonu:

1. **TabPFN (nauczyciel)** — tabular foundation model. Uczy się mapowania
   `widmo → usterka` z `val.csv`. Na T4 w Colabie trzyma się w pamięci GPU;
   na CPU też wstaje, tylko wolniej.
2. **Drzewo sklearn (student)** — destylacja. TabPFN etykietuje nieoznaczony
   `train.csv`, a płytkie drzewo uczy się na **nazwanych sygnaturach**
   (dołek 9 kHz, odbicie 12 kHz, L2 vs. profil silnika…).
   Aplikacja warsztatowa liczy tylko drzewo: CPU, milisekundy, ścieżka
   if/then dla mechanika.

Nasilenie (`male` / `srednie` / `duze`) zostaje fizyczne — progi na
amplitudzie sygnatury, nie drugi black-box.

## Split etykiet

- `val.csv` — trening (model widzi te silniki)
- `final_valid.csv` — holdout; tu liczymy uczciwy Raw_Score
- `test.csv` — submit bez etykiet

```bash
python tabpfn_diagnose.py --cv    # fit na val, score na final_valid, submit test
```

## Colab T4 (rekomendowane do treningu)

1. Runtime → Change runtime type → **T4 GPU**.
2. Wgraj `val.csv`, `final_valid.csv`, `train.csv`, `test.csv` oraz `tabpfn_diagnose.py`
   (albo cały repozytorium).
3. Notebook: `tabpfn_diagnose.ipynb`, albo:

```bash
pip install -q -r requirements-tabpfn.txt
python tabpfn_diagnose.py --cv
```

`--cv` to 5-fold GroupKFold po `engine_id` **na val** (dodatkowy score, ~kilka minut na T4).
Uczciwy test to zawsze `final_valid.csv`. Bez `--cv` dostajesz od razu artefakty pod aplikację.

Pobierz z Colaba:

- `artifacts/diagnoser_tree.joblib` — drzewo + wzorce + progi severity
- `predictions_tabpfn.csv` — submit nauczyciela (cel: część obiektywna)
- `predictions_tree.csv` — submit studenta (to samo, co liczy aplikacja)

## CPU, lokalnie

```bash
conda activate ai   # lub venv z requirements-tabpfn.txt
python tabpfn_diagnose.py            # destylacja, bez CV
python tabpfn_diagnose.py --no-train # jeszcze szybciej: drzewo tylko na val
streamlit run app_tabpfn.py
```

Pierwsze `fit` TabPFN ściąga wagi v2 z HuggingFace (~minuta). Kolejne starty
biorą je z cache.

## Pliki

| plik | rola |
|---|---|
| `tabpfn_diagnose.py` | nauczyciel + destylacja + submit |
| `tabpfn_diagnose.ipynb` | ten sam pipeline, pod Colab |
| `app_tabpfn.py` | UI warsztatowy, tylko drzewo |
| `diagnose.py` | rozwiązanie 1 (RandomForest + reguły); ten sam split val / final_valid |

## Wyjaśnialność (jury)

Dla każdego cylindra aplikacja pokazuje:

- widmo na tle reszty jednostki,
- residual vs. zdrowy profil silnika,
- **ścieżkę drzewa** w polskim (`dołek przy 9 kHz: −8.1 ≤ −7.4`),
- podświetlone pasma kHz, których drzewo naprawdę użyło.

To nie jest SHAP „dla świętego spokoju” — to dokładnie te same reguły,
którymi model podejmuje decyzję w produkcji.
