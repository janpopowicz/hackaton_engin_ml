# Hackathon AI Boost: Diagnostyka Silników Diesla dla Aesteel

Zadaniem uczestników jest przygotowanie narzędzia diagnostycznego dla inżynierów i mechaników firmy Aesteel. System ma na celu wykrywanie uszkodzeń w przemysłowych silnikach Diesla (w wariantach 8, 12 i 16-cylindrowych) na podstawie akustycznych pomiarów wtrysku paliwa, określanie stopnia nasilenia usterki oraz prezentację wyników w przystępnej aplikacji graficznej.

## Dane

Każdy wiersz w plikach reprezentuje pojedynczy cylinder i zawiera widmo akustyczne wtrysku w zakresie 0–20 kHz (kolumny `mV_0` do `mV_20`, próbkowane co 1 kHz w miliwoltach).

W paczce znajdują się cztery pliki CSV:
* `val.csv` (40 silników) – referencyjny zbiór pomiarów z pełnymi etykietami.
* `train.csv` (240 silników) – zbiór archiwalny z warsztatu bez etykiet. Zawiera naturalne szumy oraz braki pomiarowe (puste komórki NaN, ok. 5% punktów).
* `test.csv` (50 silników) – zbiór testowy do oceny końcowej (bez etykiet, zawiera luki pomiarowe).
* `sample_submit.csv` – szablon formatu odpowiedzi.

### Kolumny w plikach:
* `engine_id`: identyfikator silnika (np. `train_0000`, `val_0001`, `test_0000`)
* `cylinder`: numer cylindra (od 1 do 8, 12 lub 16)
* `n_cylinders`: łączna liczba cylindrów w danym silniku (8, 12 lub 16)
* `mV_0` ... `mV_20`: amplituda sygnału akustycznego w mV dla częstotliwości 0–20 kHz
* `label` (tylko w val): kategoria stanu cylindra
* `severity` (tylko w val): stopień nasilenia problemu

## Etykiety diagnostyczne

### Dopuszczalne wartości `label`:
* `ok` – cylinder sprawny
* `zakoksowany`
* `lejacy`
* `pompa`
* `iglica`
* `unknown` – inna anomalia

### Stopnie nasilenia (`severity`):
* Dla klas `zakoksowany`, `lejacy`, `pompa`, `iglica`: `male`, `srednie` lub `duze`.
* Dla klas `ok` oraz `unknown`: zawsze `nie_dotyczy`.

## Materiały pomocnicze

W repozytorium umieszczono gotowe wykresy ułatwiające wstępne rozeznanie w danych:
* `1_przebiegi_usterek.png` – średnie widma akustyczne dla poszczególnych etykiet ze zbioru referencyjnego.
* `2_przebiegi_silnika.png` – przebiegi wszystkich cylindrów przykładowego silnika.

W paczce znajduje się także skrypt `starter.py`, który wczytuje dane, generuje powyższe wykresy oraz tworzy przykładowy plik `baseline_predictions.csv` o prawidłowej strukturze.

## Format oddania projektu

Do końca hackathonu każdy zespół (2–3 osoby) przekazuje link do repozytorium GitHub zawierającego:
1. `predictions.csv` – plik z predykcjami dla wszystkich wierszy ze zbioru `test.csv` (struktura zgodna z `sample_submit.csv`):
```csv
engine_id,cylinder,label,severity
test_0000,1,ok,nie_dotyczy
test_0000,2,zakoksowany,srednie
test_0000,3,unknown,nie_dotyczy
```
2. Kod źródłowy projektu (aplikacja wraz z logiką analityczną / modelem).
3. Plik `README.md` z opisem rozwiązania i instrukcją uruchomienia aplikacji.
4. Prezentację projektu (plik PDF lub slajdy w repozytorium).

Podczas finału odbędzie się krótka prezentacja i demo na żywo (3–5 minut) przed jury.

## Kryteria oceny (100 punktów)

Ocena projektu dzieli się na dwie równe części:

### 1. Część obiektywna (50 punktów) – Skuteczność na zbiorze testowym
Wynik wyliczany na ukrytych etykietach zbioru `test.csv` według metryki:

`Raw_Score = 0.75 * Macro_F1(label) + 0.25 * Accuracy(severity dla uszkodzonych)`

Punkty przyznawane są w skali od progu 0.80 do 1.00:
* Dla `Raw_Score < 0.80`: **0 pkt**
* Dla `Raw_Score >= 0.80`: `Punkty = 50 * (Raw_Score - 0.80) / 0.20` (maksymalnie 50 pkt)

### 2. Część subiektywna (50 punktów) – Aplikacja, UX i prezentacja
* Użyteczność i ergonomia interfejsu (15 pkt): wygoda obsługi oraz czytelność diagnozy dla mechanika.
* Wyjaśnialność decyzji (15 pkt): przejrzyste uzasadnienie werdyktu (np. wskazanie anomalnego pasma, porównanie z pozostałymi cylindrami).
* Kultura pracy i wydajność na CPU (10 pkt): sprawne działanie bez konieczności używania GPU.
* Prezentacja i odpowiedzi na pytania jury (10 pkt).

## Nagrody i Jury

Nagrody:
* 1 miejsce: 1500 zł
* 2 miejsce: 1000 zł
* 3 miejsce: 500 zł

Jury:
* Mateusz Kwietniewski
* Krzysztof Pniaczek
* Jan Kociszewski
