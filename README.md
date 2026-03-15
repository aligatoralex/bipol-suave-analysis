# BIPOL Regional Jet - SUAVE Analysis Package

**MEiL PW Warszawa | Przedmiot: BIPOL**  
**Samolot: Regional jet, ~40 pasażerów, 2 silniki turbofan, T-tail**

---

## Struktura pakietu

```
bipol_suave/
├── README.md                    ← Ten plik (instrukcja użytkowania)
├── vehicle_definition.py        ← Definicja samolotu (geometria, napęd, masy)
├── mission_analysis.py          ← Analiza misji (wznoszenie, przelot, zniżanie)
├── aerodynamic_analysis.py      ← Analiza aerodynamiczna (biegunowa, L/D)
├── performance_analysis.py      ← Analiza osiągów (V-n, payload-range, wznoszenie)
├── run_all.py                   ← Skrypt główny (uruchamia wszystko)
└── results/                     ← Wyniki (wykresy PNG + dane CSV)
    ├── 01_altitude_profile.png
    ├── 02_velocity_profile.png
    ├── ...
    ├── 19_climb_performance.png
    ├── mission_data.csv
    └── aero_polar_data.csv
```

## Szybki start

### 1. Instalacja SUAVE

```bash
# Klonowanie SUAVE
git clone https://github.com/suavecode/SUAVE.git /tmp/SUAVE
cd /tmp/SUAVE/trunk
pip install -e .

# Zależności
pip install numpy scipy matplotlib
pip install "setuptools<80"  # Wymagane dla pkg_resources
```

### 2. Kompatybilność z Python 3.13 / NumPy 2.x

SUAVE wymaga dwóch poprawek dla nowszych wersji Pythona:

**Poprawka 1**: scipy `cumtrapz` → `cumulative_trapezoid`
```bash
# Zastąp we wszystkich plikach SUAVE:
find /tmp/SUAVE/trunk -name "*.py" -exec sed -i 's/from scipy.integrate import cumtrapz/from scipy.integrate import cumulative_trapezoid as cumtrapz/g' {} +
```

**Poprawka 2**: numpy `np.float(` → `float(`
```bash
find /tmp/SUAVE/trunk -name "*.py" -exec sed -i 's/np\.float(/float(/g' {} +
find /tmp/SUAVE/trunk -name "*.py" -exec sed -i 's/np\.int(/int(/g' {} +
```

**Poprawka 3**: VLM NumPy 2.x kompatybilność  
W pliku `SUAVE/Methods/Aerodynamics/Common/Fidelity_Zero/Lift/VLM.py`:
- Po linii `m_unique, inv = np.unique(mach,return_inverse=True)` dodaj:
  ```python
  inv = inv.ravel()  # numpy 2.x: ensure 1D index
  ```
- Zamień `GAMMA = np.linalg.solve(A,RHS)` na:
  ```python
  if A.ndim == 3 and RHS.ndim == 2:
      GAMMA = np.linalg.solve(A, RHS[:,:,np.newaxis])[:,:,0]
  else:
      GAMMA = np.linalg.solve(A, RHS)
  ```

**Poprawka 4**: Shim dla `scipy.misc.derivative`
```python
# Utwórz /tmp/scipy_compat.py:
import scipy.integrate
if not hasattr(scipy.integrate, 'cumtrapz'):
    from scipy.integrate import cumulative_trapezoid as cumtrapz
    scipy.integrate.cumtrapz = cumtrapz
```

### 3. Uruchomienie analiz

```bash
cd bipol_suave/
python run_all.py       # Uruchom wszystkie analizy
# lub poszczególne moduły:
python vehicle_definition.py     # Tylko definicja samolotu
python mission_analysis.py       # Tylko analiza misji
python aerodynamic_analysis.py   # Tylko aerodynamika
python performance_analysis.py   # Tylko osiągi
```

---

## Modyfikacja parametrów

Wszystkie parametry samolotu znajdują się w sekcji `PARAMS` w pliku `vehicle_definition.py`:

```python
# --- Zmień te wartości aby eksplorować warianty ---
PARAMS.MTOW      = 24035.0   # [kg] Max masa startowa
PARAMS.wing_area = 70.0265   # [m²] Powierzchnia skrzydła
PARAMS.wing_AR   = 10.80     # Wydłużenie
PARAMS.cruise_Mach = 0.78    # Mach przelotowy
PARAMS.cruise_alt  = 12000.0 # [m] Wysokość przelotowa
# ... itd.
```

Po zmianie parametrów uruchom ponownie `python run_all.py`.

---

## Opis analiz

### Faza 1: Definicja samolotu (`vehicle_definition.py`)
- Geometria: skrzydło główne, SOP, SOK, kadłub, gondole
- Napęd: sieć turbofan (11 komponentów: ram, inlet nozzle, LPC, HPC, LPT, HPT, combustor, core nozzle, fan nozzle, fan, thrust)
- Masy: MTOW, OEW, payload, paliwo
- Konfiguracje: base, cruise, takeoff (klapy 20°), landing (klapy 40°)

### Faza 2: Analiza misji (`mission_analysis.py`)
- 7 segmentów: 3× wznoszenie, 1× przelot, 3× zniżanie
- Wyniki: czas lotu, zużycie paliwa, profile prędkości/wysokości
- Wykresy 01-10: profil misji, ciąg vs opór, L/D

### Faza 3: Analiza aerodynamiczna (`aerodynamic_analysis.py`)
- Biegunowa oporu (CD vs CL) przy warunkach przelotowych
- L/D vs CL
- Krzywa nośności (CL vs α)
- Wykresy 11-15: biegunowe przy różnych M

### Faza 4: Analiza osiągów (`performance_analysis.py`)
- Diagram V-n (CS-25): obwiednia manewrowa + porywowa
- Diagram zasięgowo-ładunkowy (Breguet)
- Zasięg specyficzny vs masa
- Prędkość wznoszenia vs wysokość
- Wykresy 16-19

---

## Porównanie z BIPOL Excel

| Parametr | SUAVE | BIPOL Excel | Różnica |
|----------|-------|-------------|---------|
| VS1 [m/s] | 56.78 | 57.39 | -1.1% |
| VA [m/s] | 89.78 | 90.75 | -1.1% |
| CD0 (SUAVE) | 0.0153 | 0.0290 (Cx0) | -47% |
| Max L/D | 17.5 | 15.29 (Kmax) | +14% |
| Cruise L/D | 16.9 | 8.10 (Le_eff) | — |

**Uwagi:**
- SUAVE używa metody VLM (Vortex Lattice) + form factor, co daje niższy CD0 niż metoda komponentowa BIPOL
- V-n diagram zgadza się dobrze (-1.1% na VS1/VA)
- Różnice w L/D wynikają z różnych metod obliczania oporu

---

## Pliki wynikowe

### Wykresy (results/)
| Nr | Plik | Opis |
|----|------|------|
| 01 | altitude_profile.png | Profil wysokości vs czas |
| 02 | velocity_profile.png | TAS i Mach vs czas |
| 03 | throttle.png | Ustawienie przepustnicy |
| 04 | angle_of_attack.png | Kąt natarcia |
| 05 | fuel_burn_rate.png | Zużycie paliwa |
| 06 | vehicle_mass.png | Masa samolotu vs czas |
| 07 | aero_coefficients.png | CL, CD vs czas |
| 08 | lift_to_drag.png | L/D vs czas |
| 09 | drag_breakdown.png | Składowe oporu vs czas |
| 10 | thrust_vs_drag.png | Ciąg vs opór |
| 11 | drag_polar.png | Biegunowa oporu (CD vs CL) |
| 12 | LD_vs_CL.png | Doskonałość vs CL |
| 13 | CL_vs_alpha.png | Krzywa nośności |
| 14 | drag_buildup.png | Rozkład składowych oporu |
| 15 | multi_mach_polars.png | Biegunowe przy różnych M |
| 16 | vn_diagram.png | Diagram V-n (CS-25) |
| 17 | payload_range.png | Diagram zasięgowo-ładunkowy |
| 18 | specific_range.png | Zasięg specyficzny + L/D vs masa |
| 19 | climb_performance.png | Prędkość wznoszenia vs wysokość |

### Dane CSV
- `mission_data.csv` - Pełne dane trajektorii misji
- `aero_polar_data.csv` - Dane biegunowej oporu (α, CL, CD, składowe)

---

## Źródła
- SUAVE: Stanford University Aerospace Vehicle Environment (https://github.com/suavecode/SUAVE)
- CS-25: EASA Certification Specifications for Large Aeroplanes
- BIPOL Excel: BI1-BI6 arkusze (MEiL PW)
- Gudmundsson, "General Aviation Aircraft Design", 2nd ed.
- ESDU: https://itlims-zsis.meil.pw.edu.pl/pomoce/ESDU/esdu.htm
