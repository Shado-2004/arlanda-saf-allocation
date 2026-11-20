# Climate-Optimal SAF Allocation Under Uncertainty | Stockholm Arlanda

> **Which Stockholm departures should receive the available 35%-by-mass SAF blend, how much pure SAF would
> that use, and when would an altitude change help more?**

This is an offline Streamlit app for an Imperial College London MSc aerospace coursework project. It decides
two things for 12 SAS-group A320neo departures from Stockholm Arlanda (ARN) on Tuesday 17 November 2026:

1. **At the fuel deadline:** which flights get the scarce HEFA-SPK/Jet A-1 blend. This is committed before the weather is known.
2. **At each route cutoff:** which flight level to fly, using only the forecast update that dispatch can actually see.

It compares five ways of making those decisions under the same stock, truck slots, deadlines and prices. All
allocations and savings are **counterfactual coursework results**. Real dispatch would also need approved flight
levels, aircraft performance checks and air traffic clearance.

---

## Quick start

### No-install option: the single HTML file

Download **`arlanda-saf-app.html`** from this repo and double-click it. The whole app runs inside the browser
(Chrome, Edge, Firefox or Safari) using [stlite](https://github.com/whitphx/stlite), which runs Streamlit on
WebAssembly. The app code and the data CSVs are embedded in the file. The first load needs internet to fetch the
Python runtime from the jsDelivr CDN and takes about 10–60 seconds. Results are identical to the Python version.
CSV downloads appear as links. Rebuild the file after changing code with `python tools/build_html.py`.

### Full Python install

You need **Python 3.11 or newer** ([python.org/downloads](https://www.python.org/downloads/)). On Windows, tick
*"Add python.exe to PATH"* in the installer. After installation the app needs no internet, accounts or API keys,
because all data is bundled in `data/`.

**Get the code**, either way works:

- `git clone https://github.com/Shado-2004/arlanda-saf-allocation.git`, or
- on the GitHub page click **Code → Download ZIP** and unzip it (rename the folder to `arlanda-saf-allocation` if it ends in `-main`).

### macOS / Linux (Terminal)

```bash
cd arlanda-saf-allocation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

### Windows (PowerShell)

```powershell
cd arlanda-saf-allocation
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

If PowerShell refuses to run `Activate.ps1`, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once
and try again.

The app opens at <http://localhost:8501>. Stop it with `Ctrl+C` in the terminal. Next time you only need the
`activate` line and `python -m streamlit run app.py`.

---

## What you will see

The **sidebar** holds the preset picker, **Run optimisation**, **Reset to preset** and the sensitivity controls:
SAF contrail response, forecast information, stock, receipts, truck slots, the cash cap, detour energy, the
contrail preference weight, risk aversion (CVaR) and the held-out seed.
"Recompute automatically" is on by default. Turn it off and any edit is flagged as *results belong to the previous run* until you press Run.

| Tab | Contents |
|---|---|
| **Decision board** | Which flights get SAF, blend by mass and by volume, kg of pure SAF drawn and remaining, the route under each forecast signal, expected climate score and extra cost. Pick a flight to see its best ordinary and best SAF options, the saving per tonne of SAF, a one-sentence fleet explanation (from a forced re-solve) and a schematic of the planned, higher and lower profiles. |
| **Flights and forecasts** | The 12 historical records with evidence status (flown, airframe, engine, operator, conflicts). Editable modelling inputs, editable weather scenarios and probabilities, the signed-EF table, team-estimate CSV import, which GFS cycle each dispatcher sees, and humidity along each observed track. |
| **Compare strategies** | All five methods plus the ordinary-fuel reference on the same axes: fuel life-cycle emissions, signed contrail, total, cost, SAF kg, paired differences from a chosen benchmark, a per-scenario breakdown, held-out results and a CSV export. Has a compact pitch view. |
| **How the decision works** | Decision timeline, the two worked examples solved live, equations with live unit checks, the 60-second demo script, Evidence from Arlanda, the European outlook to 2036, sources with labels, and limits. |

### Presets

| Preset | What it is |
|---|---|
| **Sourced cohort** (default) | The 12 documented ARN departures in `data/arlanda/flight_manifest.csv`, with their ADS-B tracks and archived NOAA GFS forecasts. Rows, times and tracks are sourced. Energy, uplift, supply and contrail forcing are **modelled or assumed** and labelled as such. |
| **Illustrative Stockholm simulation** | 12 synthetic rows (`SYN-01`…`SYN-12`) across the same five airports. They include a cheap detour, an expensive detour, a cooling case, and positive, zero and negative EF. These rows are excluded from historical counts and exports. |
| **Small teaching fixture** | 4 synthetic rows for a quick classroom walkthrough. You can add or remove rows. |

---

## 60-second demo

1. *"Twelve SAS-group A320neo departures from Arlanda on 17 November 2026. We have about 6.6 t of pure HEFA for a 35%-by-mass blend, enough for only a few aircraft."*
2. **Decision board.** With the defaults the joint plan fuels **SK481 → OSL, SK531 → LHR and SK579 → CDG**. That uses 6,580 kg of pure SAF from stock and scores 139.8 t CO₂e against 153.3 t with ordinary fuel.
3. Pick **SK531** and read the sentence: why it gets the fuel, the saving per tonne of SAF, and the level it would fly under each forecast signal.
4. Sidebar: switch *What dispatch can distinguish* to **No update**, then **Scenario revealed**. Point out that the expected score moves (139.83 → 140.01 t CO₂e) but the list of SAF flights does not. Stock size dominates who gets fuel here, and that is a finding in itself.
5. **Compare strategies.** Same stock, same slots, five methods. Held-out weather shows the paired difference and how often the joint method does worse.

---

## How the numbers are made

- **Fuel.** `b = (w/ρS)/(w/ρS+(1−w)/ρJ)`, `q_SAF = U·w`, a well-mixed tank `w_onboard = (M0·w0+U·w)/(M0+U)`,
  `m_burn = E/H_mix`. At 760/800 kg/m³, 1,000 kg at 35% by mass holds 350 kg of SAF and is 36.18% by volume.
  HEFA-SPK Annex A2 (≤ 50% by volume) is screened. No batch is ever certified.
- **Climate score** (t CO₂e, QAEI 100-year): `C = L_fuel + k100·EF + L_extra`, with
  `k100 = 0.42/(5.1e14·31,536,000·8.8e-14·1000) ≈ 0.297 t per TJ`. A 100 kg fossil detour must avoid about 1.30 TJ to break even.
- **Contrail EF (sourced cohort).** A teaching surrogate, not CoCiP. Along each observed track, a waypoint counts
  as persistent-contrail if the GFS relative humidity over ice (× the scenario multiplier) is at least 100% and the
  temperature is below the Schmidt–Appleman threshold. EF per km is longwave warming minus a shortwave term
  proportional to sin(solar elevation), so it is signed. SAF changes EF by
  `response_scale × (−15%) × (w_onboard/0.35)`. Scale 0 is the zero-cruise-response LEAP sensitivity.
  Route-specific CoCiP estimates can be imported as a filled `option_scenario_inputs_template.csv`.
- **Route energy.** `trip_fuel_kg = 600 + 2.65 × track km` (an A20N proxy) × 43 MJ/kg. The lower and higher levels add a
  cruise penalty plus the energy for two 2,000 ft level changes. Short legs (< 150 km of cruise) keep only the
  planned profile, and levels above the A320neo ceiling (39,800 ft) are unavailable.
- **Optimiser.** A two-stage MILP solved with `scipy.optimize.milp` (HiGHS). Fuel `x[i,b]` and route
  `y[i,b,r,g]` are both binary. The constraints are cumulative stock with receipts, truck slots per window, the
  maximum number of special refuellings, an optional cash cap and booking-cutoff eligibility. Optional CVaR is
  included. Solver status is checked before anything is displayed.
- **Strategies.** Fixed priority, original-route targeting, route-first targeting, joint deterministic (one
  representative scenario) and joint stochastic. All five get the same stage-two replanning and limits.
- **Held-out evaluation.** Uses an independent seed. Each draw samples a hidden outcome, gives dispatch only its
  signal, and applies a mean-one lognormal EF variation shared by all methods.

### Default assumptions (all editable in the app)

| Input | Default | Basis |
|---|---|---|
| Opening dedicated stock | 4,400 kg pure SAF + 2,200 kg at 12:00 UTC | illustrative (≈ 2 + 1 trial-size truckloads) |
| Truck load / slots | 8,000 L; 2 trucks per 60-min window | Quante et al. trial truck; slots assumed |
| Fuel deadline | 16 Nov 22:00 UTC (after the 18Z GFS cycle was published) | assumption |
| Supplier booking lead | 6 h before the service start (STD − 60 min) | assumption |
| Scenarios | dry 0.40 (RHi × 0.90), moderate 0.35 (× 1.00), high 0.25 (× 1.25) | illustrative forecast-error states |
| Information | partial update: dry distinct, moderate/high pooled | brief |
| Prices | €640/t Jet A-1, €1,925/t aviation biofuel | EASA 2025 EU reference |
| Intensities | 89 g CO₂e/MJ (Cascade); 13.9 g CO₂e/MJ (ICAO UCO-HEFA, sensitivity) | published |
| Densities / LHV | 800 / 760 kg/m³; 43 / 44 MJ/kg | illustrative |

---

## Using your own data

- **Roster.** Replace `data/arlanda/flight_manifest.csv` with the team's dated CSV, using the same columns. No
  code changes are needed. If the file is missing, the app opens with "12 proposed departures · manifest pending"
  and shows the illustrative simulation.
- **Contrail and energy estimates.** Fill `data/arlanda/option_scenario_inputs_template.csv` and upload it in
  *Flights and forecasts → Import team EF/energy estimates*. Imported rows are labelled "team estimate".
- **Operator inputs** (stock, trucks, deadlines, fees, uplifts) are set in the sidebar and the editable tables.

## Project layout

```
app.py                  Streamlit UI (four tabs)
saf_app/physics.py      blend conversions, tank mixing, burn, climate score, costs, SAC screen
saf_app/model.py        Problem definition, information groups, input validation
saf_app/solver.py       two-stage MILP (HiGHS), recourse evaluation, CVaR, enumeration check
saf_app/strategies.py   five strategies + ordinary-fuel reference
saf_app/evaluation.py   held-out sampling and paired statistics
saf_app/explain.py      per-flight explanation and forced re-solves
saf_app/evidence.py     manifest evidence status, codeshare de-duplication, export blocking
saf_app/stockholm.py    Stockholm presets and Problem construction from the data pack
saf_app/examples.py     worked examples A and B
saf_app/runner.py       one full run + CSV export
data/arlanda/           bundled data pack (manifest, tracks, forecasts, GRIB2 weather, parameters)
tests/                  pytest acceptance tests (incl. a headless UI smoke test)
```

## Checks that were run

`python -m pytest -q` → **56 passed** (Python 3.12, Streamlit 1.67, SciPy 1.18). The tests cover:

- worked example numbers (12 / 10 / 13 / 16.7%; 15.2 / 13.2 / 14.2), with the solver checked against
  exhaustive enumeration on both tables and on 8 random problems
- zero SAF, zero slots, a binding cash cap, ineligible flights, invalid probabilities, infeasible routes,
  negative EF, equal weather states and ties
- the 350 kg/t unit check, the mass–volume round trip, dilution by carry-in fuel, the Annex A2 limit,
  reserve compliance and the k100 / 1.30 TJ checks
- zero contrail response tested separately from zero total benefit
- the 12 roster slots, codeshare de-duplication, scheduled vs flown status, disputed aircraft, incomplete
  manifests, blocked synthetic exports and booking-cutoff / truck-window eligibility
- taxi-statistic and Cascade-year changes leaving EF untouched, all strategies sharing the same limits,
  pooled outcomes sharing a route, the held-out policy seeing only the signal, the CSV matching the dashboard,
  and the documented fresh-start default
- a headless AppTest of every preset plus stale-result flagging

## Data notes and limits

- Track data © adsb.lol contributors, ODbL 1.0 (see `data/arlanda/tracks/ATTRIBUTION.md`). The GRIB2 files are
  NOAA GFS subsets. The app reads the derived CSV, and the GRIB files are bundled for anyone who wants to re-derive it.
- `fuel_aircraft_parameters.csv` lists `one_trial_size_truck_equivalent_neat_saf = 2,199,483.2 kg`. That looks
  like grams: 8,000 L × 785.53 kg/m³ × 0.35 ≈ **2,199.5 kg**. The app uses the recalculated value.
- The EF surrogate smooths thin ice-supersaturated layers (0.25° grid). It ignores the effect of SAF on the formation
  threshold and treats flights as independent (no rotations; carry-in M0 = 0). The scenarios are not a calibrated ensemble.
- The European outlook shows ReFuelEU minimum supply shares only. Cascade's simulated emissions series is not
  bundled, and EU shares cannot set these aircraft's fuel, price or ARN stock.

**Sources:** Quante et al. 2025 (doi:10.1016/j.jatrs.2024.100049) · Schmitt et al. 2026 (PMC12969261) · Teoh et
al. 2022 (doi:10.1021/acs.est.2c05781) · Woeldgen et al. 2025 (doi:10.1021/acs.est.5c02364) · Voigt et al. 2021 ·
Märkl et al. 2024 · Engberg et al. 2025 · ICAO SAF conversion processes and CORSIA default values · Boeing Cascade ·
EASA 2025 reference prices · Imperial QAEI tutorials.
