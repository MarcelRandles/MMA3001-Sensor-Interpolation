# Sensor Outage Duration and Temperature Interpolation Accuracy

MMA3001 individual project, Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data.

## Question
How does missing-data gap duration affect indoor temperature reconstruction using linear interpolation, natural cubic spline and PCHIP?

This is retrospective reconstruction using measurements on both sides of a gap, rather than future forecasting. No machine-learning training is required.

## Data preparation
The environmental ZIP is read directly. Sensor 6012002000869 supplies temperature in °C. The script checks units and finite numeric values, rejects conflicting values at identical timestamps, removes identical duplicates and sorts by measurement time. It preserves irregular sampling without filling gaps in advance. The cleaned series has 13,800 readings from 29 February to 14 June 2024 (Melbourne local time). Actual timestamps take precedence over the source filename's dates.

## Experiment
For each day, cycle through starts at midnight, 06:00, 12:00 and 18:00. At each start, separately hide readings in gaps of 0.5, 1, 2, 4 and 8 hours. The interval includes the start and excludes the end. Give every method the same remaining readings from six hours before the outage to six hours after it. Evaluate only at hidden measurement times, using recorded temperatures as references.

- Linear joins neighbouring known measurements with straight lines.
- Natural cubic spline uses smoothly joined cubic segments, with zero second derivative at the outer fitting-window boundaries.
- PCHIP uses piecewise cubic Hermite interpolation, preserving monotonicity between known points and avoiding overshoot of their values. It cannot recover an unseen peak reliably.

Main period: 2 March–29 April (59 starts per duration).
Later period: 2 May–12 June (42 starts per duration).

PCHIP was introduced after examining the original two-method May–June validation. Its later-period evaluation is an additional comparison, not an untouched confirmatory test. No PCHIP settings were tuned to those results.

Durations reuse measurements and neighbouring cases can share context; cases are not fully independent. Nominal outage duration is distinct from the span between the nearest available bracketing measurements.

## Outputs and metrics
`results/gap_experiment_results.csv` contains errors for each outage and method. The summary and PNG show mean outage MAE and RMSE by duration; each outage receives equal weight. These errors, and maximum absolute error, are in °C. Mean outage RMSE is not a pooled RMSE. `results/validation/` contains later-period outputs. The selected worst-case plot remains selected by largest natural-spline error, and shows all three methods on that same case.

Regenerate outputs after updating the scripts; old two-method tables do not include PCHIP. Conclusions must use the regenerated results.

## Windows setup and reproduction
Use Python 3.13 and a terminal opened in the project folder.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Place the original course ZIP, still compressed, at `data/raw/5EnvSensor_MayToDec2024_180kRows.zip`. Obtain it from the authorised course source. The occupancy dataset is not needed. Raw and processed data should be excluded from Git.

Run individually in order:

```powershell
.\.venv\Scripts\python.exe src/inspect_data.py
.\.venv\Scripts\python.exe src/compare_interpolation.py
.\.venv\Scripts\python.exe src/run_experiments.py
.\.venv\Scripts\python.exe src/plot_worst_case.py
.\.venv\Scripts\python.exe src/run_validation.py
```

Re-running replaces generated outputs. The scripts locate paths relative to their location, not the terminal's directory. Keep the supplied folder structure.

## Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v --junitxml=results/test_report.xml | Tee-Object -FilePath results/test_report.txt
```

Eight tests cover hand-calculated error metrics, exact reconstruction of constant and linear signals with all three methods, output generation, paired comparison rows, nonfinite temperatures, conflicting duplicates, and sorting/removing identical duplicates. Synthetic tests use temporary files. Passing tests verify these behaviours, not every possible input or realistic signal.

## HTML documentation
```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pdoc src/inspect_data.py src/compare_interpolation.py src/run_experiments.py src/plot_worst_case.py src/run_validation.py -o docs
```
Open `docs/run_experiments.html` in a browser. Regenerate documentation after source changes.

## Limitations
One sensor and observation period cannot establish performance across buildings or seasons. Artificial outages may differ from actual failure patterns. Recorded values have measurement uncertainty. Accuracy is evaluated only where observations exist. Both sides of each gap are needed. Neither shape preservation nor curve smoothness guarantees recovery of changes entirely hidden inside long gaps.

## Course connections and AI
The work applies interpolation, splines, error analysis, data management, version control, testing and documentation. ChatGPT assisted with experiment design, code, debugging and interpretation. The report must describe checks, limitations and the student's understanding. PCHIP is an extension to the original two-method comparison.
