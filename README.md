# Sensor Outage Duration and Temperature Interpolation Accuracy

MMA3001 individual project, Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data.

## Question
How does missing-data gap duration affect indoor temperature reconstruction using linear interpolation, natural cubic spline, PCHIP and local cubic Newton interpolation?

This is retrospective reconstruction using measurements on both sides of a gap, rather than future forecasting. No machine-learning training is required.

## Data preparation
The environmental ZIP is read directly. Sensor 6012002000869 supplies temperature in °C. The script checks units and finite numeric values, rejects conflicting values at identical timestamps, removes identical duplicates and sorts by measurement time. It preserves irregular sampling without filling gaps in advance. The cleaned series has 13,800 readings from 29 February to 14 June 2024 (Melbourne local time). Actual timestamps take precedence over the source filename's dates.

## Experiment
For each day, cycle through starts at midnight, 06:00, 12:00 and 18:00. At each start, separately hide readings in gaps of 1 through 48 hours in one-hour increments. The interval includes the start and excludes the end. Make the same observations available from six hours before the outage to six hours after it. Linear, natural spline and PCHIP receive that full context; Newton selects the two nearest known readings on each side. Cases without two available readings on each side are excluded from every method. Evaluate only at hidden measurement times, using recorded temperatures as references.

- Linear joins neighbouring known measurements with straight lines.
- Natural cubic spline uses smoothly joined cubic segments, with zero second derivative at the outer fitting-window boundaries.
- Local cubic Newton constructs a polynomial of degree at most three from the two nearest known readings before and after the gap. It uses divided differences and nested evaluation, with shifted/scaled time coordinates.
- PCHIP uses piecewise cubic Hermite interpolation, preserving monotonicity between known points and avoiding overshoot of their values. It cannot recover an unseen peak reliably.

- Main period: 2 March–27 April 2024 (57 starts per duration).
- Later period: 2 May–10 June 2024 (40 starts per duration).

The current runs have no skipped cases: 2,736 outage cases in the main period and 1,920 in the later period, with four method results per case. The dates allow the longest outage and its surrounding context to fit within each observation period.

PCHIP, Newton and the expanded duration range were introduced after inspecting earlier results, including May–June. The final later-period comparison is exploratory, not an untouched confirmatory test. The local Newton degree and four-point selection are fixed across all cases.

Durations reuse measurements and neighbouring cases can share context; cases are not fully independent. Nominal outage duration is distinct from the span between the nearest available bracketing measurements.

## Outputs and metrics
`results/gap_experiment_results.csv` contains errors for each outage and method. The summary and PNG show mean outage MAE and RMSE by duration; each outage receives equal weight. These errors, and maximum absolute error, are in °C. Mean outage RMSE is not a pooled RMSE. `results/validation/` contains later-period outputs. The diagnostic script selects the largest individual absolute error separately for the natural spline and Newton in each period, plots all four methods on those cases, and checks their scores against the saved experiment. It also compares Newton with an independent barycentric evaluation of the same four-point polynomial. These selected extremes are not representative average cases. Outputs are saved in `results/diagnostics/` and `results/validation/diagnostics/`.

Regenerate outputs after source changes. Old root-level `worst_case_comparison.png` and `worst_case_predictions.csv`, if present, are historical outputs; use the new diagnostics folders for the final study.

## Selected results

Mean outage MAE in °C for the later period (May–June):

| Outage hours | Linear | PCHIP | Local Newton | Natural spline |
|---|---:|---:|---:|---:|
| 1 | 0.081 | 0.091 | 0.146 | 0.154 |
| 8 | 0.821 | 0.785 | 0.842 | 1.031 |
| 12 | 0.829 | 0.789 | 1.178 | 1.457 |
| 24 | 1.190 | 1.194 | 2.055 | 2.558 |
| 48 | 1.204 | 1.213 | 3.404 | 4.513 |

Linear and PCHIP had the lowest mean MAE at all tested durations across the two periods, with the winner depending on duration and error metric. Their errors flatten around 1–1.2°C for long outages in these data; this does not establish a general error bound. Natural spline and Newton can produce large excursions inside long gaps. In the selected May–June 43-hour outage, recorded temperatures ranged from 19.8°C to 24.3°C, while the spline fell to about −11.3°C and Newton to −9.7°C. Keep these failures in the results.

## Runtime benchmark

`src/benchmark_methods.py` measures construction and prediction for a fixed eight-hour case (67 available readings, 44 queries), using seven batches of 200 calls. It excludes file loading, window selection, plotting and scoring. Newton's timed function includes its validation and four-point selection. Timings describe these implementations and this case, not equal-sized mathematical workloads or every duration.

The recorded Windows run had median milliseconds per outage: linear 0.000895, Newton 0.02994, natural spline 0.08177 and PCHIP 0.09243. Use `results/method_runtime.csv` and `results/benchmark_environment.txt` for the current run. Numerical accuracy and long-gap robustness matter more here than these small absolute runtime differences.

Optional full-script profiling (includes imports and plotting, with profiling overhead):

```powershell
.\.venv\Scripts\python.exe -m cProfile -o results/experiment_profile.prof src/run_experiments.py
.\.venv\Scripts\python.exe -c "import pstats; pstats.Stats('results/experiment_profile.prof').strip_dirs().sort_stats('cumulative').print_stats(25)" | Tee-Object -FilePath results/experiment_profile.txt
```

## Introductory demonstration

`compare_interpolation.py` is the earlier three-method two-hour illustration from 4 March. It has a full-day context window, so its scores should not be confused with the repeated experiment's six-hour context settings. The final four-method conclusions come from `run_experiments.py` and `run_validation.py`.

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
.\.venv\Scripts\python.exe src/run_validation.py
.\.venv\Scripts\python.exe src/plot_worst_case.py
.\.venv\Scripts\python.exe src/benchmark_methods.py
```

Re-running replaces generated outputs. The scripts locate paths relative to their location, not the terminal's directory. Keep the supplied folder structure.

## Tests
```powershell
.\.venv\Scripts\python.exe -m pytest tests -v --junitxml=results/test_report.xml | Tee-Object -FilePath results/test_report.txt
```

Sixteen tests cover hand-calculated error metrics, exact reconstruction of constant and linear signals in the four-method experiment, output generation, paired comparison rows, nonfinite temperatures, conflicting duplicates, sorting/removing identical duplicates, Newton reconstruction of polynomials up to degree three, nearest-point selection, and invalid Newton inputs. Synthetic tests use temporary files. Passing tests verify these behaviours, not every possible input or realistic signal.

## HTML documentation
```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m pdoc src/inspect_data.py src/compare_interpolation.py src/run_experiments.py src/plot_worst_case.py src/run_validation.py src/newton_interpolation.py src/benchmark_methods.py -o docs
```
Open `docs/run_experiments.html` in a browser. Regenerate documentation after source changes.

## Limitations
One sensor and observation period cannot establish performance across buildings or seasons. Artificial outages may differ from actual failure patterns. Recorded values have measurement uncertainty. Accuracy is evaluated only where observations exist. Both sides of each gap are needed. Neither shape preservation nor curve smoothness guarantees recovery of changes entirely hidden inside long gaps.

## Course connections and AI
The work applies interpolation, splines, error analysis, data management, version control, testing and documentation. ChatGPT assisted with experiment design, code, debugging and interpretation. The report must describe checks, limitations and the student's understanding. PCHIP and local Newton are extensions to the original two-method comparison. Newton polynomial interpolation connects with the Week 3 topic; the exact course notebook should be consulted alongside the implementation.

## Licence

Project source code is provided under the MIT licence; see `LICENSE`.
The sensor dataset and course materials are not covered by that licence; their original ownership and usage conditions continue to apply.
