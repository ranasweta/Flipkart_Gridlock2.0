# ParkSight

Parking-congestion intelligence for Bengaluru Traffic Police.

ParkSight takes the raw BTP parking-violation log (~298k records, Nov 2023 to Apr

2024) and turns it into three things a traffic planner can actually use:

1. a map of where illegal parking is choking traffic, weighted by how much each

	violation obstructs moving vehicles rather than just counting tickets,

2. a one-week-ahead forecast of where congestion will build next, and

3. a patrol schedule that places a fixed number of units where they cover the most

	obstruction.

Everything runs off the single provided CSV. No external data.

## How impact is measured

A raw violation count is misleading: a bus blocking a main road at a junction is not

the same as a scooter in a no-parking bay. Each violation gets a congestion-impact

score:

```

impact = severity(violation subtype) x bulk(vehicle type) x junction multiplier

```

The weights live in `src/[config.py](http://config.py/)` so they can be read and changed in one place.

Parking on a main road or double parking scores highest (3.0), footpath 2.0, plain

no-parking 1.2. Buses multiply by 3.0, cars 1.5, two-wheelers ~0.7. A violation logged

at a named junction gets a 1.5x multiplier. Scoring is in `src/[scoring.py](http://scoring.py/)`.

## Pipeline

The batch pipeline (`src/[pipeline.py](http://pipeline.py/)`) runs the full dataset in a few seconds and

writes small CSVs that the dashboard reads:

| Step | Module | What it does |

|------|--------|--------------|

| 1 | `data_loader.py` | Parse the JSON violation arrays, convert timestamps UTC to IST, drop out-of-Bengaluru coordinates, normalise vehicle and junction fields |

| 2 | `[scoring.py](http://scoring.py/)` | Congestion-impact score per violation |

| 3 | `[hotspots.py](http://hotspots.py/)` | Bin into H3 res-9 hexes (~street-block size); per-hex totals, severity mix, junction share, trend; hex-by-hour matrix |

| 4 | `[bias.py](http://bias.py/)` | Blind-spot index: cells with high obstruction and a rising trend but few tickets (the data is enforcement-logged, so raw counts track where patrols already go) |

| 5 | `[cascade.py](http://cascade.py/)` | Ring diffusion over the H3 neighbour graph; finds choke points whose blockage backs traffic into adjacent cells |

| 6 | `[anomaly.py](http://anomaly.py/)` | Emerging hotspots: robust median/MAD surge z-score plus an IsolationForest over recent behaviour |

| 7 | `[forecast.py](http://forecast.py/)` | One-week-ahead per-hex impact model, evaluated on a forward time split |

| 8 | `impact_eval.py` | Coverage backtest: does a forecast-driven plan beat the current footprint out-of-sample |

| 9 | `[optimizer.py](http://optimizer.py/)` | Greedy weighted maximum-coverage patrol schedule under a unit budget, with a reserve for blind spots |

## The forecasting model

`src/[forecast.py](http://forecast.py/)` resamples each hex to a weekly impact series and trains a

`GradientBoostingRegressor` to predict the next week's impact, pooled across all hexes

with at least 30 violations. Features are everything observable up to the current week:

lags 1-4, 4- and 8-week rolling mean/std, a 4-week trend slope, week-of-year

seasonality, and static hex descriptors (severity mix, junction share, size). Partial

boundary weeks are dropped so they don't distort the fit.

Evaluation is a forward time split: the last 4 weeks of targets are held out, the

model never sees them, and it is scored against two baselines on the same rows.

Held-out results (about 13.8k train / 3.4k test hex-weeks):

| Model | MAE | RMSE | R² |

|-------|-----|------|-----|

| GradientBoosting | 19.47 | 45.42 | 0.743 |

| Persistence (last week) | 22.65 | 56.25 | 0.606 |

| 4-week moving average | 19.86 | 47.47 | 0.719 |

The model beats both baselines on every metric. The most important features are total

hex size, the EWM of recent impact, and the recent lags.

## Coverage backtest

Accuracy alone does not prove the model helps operations, so `src/impact_eval.py` tests

the operational question directly: plan on the past, then measure how much of the

held-out future congestion-impact each allocation of the same patrol budget actually

sits on.

At a 40-cell budget (10 units x 4 shifts):

| Strategy | Held-out impact covered |

|----------|-------------------------|

| Status quo (top cells by ticket count) | 41.7% |

| Impact-weighted, static | 44.0% |

| Forecast-driven | 44.5% |

| Oracle (perfect foresight) | 46.4% |

For the same number of units, the forecast-driven plan covers 2.8 points more real

obstruction than the current ticket-count footprint, which is about 60% of the gap to a

perfect-foresight oracle. The oracle topping out near 46% at 40 cells shows impact is

spread out; the gain widens as the budget grows.

## Running it

```bash

pip install -r requirements.txt        # pandas, numpy, h3, scikit-learn, streamlit, pydeck

python -m src.pipeline                  # build artifacts from the full CSV

python -m src.pipeline 50000            # quick subsample while iterating

streamlit run app/[dashboard.py](http://dashboard.py/)          # open the dashboard

```

The pipeline writes to `data/processed/`. The dashboard reads those files, so it loads

instantly and recomputes the schedule live when you change the unit slider.

## Dashboard

`app/[dashboard.py](http://dashboard.py/)` (Streamlit + pydeck) has five map lenses (congestion, blind spots,

network choke points, next-week forecast, emerging hotspots), a forecast panel with the

accuracy table, feature importances, a predicted-vs-actual chart and the coverage

curve, an emerging-hotspot board, the hotspot/blind-spot/choke-point tables, and the

downloadable patrol schedule.

## Layout

```

parksight/

 src/

	 [config.py](http://config.py/)        weights, paths, model and cascade parameters

	 data_loader.py   load + clean

	 [scoring.py](http://scoring.py/)       congestion-impact score

	 [hotspots.py](http://hotspots.py/)      H3 aggregation + temporal profiles

	 [bias.py](http://bias.py/)          blind-spot index

	 [cascade.py](http://cascade.py/)       network choke points

	 [anomaly.py](http://anomaly.py/)       emerging-hotspot detection

	 [forecast.py](http://forecast.py/)      weekly forecasting model + evaluation

	 impact_eval.py   coverage backtest

	 [optimizer.py](http://optimizer.py/)     patrol schedule

	 [pipeline.py](http://pipeline.py/)      runs the whole thing

 app/

	 [dashboard.py](http://dashboard.py/)     Streamlit UI

 data/processed/    generated artifacts

 requirements.txt

```

## Notes on the data

Timestamps in the source are UTC and converted to IST. They record when a ticket was

logged, not when an obstruction started, which is exactly why the blind-spot index

exists as a correction for patrol bias. The `closed_datetime` column is empty in the

source, so the project makes no response-time claims.

Insert it in my readme but just `main.py` is mix of all files so keep that only rest the content should be intact
