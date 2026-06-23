"""ParkSight - single-file parking-congestion intelligence for BTP.

Everything (data cleaning, scoring, H3 aggregation, blind-spot index, network
cascade, emerging-hotspot detection, the forecasting model, the coverage backtest,
the patrol optimizer, and the Streamlit dashboard) lives in this one file so the whole
project can be shared as a single script plus the dataset CSV.

Two ways to run
---------------
    streamlit run parksight.py            # interactive dashboard (computes in-memory)
    streamlit run parksight.py -- data.csv   # ...with an explicit CSV path

    python parksight.py                   # headless: run analysis, print summary, write CSVs
    python parksight.py data.csv          # ...with an explicit CSV path
    python parksight.py data.csv 50000    # ...on a 50k-row subsample while iterating

The script looks for the violation CSV in its own directory if no path is given
(preferring a filename containing "violation"). Output CSVs (headless mode) are
written to ./processed/ next to the script.

Dependencies: pandas, numpy, h3, scikit-learn (optional - falls back without it),
and for the dashboard streamlit + pydeck.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

# h3 v4 renamed the API; support both so the code runs on whatever pip resolved.
import h3

if hasattr(h3, "latlng_to_cell"):          # v4
    _to_cell = h3.latlng_to_cell
    _to_latlng = h3.cell_to_latlng
else:                                        # v3
    _to_cell = h3.geo_to_h3
    _to_latlng = h3.h3_to_geo

if hasattr(h3, "grid_ring"):                 # v4
    _ring = h3.grid_ring
else:                                        # v3 fallback
    _ring = h3.k_ring_smoothed

# scikit-learn is optional; the forecaster/anomaly detector degrade gracefully without it
try:
    from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    _HAVE_SK = True
except Exception:                            # pragma: no cover
    _HAVE_SK = False


# ============================================================================
#  CONFIG  - all tunable knobs in one place
# ============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed")

# geo / time -----------------------------------------------------------------
BLR_LAT = (12.7, 13.25)            # Bengaluru bounding box - drop bad coordinates
BLR_LON = (77.35, 77.85)
IST_OFFSET_HOURS = 5.5             # source timestamps are UTC (+00); BTP plans in IST
H3_RES = 9                         # ~0.10 sq km hexes ~ street-block granularity

# severity model -------------------------------------------------------------
# How much a parked vehicle of this violation subtype obstructs moving traffic.
# Matched as case-insensitive substrings against the violation_type text.
SEVERITY_WEIGHTS = {
    "parking in a main road": 3.0,
    "double parking": 3.0,
    "parking near road crossing": 2.5,
    "parking near traffic light or zebra": 2.5,
    "parking near bustop/school/hospital": 2.2,
    "parking on footpath": 2.0,
    "parking other than bus stop": 1.8,
    "parking opposite to another parked": 1.8,
    "wrong parking": 1.5,
    "no parking": 1.2,
}
DEFAULT_SEVERITY = 1.0

# Lane-width a stationary vehicle steals - bulk multiplier.
VEHICLE_BULK = {
    "BUS (BMTC/KSRTC)": 3.0, "PRIVATE BUS": 3.0, "TEMPO": 2.2, "LGV": 2.2,
    "MAXI-CAB": 1.8, "VAN": 1.8, "GOODS AUTO": 1.6, "CAR": 1.5,
    "PASSENGER AUTO": 1.2, "MOTOR CYCLE": 0.7, "SCOOTER": 0.7, "MOPED": 0.6,
}
DEFAULT_BULK = 1.0

JUNCTION_MULTIPLIER = 1.5           # a violation at a named junction blocks a bigger node
NON_JUNCTION_TOKENS = {"", "no junction", "null", "none"}

# cascade / network model ----------------------------------------------------
CASCADE_DEPTH = 2                   # hex hops a blockage propagates (~350m at res 9)
CASCADE_DECAY = 0.38                # attenuation per hop
CASCADE_MIN_VIOLATIONS = 30         # floor below which a hex isn't an actionable target

# forecasting model ----------------------------------------------------------
FORECAST_MIN_VIOLATIONS = 30
FORECAST_HOLDOUT_WEEKS = 4          # final weeks held out for honest evaluation
FORECAST_N_ESTIMATORS = 300
FORECAST_MAX_DEPTH = 3
FORECAST_LR = 0.05
RANDOM_SEED = 42

# anomaly / emerging ---------------------------------------------------------
ANOMALY_RECENT_WEEKS = 4
ANOMALY_Z_THRESHOLD = 2.0
ANOMALY_MIN_RECENT_IMPACT = 15.0
ANOMALY_CONTAMINATION = 0.08

# enforcement plan -----------------------------------------------------------
SHIFTS = {
    "Morning (06-12)": range(6, 12),
    "Afternoon (12-18)": range(12, 18),
    "Evening (18-24)": range(18, 24),
    "Night (00-06)": list(range(0, 6)),
}
DEFAULT_UNITS_PER_SHIFT = 10


def find_dataset(explicit: str | None = None) -> str:
    """Locate the violation CSV: explicit arg > env var > any .csv on argv >
    a same-directory CSV (preferring one whose name contains 'violation')."""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("PARKSIGHT_CSV"):
        candidates.append(os.environ["PARKSIGHT_CSV"])
    for a in sys.argv[1:]:
        if a.lower().endswith(".csv"):
            candidates.append(a)
    same_dir = sorted(glob.glob(os.path.join(SCRIPT_DIR, "*.csv")))
    viol = [c for c in same_dir if "violation" in os.path.basename(c).lower()]
    candidates += viol + same_dir
    for c in candidates:
        if c and os.path.exists(c):
            return c
    raise FileNotFoundError(
        "No violation CSV found. Put the dataset next to this script, pass it as an "
        "argument (python parksight.py path/to.csv), or set PARKSIGHT_CSV."
    )


# ============================================================================
#  DATA LOADER  - load + clean
# ============================================================================
def _parse_list(val: str) -> list[str]:
    """`["WRONG PARKING","NO PARKING"]` -> ['WRONG PARKING', 'NO PARKING']."""
    if not isinstance(val, str) or not val.strip():
        return []
    s = val.strip()
    if s.startswith("["):
        try:
            return [str(x).strip() for x in json.loads(s)]
        except Exception:
            pass
    return [s]


def _primary_violation(types: list[str]) -> str:
    """The single most-severe subtype on a ticket drives its severity weight."""
    best, best_w = "", -1.0
    for t in types:
        tl = t.lower()
        w = DEFAULT_SEVERITY
        for key, wt in SEVERITY_WEIGHTS.items():
            if key in tl:
                w = wt
                break
        if w > best_w:
            best, best_w = t, w
    return best or (types[0] if types else "UNKNOWN")


def load_violations(path: str, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=[
            "id", "latitude", "longitude", "vehicle_type", "violation_type",
            "created_datetime", "police_station", "junction_name",
        ],
        nrows=nrows,
        dtype=str,
    )

    # coordinates
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    in_blr = df["latitude"].between(*BLR_LAT) & df["longitude"].between(*BLR_LON)
    df = df[in_blr].copy()

    # timestamps (UTC -> IST)
    ts = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
    ts = ts.dt.tz_localize(None) + timedelta(hours=IST_OFFSET_HOURS)
    df["ts_ist"] = ts
    df = df[df["ts_ist"].notna()].copy()
    df["hour"] = df["ts_ist"].dt.hour
    df["dow"] = df["ts_ist"].dt.dayofweek
    df["date"] = df["ts_ist"].dt.date

    # violations
    types = df["violation_type"].apply(_parse_list)
    df["primary_violation"] = types.apply(_primary_violation)

    # categoricals
    df["vehicle_type"] = df["vehicle_type"].fillna("UNKNOWN").str.strip().str.upper()
    jn = df["junction_name"].fillna("").str.strip()
    df["at_junction"] = ~jn.str.lower().isin(NON_JUNCTION_TOKENS)
    df["junction_name"] = jn
    df["police_station"] = df["police_station"].fillna("UNKNOWN").str.strip()

    return df.reset_index(drop=True)


# ============================================================================
#  SCORING  - congestion-impact score
# ============================================================================
def _severity(primary_violation: str) -> float:
    tl = str(primary_violation).lower()
    for key, wt in SEVERITY_WEIGHTS.items():
        if key in tl:
            return wt
    return DEFAULT_SEVERITY


def add_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """impact = severity(subtype) x bulk(vehicle) x junction_multiplier."""
    df = df.copy()
    df["severity_w"] = df["primary_violation"].apply(_severity)
    df["bulk_w"] = df["vehicle_type"].map(VEHICLE_BULK).fillna(DEFAULT_BULK)
    df["junction_w"] = df["at_junction"].map({True: JUNCTION_MULTIPLIER, False: 1.0})
    df["impact"] = df["severity_w"] * df["bulk_w"] * df["junction_w"]
    return df


# ============================================================================
#  HOTSPOTS  - H3 aggregation + temporal profiles
# ============================================================================
def assign_hex(df: pd.DataFrame, res: int = H3_RES) -> pd.DataFrame:
    df = df.copy()
    df["hex"] = [
        _to_cell(lat, lon, res)
        for lat, lon in zip(df["latitude"].values, df["longitude"].values)
    ]
    return df


def _trend_ratio(dates: pd.Series) -> float:
    """Recent-half vs earlier-half daily-rate ratio. >1 means rising activity."""
    d = pd.to_datetime(pd.Series(dates))
    if d.empty:
        return 1.0
    span_start, span_end = d.min(), d.max()
    mid = span_start + (span_end - span_start) / 2
    early = (d <= mid).sum()
    late = (d > mid).sum()
    return float((late + 1) / (early + 1))     # +1 smoothing


def build_hex_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hex_id, sub in df.groupby("hex"):
        lat, lon = _to_latlng(hex_id)
        total_impact = float(sub["impact"].sum())
        high_sev = float((sub["severity_w"] >= 2.0).mean())
        rows.append({
            "hex": hex_id, "lat": lat, "lon": lon,
            "violations": int(len(sub)),
            "impact": total_impact,
            "impact_per_violation": total_impact / len(sub),
            "high_severity_share": high_sev,
            "junction_share": float(sub["at_junction"].mean()),
            "top_violation": sub["primary_violation"].mode().iat[0],
            "top_vehicle": sub["vehicle_type"].mode().iat[0],
            "police_station": sub["police_station"].mode().iat[0],
            "trend_ratio": _trend_ratio(sub["date"]),
        })
    hex_df = pd.DataFrame(rows).sort_values("impact", ascending=False)
    hex_df["impact_rank"] = range(1, len(hex_df) + 1)
    mx = hex_df["impact"].max()
    hex_df["impact_score"] = (100 * hex_df["impact"] / mx).round(1) if mx else 0.0
    return hex_df.reset_index(drop=True)


def build_hex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """hex x hour impact matrix (long form)."""
    return df.groupby(["hex", "hour"])["impact"].sum().reset_index()


# ============================================================================
#  BIAS  - under-enforced blind-spot index
# ============================================================================
def _pct_rank(s: pd.Series) -> pd.Series:
    if s.nunique() <= 1:
        return pd.Series(0.5, index=s.index)
    return s.rank(pct=True)


def add_blindspot_score(hex_df: pd.DataFrame, min_violations: int = 20) -> pd.DataFrame:
    """blindspot = high_severity_share x trend(rising) x (1 - enforcement_volume)."""
    df = hex_df.copy()
    severity_factor = _pct_rank(df["high_severity_share"])
    rising = (df["trend_ratio"] - 1.0).clip(lower=0)
    trend_factor = _pct_rank(rising)
    under_enforced = 1.0 - _pct_rank(df["violations"])
    df["blindspot_score"] = (
        100 * severity_factor * (0.5 + 0.5 * trend_factor) * under_enforced
    ).round(1)
    df.loc[df["violations"] < min_violations, "blindspot_score"] = 0.0
    df["is_blindspot"] = df["blindspot_score"] >= df["blindspot_score"].quantile(0.95)
    return df


# ============================================================================
#  CASCADE  - network choke points via H3 ring diffusion
# ============================================================================
def add_cascade(hex_df: pd.DataFrame,
                depth: int = CASCADE_DEPTH,
                decay: float = CASCADE_DECAY,
                min_violations: int = CASCADE_MIN_VIOLATIONS) -> pd.DataFrame:
    """cascade_impact(h) = sum_k decay^k * sum_{ring-k neighbours} impact.

    network_leverage = cascade_impact / local_impact. High leverage = small local
    problem, large network consequence - the highest-ROI enforcement targets.
    """
    impact_map = dict(zip(hex_df["hex"], hex_df["impact"].astype(float)))

    cascade = {}
    for hex_id in hex_df["hex"]:
        total = 0.0
        for k in range(depth + 1):
            ring = {hex_id} if k == 0 else _ring(hex_id, k)
            total += (decay ** k) * sum(impact_map.get(h, 0.0) for h in ring)
        cascade[hex_id] = total

    df = hex_df.copy()
    df["cascade_impact"] = df["hex"].map(cascade)

    # leverage only meaningful where local impact is non-trivial; floor at min_violations
    local_safe = df["impact"].where(df["violations"] >= min_violations)
    df["network_leverage"] = (df["cascade_impact"] / local_safe).fillna(1.0).round(2)
    df["cascade_rank"] = df["cascade_impact"].rank(ascending=False, method="first").astype(int)
    df["rank_jump"] = (df["impact_rank"] - df["cascade_rank"]).where(
        df["violations"] >= min_violations, other=0
    )
    mx = df["cascade_impact"].max()
    df["cascade_score"] = (100 * df["cascade_impact"] / mx).round(1) if mx else 0.0
    return df


# ============================================================================
#  ANOMALY  - emerging-hotspot / surge detection
# ============================================================================
def add_anomaly_scores(df: pd.DataFrame,
                       hex_static: pd.DataFrame,
                       recent_weeks: int = ANOMALY_RECENT_WEEKS,
                       min_recent_impact: float = ANOMALY_MIN_RECENT_IMPACT) -> pd.DataFrame:
    """Robust median/MAD surge z-score + IsolationForest over recent behaviour."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time
    # drop partial boundary weeks so the recent window isn't understated
    dmin, dmax = d["date"].min(), d["date"].max()
    full = d[(d["week"] >= dmin) & (d["week"] + pd.Timedelta(days=6) <= dmax)]
    d = full if not full.empty else d
    wk = d.groupby(["hex", "week"])["impact"].sum().reset_index()

    weeks = np.sort(wk["week"].unique())
    if len(weeks) < recent_weeks + 2:
        out = hex_static.copy()
        for c, v in {"surge_z": 0.0, "recent_week_impact": 0.0, "baseline_impact": 0.0,
                     "anomaly_score": 0.0, "iso_outlier": False, "is_emerging": False}.items():
            out[c] = v
        return out

    recent_set = set(weeks[-recent_weeks:])
    rows = []
    for hex_id, sub in wk.groupby("hex"):
        s = sub.set_index("week")["impact"].reindex(weeks, fill_value=0.0)
        recent = s[s.index.isin(recent_set)]
        hist = s[~s.index.isin(recent_set)]
        recent_mean = float(recent.mean())
        if len(hist) >= 2:
            med = float(np.median(hist))
            mad = float(np.median(np.abs(hist - med)))
            z = (recent_mean - med) / (1.4826 * mad + 1e-6)
        else:
            med, z = recent_mean, 0.0
        slope = float(np.polyfit(np.arange(len(s)), s.values, 1)[0]) if len(s) >= 2 else 0.0
        recent_share = float(recent.sum() / s.sum()) if s.sum() > 0 else 0.0
        rows.append({"hex": hex_id, "surge_z": z, "recent_week_impact": recent_mean,
                     "baseline_impact": med, "trend_slope": slope, "recent_share": recent_share})

    a = pd.DataFrame(rows)

    if _HAVE_SK and len(a) >= 20:
        feats = a[["recent_week_impact", "surge_z", "trend_slope", "recent_share"]].fillna(0.0).values
        iso = IsolationForest(n_estimators=200, contamination=ANOMALY_CONTAMINATION,
                              random_state=RANDOM_SEED)
        iso.fit(feats)
        raw = -iso.score_samples(feats)        # higher = more anomalous
        a["iso_score"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        a["iso_outlier"] = iso.predict(feats) == -1
    else:
        a["iso_score"] = 0.0
        a["iso_outlier"] = False

    z01 = (a["surge_z"].clip(lower=0) / 3.0).clip(0, 1)
    a["anomaly_score"] = (100 * (0.6 * z01 + 0.4 * a["iso_score"])).round(1)
    a["is_emerging"] = (
        (a["surge_z"] >= ANOMALY_Z_THRESHOLD)
        & (a["recent_week_impact"] >= min_recent_impact)
        & (a["trend_slope"] > 0)
    )

    out = hex_static.merge(
        a[["hex", "surge_z", "recent_week_impact", "baseline_impact",
           "anomaly_score", "iso_outlier", "is_emerging"]],
        on="hex", how="left",
    )
    fill = {"surge_z": 0.0, "recent_week_impact": 0.0, "baseline_impact": 0.0,
            "anomaly_score": 0.0, "iso_outlier": False, "is_emerging": False}
    return out.fillna(fill)


# ============================================================================
#  FORECAST  - one-week-ahead per-hex impact model + forward time-split eval
# ============================================================================
_FEATURES = [
    "lag1", "lag2", "lag3", "lag4",
    "roll4_mean", "roll4_std", "roll8_mean",
    "trend_slope", "weeks_active", "ewm",
    "high_severity_share", "junction_share", "log_total", "month_sin", "month_cos",
]


def build_weekly_panel(df: pd.DataFrame, min_violations: int) -> pd.DataFrame:
    """hex x ISO-week dense impact panel (zero-filled) for hexes with signal."""
    counts = df.groupby("hex").size()
    keep = counts[counts >= min_violations].index
    sub = df[df["hex"].isin(keep)].copy()

    sub["date"] = pd.to_datetime(sub["date"])
    sub["week"] = sub["date"].dt.to_period("W-SUN").dt.start_time

    # drop partial boundary weeks (would poison both training and eval)
    dmin, dmax = sub["date"].min(), sub["date"].max()
    full = sub[(sub["week"] >= dmin) & (sub["week"] + pd.Timedelta(days=6) <= dmax)]
    sub = full if not full.empty else sub

    wk = (sub.groupby(["hex", "week"])
          .agg(impact=("impact", "sum"), violations=("impact", "size")).reset_index())

    weeks = pd.date_range(wk["week"].min(), wk["week"].max(), freq="W-MON")
    hexes = wk["hex"].unique()
    grid = pd.MultiIndex.from_product([hexes, weeks], names=["hex", "week"]).to_frame(index=False)
    panel = grid.merge(wk, on=["hex", "week"], how="left")
    panel[["impact", "violations"]] = panel[["impact", "violations"]].fillna(0.0)
    return panel.sort_values(["hex", "week"]).reset_index(drop=True)


def _slope_past(s: pd.Series) -> pd.Series:
    """Least-squares slope over the trailing 4 weeks strictly before each row."""
    vals = s.values
    out = np.full(len(s), np.nan)
    for i in range(len(s)):
        y = vals[max(0, i - 4):i]
        if len(y) >= 2:
            out[i] = np.polyfit(np.arange(len(y)), y, 1)[0]
    return pd.Series(out, index=s.index)


def _add_features(panel: pd.DataFrame, hex_static: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    g = df.groupby("hex")["impact"]
    for k in (1, 2, 3, 4):
        df[f"lag{k}"] = g.shift(k)

    shifted = g.shift(1)                        # past-only values
    sg = shifted.groupby(df["hex"])
    df["roll4_mean"] = sg.transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["roll4_std"] = sg.transform(lambda s: s.rolling(4, min_periods=1).std())
    df["roll8_mean"] = sg.transform(lambda s: s.rolling(8, min_periods=1).mean())
    df["ewm"] = sg.transform(lambda s: s.ewm(span=4, min_periods=1).mean())
    df["trend_slope"] = g.transform(_slope_past)
    df["weeks_active"] = shifted.gt(0).groupby(df["hex"]).cumsum()

    woy = df["week"].dt.isocalendar().week.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * woy / 52.0)
    df["month_cos"] = np.cos(2 * np.pi * woy / 52.0)

    stat = hex_static.set_index("hex")
    df["high_severity_share"] = df["hex"].map(stat["high_severity_share"]).fillna(0.0)
    df["junction_share"] = df["hex"].map(stat["junction_share"]).fillna(0.0)
    df["log_total"] = np.log1p(df["hex"].map(stat["violations"]).fillna(0.0))

    df[_FEATURES] = df[_FEATURES].fillna(0.0)
    return df


def run_forecast(df: pd.DataFrame,
                 hex_static: pd.DataFrame,
                 min_violations: int = FORECAST_MIN_VIOLATIONS,
                 holdout_weeks: int = FORECAST_HOLDOUT_WEEKS):
    """Returns (forecast_hex_df, eval_curve_df, metrics_dict, holdout_df)."""
    panel = build_weekly_panel(df, min_violations)
    feat = _add_features(panel, hex_static)
    feat["target"] = feat.groupby("hex")["impact"].shift(-1)
    feat["target_week"] = feat.groupby("hex")["week"].shift(-1)
    sup = feat.dropna(subset=["target"]).copy()

    weeks_sorted = np.sort(sup["target_week"].unique())
    if len(weeks_sorted) <= holdout_weeks + 2 or not _HAVE_SK:
        return _fallback_forecast(feat, hex_static)

    cutoff = weeks_sorted[-holdout_weeks]
    train = sup[sup["target_week"] < cutoff]
    test = sup[sup["target_week"] >= cutoff]

    model = GradientBoostingRegressor(
        n_estimators=FORECAST_N_ESTIMATORS, max_depth=FORECAST_MAX_DEPTH,
        learning_rate=FORECAST_LR, subsample=0.8, random_state=RANDOM_SEED)
    model.fit(train[_FEATURES].values, train["target"].values)
    pred = np.clip(model.predict(test[_FEATURES].values), 0, None)

    y_te = test["target"].values
    base_persist = np.clip(test["lag1"].values, 0, None)
    base_mean = np.clip(test["roll4_mean"].values, 0, None)

    def _metrics(y, p):
        return {"mae": float(mean_absolute_error(y, p)),
                "rmse": float(np.sqrt(mean_squared_error(y, p))),
                "r2": float(r2_score(y, p))}

    m_model, m_persist, m_mean = _metrics(y_te, pred), _metrics(y_te, base_persist), _metrics(y_te, base_mean)
    lift_p = (m_persist["mae"] - m_model["mae"]) / m_persist["mae"] * 100 if m_persist["mae"] else 0.0
    lift_m = (m_mean["mae"] - m_model["mae"]) / m_mean["mae"] * 100 if m_mean["mae"] else 0.0
    importances = sorted(zip(_FEATURES, model.feature_importances_), key=lambda t: -t[1])

    test = test.assign(pred=pred)
    curve = (test.groupby("target_week")
             .agg(actual=("target", "sum"), predicted=("pred", "sum"))
             .reset_index().rename(columns={"target_week": "week"}))

    # forward forecast: retrain on all data, predict the next unseen week
    model_full = GradientBoostingRegressor(
        n_estimators=FORECAST_N_ESTIMATORS, max_depth=FORECAST_MAX_DEPTH,
        learning_rate=FORECAST_LR, subsample=0.8, random_state=RANDOM_SEED)
    model_full.fit(sup[_FEATURES].values, sup["target"].values)

    last_week = feat["week"].max()
    latest = feat[feat["week"] == last_week].copy()
    latest["pred_next_impact"] = np.clip(model_full.predict(latest[_FEATURES].values), 0, None)

    stat = hex_static.set_index("hex")
    fc = latest[["hex", "impact", "pred_next_impact"]].rename(columns={"impact": "last_week_impact"})
    fc["lat"] = fc["hex"].map(stat["lat"])
    fc["lon"] = fc["hex"].map(stat["lon"])
    fc["police_station"] = fc["hex"].map(stat["police_station"])
    fc["top_violation"] = fc["hex"].map(stat["top_violation"])
    fc["forecast_delta"] = (fc["pred_next_impact"] - fc["last_week_impact"]).round(1)
    fc["forecast_change_pct"] = (
        100 * fc["forecast_delta"] / fc["last_week_impact"].replace(0, np.nan)
    ).fillna(0.0).round(0)
    mxp = fc["pred_next_impact"].max() or 1.0
    fc["forecast_score"] = (100 * fc["pred_next_impact"] / mxp).round(1)
    fc = fc.sort_values("pred_next_impact", ascending=False).reset_index(drop=True)

    metrics = {
        "model": "GradientBoostingRegressor", "horizon": "1 week ahead",
        "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
        "holdout_weeks": int(holdout_weeks), "forecast_hexes": int(len(fc)),
        "forecast_week_start": str(pd.to_datetime(last_week).date()),
        "metrics_model": m_model, "metrics_persistence": m_persist, "metrics_mean": m_mean,
        "mae_lift_vs_persistence_pct": round(lift_p, 1), "mae_lift_vs_mean_pct": round(lift_m, 1),
        "top_features": [{"feature": f, "importance": round(float(i), 3)} for f, i in importances[:8]],
        "fallback": False,
    }
    holdout = test[["hex", "target_week", "target", "pred", "lag1", "roll4_mean"]].rename(
        columns={"target_week": "week", "target": "actual"})
    return fc, curve, metrics, holdout


def _fallback_forecast(feat, hex_static):
    """Persistence forecast when sklearn is unavailable or history is too short."""
    last_week = feat["week"].max()
    latest = feat[feat["week"] == last_week].copy()
    latest["pred_next_impact"] = latest["roll4_mean"].clip(lower=0)
    stat = hex_static.set_index("hex")
    fc = latest[["hex", "impact", "pred_next_impact"]].rename(columns={"impact": "last_week_impact"})
    for col in ("lat", "lon", "police_station", "top_violation"):
        fc[col] = fc["hex"].map(stat[col])
    fc["forecast_delta"] = (fc["pred_next_impact"] - fc["last_week_impact"]).round(1)
    fc["forecast_change_pct"] = 0.0
    mxp = fc["pred_next_impact"].max() or 1.0
    fc["forecast_score"] = (100 * fc["pred_next_impact"] / mxp).round(1)
    fc = fc.sort_values("pred_next_impact", ascending=False).reset_index(drop=True)
    curve = pd.DataFrame(columns=["week", "actual", "predicted"])
    holdout = pd.DataFrame(columns=["hex", "week", "actual", "pred", "lag1", "roll4_mean"])
    metrics = {"model": "persistence (fallback)", "fallback": True,
               "reason": "scikit-learn unavailable or insufficient weekly history"}
    return fc, curve, metrics, holdout


# ============================================================================
#  IMPACT EVAL  - out-of-sample coverage backtest
# ============================================================================
def _coverage(ranked_hexes, eval_impact, k):
    chosen = ranked_hexes[:k]
    total = eval_impact.sum()
    return float(eval_impact.reindex(chosen).fillna(0.0).sum() / total) if total > 0 else 0.0


def evaluate_coverage(df: pd.DataFrame, holdout: pd.DataFrame, plan_size: int,
                      budgets: list[int] | None = None):
    """Plan on the past, measure share of held-out future impact each allocation
    of the same patrol budget actually sits on. Returns (curve_df, summary_dict)."""
    if holdout is None or holdout.empty:
        return pd.DataFrame(), {"available": False}

    eval_weeks = set(pd.to_datetime(holdout["week"]).unique())
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["week"] = d["date"].dt.to_period("W-SUN").dt.start_time
    is_eval = d["week"].isin(eval_weeks)

    eval_impact = d[is_eval].groupby("hex")["impact"].sum()
    plan = d[~is_eval].groupby("hex").agg(plan_count=("impact", "size"), plan_impact=("impact", "sum"))
    fc_pred = holdout.groupby("hex")["pred"].sum()

    universe = pd.Index(eval_impact.index).union(plan.index).union(fc_pred.index)
    plan = plan.reindex(universe).fillna(0.0)
    fc_pred = fc_pred.reindex(universe).fillna(0.0)
    eval_impact = eval_impact.reindex(universe).fillna(0.0)

    rank = {
        "Status quo (ticket count)": plan["plan_count"].sort_values(ascending=False).index.tolist(),
        "Impact-weighted (static)": plan["plan_impact"].sort_values(ascending=False).index.tolist(),
        "Forecast-driven (ours)": fc_pred.sort_values(ascending=False).index.tolist(),
        "Oracle (ceiling)": eval_impact.sort_values(ascending=False).index.tolist(),
    }

    budgets = budgets or [10, 20, 30, plan_size, 50, 75, 100, 150, 200, 300]
    budgets = sorted({b for b in budgets if b <= len(universe)})
    rows = []
    for k in budgets:
        row = {"budget": k}
        for name, order in rank.items():
            row[name] = round(100 * _coverage(order, eval_impact, k), 1)
        rows.append(row)
    curve = pd.DataFrame(rows)

    at = curve[curve["budget"] == plan_size]
    at = at.iloc[0] if not at.empty else curve.iloc[len(curve) // 2]
    sq, ours = float(at["Status quo (ticket count)"]), float(at["Forecast-driven (ours)"])
    static, oracle = float(at["Impact-weighted (static)"]), float(at["Oracle (ceiling)"])
    achievable = oracle - sq
    summary = {
        "available": True, "plan_size": int(at["budget"]),
        "coverage_status_quo_pct": sq, "coverage_impact_static_pct": static,
        "coverage_forecast_pct": ours, "coverage_oracle_pct": oracle,
        "gain_vs_status_quo_pts": round(ours - sq, 1),
        "relative_uplift_pct": round(100 * (ours - sq) / sq, 1) if sq > 0 else 0.0,
        "share_of_achievable_gain_pct": round(100 * (ours - sq) / achievable, 1) if achievable > 0 else 0.0,
        "n_eval_weeks": int(len(eval_weeks)),
    }
    return curve, summary


# ============================================================================
#  OPTIMIZER  - greedy weighted maximum-coverage patrol schedule
# ============================================================================
def build_schedule(hex_hourly: pd.DataFrame, hex_df: pd.DataFrame,
                   units_per_shift: int = DEFAULT_UNITS_PER_SHIFT,
                   blindspot_reserve_frac: float = 0.2) -> pd.DataFrame:
    """Per shift, reserve a slice of units for active blind spots, then fill the
    rest with the highest-impact hexes in that time window."""
    meta = hex_df.set_index("hex")
    blind_rank = (hex_df[hex_df["is_blindspot"]]
                  .sort_values("blindspot_score", ascending=False)["hex"].tolist())
    rows = []
    for shift_name, hours in SHIFTS.items():
        hours = list(hours)
        window_impact = (hex_hourly[hex_hourly["hour"].isin(hours)]
                         .groupby("hex")["impact"].sum().sort_values(ascending=False))
        active = set(window_impact.index)
        n_blind = min(len(blind_rank), max(1, round(units_per_shift * blindspot_reserve_frac)))
        chosen_blind = [h for h in blind_rank if h in active][:n_blind]

        chosen = [(h, window_impact.get(h, 0.0), "BLIND SPOT") for h in chosen_blind]
        for hex_id, impact in window_impact.items():
            if len(chosen) >= units_per_shift:
                break
            if hex_id in chosen_blind:
                continue
            tag = "BLIND SPOT" if (hex_id in meta.index and bool(meta.loc[hex_id, "is_blindspot"])) else "Known hotspot"
            chosen.append((hex_id, impact, tag))

        for unit_no, (hex_id, impact, tag) in enumerate(chosen, start=1):
            info = meta.loc[hex_id] if hex_id in meta.index else None
            rows.append({
                "shift": shift_name, "unit": unit_no, "hex": hex_id,
                "lat": float(info["lat"]) if info is not None else None,
                "lon": float(info["lon"]) if info is not None else None,
                "police_station": info["police_station"] if info is not None else "-",
                "expected_impact": round(float(impact), 1),
                "top_violation": info["top_violation"] if info is not None else "-",
                "blindspot_score": float(info["blindspot_score"]) if info is not None else 0.0,
                "primary_target": tag,
            })
    return pd.DataFrame(rows)


# ============================================================================
#  PIPELINE  - run everything, return one results bundle
# ============================================================================
def compute_all(csv_path: str, nrows: int | None = None,
                units_per_shift: int = DEFAULT_UNITS_PER_SHIFT, verbose: bool = False) -> dict:
    """Run the full analysis in-memory and return every artifact the dashboard needs."""
    def say(msg):
        if verbose:
            print(msg)

    say("[1/9] loading + cleaning ...")
    df = load_violations(csv_path, nrows=nrows)
    say(f"      {len(df):,} clean in-Bengaluru violations")

    say("[2/9] scoring congestion impact ...")
    df = add_impact_score(df)

    say("[3/9] assigning H3 hexes ...")
    df = assign_hex(df)

    say("[4/9] building hotspots + temporal profiles ...")
    hex_df = build_hex_hotspots(df)
    hourly = build_hex_hourly(df)

    say("[5/9] detecting under-enforced blind spots ...")
    hex_df = add_blindspot_score(hex_df)

    say("[6/9] computing network cascade (chokepoints) ...")
    hex_df = add_cascade(hex_df)

    say("[7/9] detecting emerging hotspots ...")
    hex_df = add_anomaly_scores(df, hex_df)

    say("[8/9] training one-week-ahead forecaster ...")
    forecast_df, eval_curve, fc_metrics, holdout = run_forecast(df, hex_df)
    hex_df = hex_df.merge(
        forecast_df[["hex", "pred_next_impact", "forecast_delta",
                     "forecast_change_pct", "forecast_score"]],
        on="hex", how="left")
    if verbose and not fc_metrics.get("fallback"):
        mm = fc_metrics["metrics_model"]
        say(f"      model MAE {mm['mae']:.2f} | R2 {mm['r2']:.3f} | "
            f"lift vs persistence {fc_metrics['mae_lift_vs_persistence_pct']}%")

    plan_size = units_per_shift * len(SHIFTS)
    coverage_curve, coverage = evaluate_coverage(df, holdout, plan_size=plan_size)
    if verbose and coverage.get("available"):
        say(f"      coverage @ {coverage['plan_size']} cells: forecast "
            f"{coverage['coverage_forecast_pct']}% vs status-quo "
            f"{coverage['coverage_status_quo_pct']}%")

    say("[9/9] optimising enforcement schedule ...")
    sched = build_schedule(hourly, hex_df, units_per_shift=units_per_shift)

    meta = {
        "total_violations": int(len(df)),
        "n_hexes": int(len(hex_df)),
        "n_blindspots": int(hex_df["is_blindspot"].sum()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "total_impact": float(df["impact"].sum()),
        "h3_res": H3_RES,
        "units_per_shift": units_per_shift,
        "impact_top1pct": float(hex_df.head(max(1, len(hex_df) // 100))["impact"].sum() / hex_df["impact"].sum()),
        "impact_top50_hexes": float(hex_df.head(50)["impact"].sum() / hex_df["impact"].sum()),
        "n_chokepoints": int((hex_df["rank_jump"] >= 10).sum()),
        "max_rank_jump": int(hex_df["rank_jump"].max()),
        "n_emerging": int(hex_df["is_emerging"].sum()),
        "anomaly_recent_weeks": ANOMALY_RECENT_WEEKS,
        "forecast": fc_metrics,
        "coverage": coverage,
    }
    return {"hex_df": hex_df, "hourly": hourly, "sched": sched, "forecast": forecast_df,
            "curve": eval_curve, "coverage_curve": coverage_curve, "meta": meta}


def run_headless(csv_path: str, nrows: int | None = None,
                 units_per_shift: int = DEFAULT_UNITS_PER_SHIFT) -> dict:
    """Headless mode: compute everything, write CSVs to ./processed/, print summary."""
    print(f"ParkSight - analysing {os.path.basename(csv_path)}")
    res = compute_all(csv_path, nrows=nrows, units_per_shift=units_per_shift, verbose=True)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    res["hex_df"].to_csv(os.path.join(PROCESSED_DIR, "hex_hotspots.csv"), index=False)
    res["hourly"].to_csv(os.path.join(PROCESSED_DIR, "hex_hourly.csv"), index=False)
    res["sched"].to_csv(os.path.join(PROCESSED_DIR, "enforcement_schedule.csv"), index=False)
    res["forecast"].to_csv(os.path.join(PROCESSED_DIR, "forecast_hex.csv"), index=False)
    res["curve"].to_csv(os.path.join(PROCESSED_DIR, "forecast_eval.csv"), index=False)
    res["coverage_curve"].to_csv(os.path.join(PROCESSED_DIR, "coverage_curve.csv"), index=False)
    with open(os.path.join(PROCESSED_DIR, "meta.json"), "w") as f:
        json.dump(res["meta"], f, indent=2)

    print("\nDONE. CSVs written to", PROCESSED_DIR)
    print(json.dumps(res["meta"], indent=2))
    return res


# ============================================================================
#  DASHBOARD  - Streamlit + pydeck
# ============================================================================
def run_dashboard():
    import pydeck as pdk
    import streamlit as st

    st.set_page_config(page_title="ParkSight - Parking-Congestion Intelligence",
                       layout="wide", page_icon="🚦")
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500..800&family=JetBrains+Mono:wght@400;500;700&family=Sora:wght@400;500;600&display=swap');

          :root {
            --bg:#0a0e14; --panel-a:#151d29; --panel-b:#0e151f;
            --border:#22303f; --border-soft:#1a2533;
            --text:#e7eef6; --dim:#7f8ea3; --dim-2:#5d6b7d;
            --amber:#ffb454; --amber-deep:#f59e0b;
            --green:#4ade80; --blue:#5eb0ef; --red:#f87171;
            --mono:'JetBrains Mono', ui-monospace, monospace;
            --display:'Bricolage Grotesque', sans-serif;
            --body:'Sora', system-ui, sans-serif;
          }

          /* --- atmosphere: deep slate + signal-amber glow + faint grid --- */
          .stApp {
            background:
              radial-gradient(1100px 520px at 78% -8%, rgba(255,180,84,.10), transparent 60%),
              radial-gradient(900px 600px at 0% 0%, rgba(94,176,239,.06), transparent 55%),
              linear-gradient(180deg,#0b1018 0%, var(--bg) 60%);
            background-attachment: fixed;
          }
          .stApp::before {
            content:""; position:fixed; inset:0; pointer-events:none; z-index:0; opacity:.5;
            background-image:
              linear-gradient(rgba(120,150,190,.05) 1px, transparent 1px),
              linear-gradient(90deg, rgba(120,150,190,.05) 1px, transparent 1px);
            background-size:46px 46px;
            -webkit-mask-image: radial-gradient(circle at 50% 0%, #000 0%, transparent 75%);
                    mask-image: radial-gradient(circle at 50% 0%, #000 0%, transparent 75%);
          }
          .block-container {padding-top: 1.6rem; padding-bottom: 3rem; position:relative; z-index:1;}
          html, body, [class*="css"] { font-family: var(--body); }
          h1, h2, h3 { font-family: var(--display); letter-spacing:-.6px; color:var(--text); }

          /* --- custom header --- */
          .ps-head { margin:.2rem 0 1.4rem; animation: psRise .7s cubic-bezier(.2,.8,.2,1) both; }
          .ps-eyebrow {
            font-family:var(--mono); font-size:.72rem; letter-spacing:.28em; text-transform:uppercase;
            color:var(--amber); display:flex; align-items:center; gap:.6rem; margin-bottom:.55rem;
          }
          .ps-dot { width:8px; height:8px; border-radius:50%; background:var(--amber);
                    box-shadow:0 0 0 0 rgba(255,180,84,.7); animation: psPulse 2.4s infinite; }
          .ps-title { font-size:3.1rem; font-weight:800; line-height:.95; margin:0;
                      background:linear-gradient(92deg,#fff 0%, #cdd9e8 48%, var(--amber) 130%);
                      -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
          .ps-title b { -webkit-text-fill-color:var(--amber); font-weight:800; }
          .ps-sub { color:var(--dim); font-size:1.0rem; max-width:60ch; margin:.6rem 0 0;
                    border-left:2px solid var(--amber-deep); padding-left:.85rem; line-height:1.5; }

          /* --- metric cards --- */
          div[data-testid="stMetric"] {
            position:relative; overflow:hidden;
            background: linear-gradient(158deg, var(--panel-a) 0%, var(--panel-b) 100%);
            border: 1px solid var(--border); border-radius: 14px;
            padding: 15px 16px 12px; box-shadow: 0 6px 18px rgba(0,0,0,.35);
            transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
            animation: psRise .6s cubic-bezier(.2,.8,.2,1) both;
          }
          div[data-testid="stMetric"]::before {
            content:""; position:absolute; top:0; left:0; right:0; height:2px;
            background:linear-gradient(90deg, var(--amber), transparent 75%); opacity:.8;
          }
          div[data-testid="stMetric"]:hover {
            transform: translateY(-3px); border-color:#36475c;
            box-shadow:0 12px 28px rgba(0,0,0,.45), 0 0 0 1px rgba(255,180,84,.12);
          }
          /* staggered load-in across the metric row */
          div[data-testid="stMetric"]:nth-child(1){animation-delay:.02s}
          div[data-testid="stMetric"]:nth-child(2){animation-delay:.07s}
          div[data-testid="stMetric"]:nth-child(3){animation-delay:.12s}
          div[data-testid="stMetric"]:nth-child(4){animation-delay:.17s}
          div[data-testid="stMetric"]:nth-child(5){animation-delay:.22s}
          div[data-testid="stMetric"]:nth-child(6){animation-delay:.27s}
          div[data-testid="stMetric"]:nth-child(7){animation-delay:.32s}
          div[data-testid="stMetricValue"] { font-family:var(--mono); font-size:1.55rem; font-weight:700; color:#fff; }
          div[data-testid="stMetricLabel"] { color:var(--dim); font-weight:600; letter-spacing:.01em; }
          div[data-testid="stMetricLabel"] p { font-size:.78rem; }

          /* --- pills --- */
          .pill { display:inline-block; padding:4px 11px; border-radius:999px; font-family:var(--mono);
                  font-size:0.74rem; font-weight:600; margin-right:6px; letter-spacing:.02em; }
          .pill-good { background:rgba(74,222,128,.12); color:var(--green); border:1px solid rgba(74,222,128,.35); }
          .pill-info { background:rgba(94,176,239,.12); color:var(--blue); border:1px solid rgba(94,176,239,.35); }

          /* --- tabs: terminal-style underline --- */
          .stTabs [data-baseweb="tab-list"] { gap:2px; border-bottom:1px solid var(--border-soft); }
          .stTabs [data-baseweb="tab"] {
            font-family:var(--mono); font-size:.86rem; font-weight:500; letter-spacing:.02em;
            color:var(--dim); padding:.4rem .9rem; background:transparent;
          }
          .stTabs [aria-selected="true"] { color:var(--amber) !important; }
          .stTabs [data-baseweb="tab-highlight"] { background:var(--amber) !important; height:2px; }

          /* --- sidebar --- */
          section[data-testid="stSidebar"] {
            background:linear-gradient(180deg,#0d141d 0%, #0a0f16 100%);
            border-right:1px solid var(--border-soft);
          }
          section[data-testid="stSidebar"] h2 { font-size:1.1rem; }

          /* --- dataframes & headings polish --- */
          [data-testid="stDataFrame"] { border:1px solid var(--border-soft); border-radius:10px; overflow:hidden; }
          .caption-dim { color:var(--dim); font-size:0.86rem; }
          hr { border-color:var(--border-soft); }

          /* --- command-bar status chips --- */
          .ps-chips { display:flex; gap:.5rem; flex-wrap:wrap; margin:-.4rem 0 1.2rem; }
          .ps-chip { font-family:var(--mono); font-size:.72rem; color:var(--dim);
                     border:1px solid var(--border-soft); border-radius:8px; padding:.32rem .65rem;
                     background:rgba(255,255,255,.018); }
          .ps-chip b { color:var(--text); font-weight:600; }
          .ps-chip .ok { color:var(--green); } .ps-chip .warn { color:var(--amber); }

          /* --- numbered section headers --- */
          .ps-sec { display:flex; align-items:baseline; gap:.7rem; margin:.3rem 0 .1rem; }
          .ps-sec .n { font-family:var(--mono); font-size:.74rem; color:var(--amber);
                       border:1px solid rgba(255,180,84,.32); border-radius:6px; padding:.08rem .4rem; }
          .ps-sec h3 { margin:0; font-size:1.18rem; }
          .ps-sec .meta { font-family:var(--mono); font-size:.72rem; color:var(--dim-2); margin-left:auto; }

          /* --- live target panel --- */
          .ps-panel { border:1px solid var(--border); border-radius:14px; overflow:hidden;
                      background:linear-gradient(168deg,var(--panel-a),var(--panel-b));
                      box-shadow:0 6px 18px rgba(0,0,0,.35); animation: psRise .6s cubic-bezier(.2,.8,.2,1) .15s both; }
          .ps-panel-h { font-family:var(--mono); font-size:.7rem; letter-spacing:.18em; text-transform:uppercase;
                        color:var(--amber); padding:.75rem .9rem; border-bottom:1px solid var(--border-soft);
                        display:flex; justify-content:space-between; align-items:center; }
          .ps-panel-h .cnt { color:var(--dim-2); }
          .ps-tgt { display:flex; align-items:center; gap:.7rem; padding:.5rem .9rem;
                    border-bottom:1px solid rgba(255,255,255,.04); transition:background .15s; }
          .ps-tgt:hover { background:rgba(255,180,84,.05); }
          .ps-tgt .rk { font-family:var(--mono); font-size:.78rem; color:var(--dim-2); width:1.5rem; flex:none; }
          .ps-tgt .nm { flex:1; min-width:0; }
          .ps-tgt .nm b { display:block; font-size:.84rem; color:var(--text); font-weight:600;
                          white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
          .ps-tgt .nm span { font-size:.71rem; color:var(--dim); }
          .ps-tgt .val { font-family:var(--mono); font-weight:700; font-size:.9rem; flex:none; }
          .ps-legend { padding:.7rem .9rem; font-size:.78rem; color:var(--dim); line-height:1.45;
                       border-top:1px solid var(--border-soft); background:rgba(0,0,0,.18); }
          .ps-legend .sw { display:inline-block; width:10px; height:10px; border-radius:3px;
                           margin-right:.4rem; vertical-align:middle; }

          /* --- lens selector: segmented pills (style the horizontal radio) --- */
          div[role="radiogroup"] { gap:.4rem !important; }
          div[role="radiogroup"] > label {
            background:linear-gradient(160deg,var(--panel-a),var(--panel-b));
            border:1px solid var(--border-soft); border-radius:10px;
            padding:.35rem .7rem !important; margin:0 !important; transition:all .15s;
          }
          div[role="radiogroup"] > label:hover { border-color:#3a4b60; }
          div[role="radiogroup"] > label:has(input:checked) {
            border-color:var(--amber); background:rgba(255,180,84,.10);
            box-shadow:0 0 0 1px rgba(255,180,84,.25);
          }
          div[role="radiogroup"] > label > div:first-child { display:none; }   /* hide the dot */
          div[role="radiogroup"] label p { font-family:var(--mono); font-size:.8rem; font-weight:500; }

          /* --- ranked cards (replace data tables) --- */
          .ps-list { display:flex; flex-direction:column; gap:.45rem; margin:.4rem 0 .3rem; }
          .ps-card { border:1px solid var(--border-soft); border-radius:12px; padding:.6rem .8rem;
                     background:linear-gradient(160deg,var(--panel-a),var(--panel-b)); transition:.15s; }
          .ps-card:hover { border-color:#36475c; transform:translateX(3px); }
          .ps-card .top { display:flex; align-items:center; gap:.55rem; }
          .ps-card .rk { font-family:var(--mono); font-size:.76rem; color:var(--dim-2); flex:none; min-width:1.3rem; }
          .ps-card .nm { font-weight:600; color:var(--text); font-size:.9rem;
                         white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
          .ps-card .val { margin-left:auto; font-family:var(--mono); font-weight:700; font-size:.95rem; flex:none; }
          .ps-card .sub { color:var(--dim); font-size:.75rem; margin:.22rem 0 .45rem; }
          .ps-bar { height:6px; border-radius:999px; background:rgba(255,255,255,.06); overflow:hidden; }
          .ps-bar > i { display:block; height:100%; border-radius:999px; opacity:.92;
                        animation: psGrow .6s cubic-bezier(.2,.8,.2,1) both; }
          .badge { font-family:var(--mono); font-size:.65rem; font-weight:600; padding:.12rem .42rem;
                   border-radius:6px; flex:none; letter-spacing:.02em; }
          .badge-up   { background:rgba(248,113,113,.14); color:var(--red);   border:1px solid rgba(248,113,113,.4); }
          .badge-dn   { background:rgba(74,222,128,.14);  color:var(--green); border:1px solid rgba(74,222,128,.4); }
          .badge-blind{ background:rgba(255,140,0,.14);   color:#ffb454;      border:1px solid rgba(255,140,0,.4); }
          .badge-known{ background:rgba(94,176,239,.12);  color:var(--blue);  border:1px solid rgba(94,176,239,.35); }

          /* --- stat comparison cards (model vs baselines) --- */
          .ps-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:.6rem; margin:.4rem 0; }
          .ps-stat { border:1px solid var(--border-soft); border-radius:12px; padding:.75rem .85rem;
                     background:linear-gradient(160deg,var(--panel-a),var(--panel-b)); }
          .ps-stat.best { border-color:var(--amber); box-shadow:0 0 0 1px rgba(255,180,84,.22); }
          .ps-stat .lbl { font-family:var(--mono); font-size:.72rem; color:var(--dim); margin-bottom:.3rem; }
          .ps-stat .big { font-family:var(--mono); font-size:1.5rem; font-weight:700; color:#fff; line-height:1; }
          .ps-stat .big span { font-size:.64rem; color:var(--dim-2); margin-left:.35rem; font-weight:500; }
          .ps-stat .mini { font-family:var(--mono); font-size:.7rem; color:var(--dim); margin-top:.35rem; }

          @keyframes psGrow { from{width:0;} }
          @keyframes psRise { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
          @keyframes psPulse {
            0%{box-shadow:0 0 0 0 rgba(255,180,84,.6);}
            70%{box-shadow:0 0 0 7px rgba(255,180,84,0);}
            100%{box-shadow:0 0 0 0 rgba(255,180,84,0);}
          }
        </style>
        """, unsafe_allow_html=True)

    try:
        csv_path = find_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    @st.cache_data(show_spinner="Crunching the violation log (first load only)…")
    def _load(path, mtime, nrows):
        return compute_all(path, nrows=nrows)

    res = _load(csv_path, os.path.getmtime(csv_path), None)
    hex_df, hourly = res["hex_df"], res["hourly"]
    curve, forecast, coverage = res["curve"], res["forecast"], res["coverage_curve"]
    meta = res["meta"]
    fc = meta.get("forecast", {}) or {}
    fc_ok = not fc.get("fallback", True)
    cov = meta.get("coverage", {}) or {}
    cov_ok = cov.get("available", False)

    # ---- header ----
    st.markdown(
        f"""
        <div class="ps-head">
          <div class="ps-eyebrow"><span class="ps-dot"></span>
            Bengaluru Traffic Police · Carriageway Intelligence ·
            {meta['date_min']} → {meta['date_max']}</div>
          <h1 class="ps-title">Park<b>Sight</b></h1>
          <p class="ps-sub">From {meta['total_violations']:,} anonymised parking-violation records
            to a quantified carriageway-obstruction map, a one-week-ahead forecast,
            and a deployable patrol plan.</p>
        </div>
        """, unsafe_allow_html=True)

    # ---- command status bar ----
    fc_chip = ("<span class='ps-chip'>FORECAST <b class='ok'>ONLINE</b></span>" if fc_ok
               else "<span class='ps-chip'>FORECAST <b class='warn'>FALLBACK</b></span>")
    st.markdown(
        "<div class='ps-chips'>"
        f"<span class='ps-chip'>WINDOW <b>{meta['date_min']} → {meta['date_max']}</b></span>"
        f"<span class='ps-chip'>H3 RES <b>{meta['h3_res']}</b></span>"
        f"<span class='ps-chip'>ACTIVE CELLS <b>{meta['n_hexes']:,}</b></span>"
        f"<span class='ps-chip'>TOTAL IMPACT <b>{meta['total_impact']:,.0f}</b></span>"
        f"{fc_chip}</div>", unsafe_allow_html=True)

    # ---- KPI strip ----
    m = st.columns(7)
    m[0].metric("Violations analysed", f"{meta['total_violations']:,}")
    m[1].metric("Top-50 cells hold", f"{meta['impact_top50_hexes']*100:.0f}%",
                help="Share of total congestion-impact in just 50 hexes.")
    m[2].metric("Under-enforced blind spots", f"{meta['n_blindspots']:,}")
    m[3].metric("Network chokepoints", f"{meta.get('n_chokepoints', '-')}",
                help="Hexes that jump >=10 priority places once network cascade is applied.")
    m[4].metric("Emerging hotspots", f"{meta.get('n_emerging', '-')}",
                help=f"Cells surging vs their own baseline over the last {meta.get('anomaly_recent_weeks','?')} weeks.")
    if fc_ok:
        m[5].metric("Forecast R² (held-out)", f"{fc['metrics_model']['r2']:.2f}",
                    delta=f"{fc['mae_lift_vs_persistence_pct']:.0f}% lift vs persistence")
    else:
        m[5].metric("Forecast model", "fallback")
    if cov_ok:
        m[6].metric("Held-out impact covered", f"{cov['coverage_forecast_pct']:.0f}%",
                    delta=f"+{cov['gain_vs_status_quo_pts']:.1f} pts vs status-quo")
    else:
        m[6].metric("Coverage backtest", "-")

    def section(tag, title, meta_txt=""):
        st.markdown(
            f"<div class='ps-sec'><span class='n'>{tag}</span><h3>{title}</h3>"
            f"<span class='meta'>{meta_txt}</span></div>", unsafe_allow_html=True)

    def rank_list(rows, name_col, value_col, accent, sub_fn,
                  fmt="{:,.1f}", suffix="", note_fn=None, max_n=15, empty="Nothing to show here."):
        """Render a dataframe as a ranked card list with value bars (shadcn-style)."""
        if rows is None or rows.empty:
            st.caption(empty)
            return
        mx = float(rows[value_col].abs().max()) or 1.0
        out = ["<div class='ps-list'>"]
        for i, (_, r) in enumerate(rows.head(max_n).iterrows(), 1):
            v = float(r[value_col])
            w = max(5.0, abs(v) / mx * 100)
            note = note_fn(r) if note_fn else ""
            out.append(
                "<div class='ps-card'><div class='top'>"
                f"<span class='rk'>{i:02d}</span><span class='nm'>{r[name_col]}</span>{note}"
                f"<span class='val' style='color:{accent}'>{fmt.format(v)}{suffix}</span></div>"
                f"<div class='sub'>{sub_fn(r)}</div>"
                f"<div class='ps-bar'><i style='width:{w:.0f}%;background:{accent}'></i></div></div>")
        out.append("</div>")
        st.markdown("".join(out), unsafe_allow_html=True)

    LENSES = ["Congestion-impact hotspots", "Under-enforced blind spots",
              "Network choke points", "🔮 Next-week forecast", "⚠️ Emerging hotspots"]

    # ======================= MAP · live congestion console =======================
    section("MAP", "Live congestion map", "extruded H3 · interactive")
    map_col, panel_col = st.columns([2.6, 1], gap="large")

    with map_col:
        view = st.radio("Map lens", LENSES, horizontal=True, label_visibility="collapsed")
        cc1, cc2 = st.columns([2, 1.4])
        hr = cc1.slider("Hour of day (IST)", 0, 23, (7, 11))
        TOPN = {"100": 100, "250": 250, "500": 500, "1000": 1000, "All": 100_000}
        with cc2:
            st.markdown("<span class='caption-dim'>Cells shown</span>", unsafe_allow_html=True)
            top_n = TOPN[st.radio("Cells shown", list(TOPN), index=2, horizontal=True,
                                  label_visibility="collapsed")]

        hours = list(range(hr[0], hr[1] + 1))
        win = hourly[hourly["hour"].isin(hours)].groupby("hex")["impact"].sum()
        df = hex_df.copy()
        df["impact_window"] = df["hex"].map(win).fillna(0.0)

        if view == "Under-enforced blind spots":
            df = df[df["blindspot_score"] > 0].sort_values("blindspot_score", ascending=False)
            weight_col, color_hot = "blindspot_score", [255, 140, 0]
            legend = "Under-enforced blind spot — high obstruction, rising, few tickets."
        elif view == "Network choke points":
            df = df.sort_values("network_leverage", ascending=False)
            weight_col, color_hot = "network_leverage", [138, 43, 226]
            legend = "Network leverage (cascade ÷ local impact) — modest local violation, outsized downstream congestion."
        elif view == "🔮 Next-week forecast":
            if "forecast_score" in df.columns and df["forecast_score"].notna().any():
                df = df[df["forecast_score"].notna()].sort_values("pred_next_impact", ascending=False)
                weight_col, color_hot = "forecast_score", [56, 189, 248]
                legend = "Model-predicted congestion impact for next week."
            else:
                df = df.sort_values("impact_window", ascending=False)
                weight_col, color_hot = "impact_window", [220, 30, 40]
                legend = "Congestion impact (forecast unavailable)."
        elif view == "⚠️ Emerging hotspots":
            df = df[df.get("anomaly_score", 0) > 0].sort_values("anomaly_score", ascending=False)
            weight_col, color_hot = "anomaly_score", [244, 63, 94]
            legend = "Anomaly score — cells surging vs their own recent baseline."
        else:
            df = df.sort_values("impact_window", ascending=False)
            weight_col, color_hot = "impact_window", [220, 30, 40]
            legend = "Congestion impact in the selected hours."

        if df.empty or df[weight_col].max() == 0:
            st.info("No cells to display for this lens / filter.")
        else:
            plot = df.head(top_n).copy()
            mx = plot[weight_col].max() or 1.0
            plot["norm"] = plot[weight_col] / mx
            plot["r"] = color_hot[0]
            plot["g"] = (color_hot[1] + (1 - plot["norm"]) * 180).clip(0, 255).astype(int)
            plot["b"] = (color_hot[2] + (1 - plot["norm"]) * 120).clip(0, 255).astype(int)
            plot["elevation"] = plot[weight_col]
            for c in ("pred_next_impact", "forecast_change_pct", "anomaly_score",
                      "surge_z", "network_leverage", "rank_jump", "cascade_score"):
                plot[c] = plot[c].fillna(0).round(1) if c in plot.columns else 0

            layer = pdk.Layer("H3HexagonLayer", plot, get_hexagon="hex",
                              get_fill_color="[r, g, b, 180]", get_elevation="elevation",
                              elevation_scale=4, extruded=True, pickable=True, coverage=0.95)
            view_state = pdk.ViewState(latitude=12.97, longitude=77.59, zoom=10.5, pitch=45)
            tooltip = {
                "html": "<b>{police_station}</b><br/>Top: {top_violation}<br/>"
                        "Violations: {violations} · Local score: {impact_score}<br/>"
                        "Forecast next wk: {pred_next_impact} ({forecast_change_pct}%)<br/>"
                        "Anomaly: {anomaly_score} · Net leverage: {network_leverage}×",
                "style": {"backgroundColor": "#0f1620", "color": "white",
                          "fontSize": "12px", "border": "1px solid #2a3647"}}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                                     tooltip=tooltip, map_style="dark"), use_container_width=True)

    with panel_col:
        mlabel = {"blindspot_score": "Blindspot", "network_leverage": "Leverage",
                  "forecast_score": "Forecast", "anomaly_score": "Anomaly",
                  "impact_window": "Impact"}.get(weight_col, "Impact")
        sw = f"rgb({color_hot[0]},{color_hot[1]},{color_hot[2]})"
        val_rgb = (f"rgb({min(color_hot[0]+40,255)},{min(color_hot[1]+90,255)},"
                   f"{min(color_hot[2]+90,255)})")
        targets = df.head(12)
        rows = ""
        for i, (_, r) in enumerate(targets.iterrows(), 1):
            v = float(r[weight_col])
            vtxt = f"{v:,.1f}×" if weight_col == "network_leverage" else f"{v:,.1f}"
            rows += (
                f"<div class='ps-tgt'><span class='rk'>{i:02d}</span>"
                f"<span class='nm'><b>{r['police_station']}</b>"
                f"<span>{r['top_violation']} · {int(r['violations'])} tickets</span></span>"
                f"<span class='val' style='color:{val_rgb}'>{vtxt}</span></div>")
        if not rows:
            rows = ("<div class='ps-tgt'><span class='nm'>"
                    "<span>No cells for this lens.</span></span></div>")
        st.markdown(
            "<div class='ps-panel'>"
            f"<div class='ps-panel-h'><span>Top targets · {mlabel}</span>"
            f"<span class='cnt'>{len(targets)} of {len(df):,}</span></div>"
            f"{rows}"
            f"<div class='ps-legend'><span class='sw' style='background:{sw}'></span>{legend}</div>"
            "</div>", unsafe_allow_html=True)

    st.markdown("---")
    tab_fc, tab_emerge, tab_tables, tab_plan = st.tabs(
        ["🔮 Forecast", "⚠️ Emerging", "📊 Hotspots", "📋 Enforcement plan"])

    # ---- FORECAST ----
    with tab_fc:
        section("FCST", "One-week-ahead congestion forecast")
        if not fc_ok:
            st.warning("Forecasting fell back to a persistence baseline "
                       f"({fc.get('reason', 'scikit-learn unavailable')}). Install scikit-learn to enable the model.")
        else:
            st.markdown(
                f"<span class='pill pill-info'>{fc['model']}</span>"
                f"<span class='pill pill-info'>{fc['horizon']}</span>"
                f"<span class='pill pill-good'>R² {fc['metrics_model']['r2']:.2f}</span>"
                f"<span class='pill pill-good'>{fc['mae_lift_vs_persistence_pct']:.0f}% better MAE than persistence</span>",
                unsafe_allow_html=True)
            st.caption(
                f"Trained on {fc['n_train_rows']:,} hex-week rows, evaluated on "
                f"{fc['n_test_rows']:,} strictly future rows (final {fc['holdout_weeks']} weeks held out). "
                "Baselines scored on the identical rows.")

            cL, cR = st.columns(2)
            with cL:
                st.markdown("**Held-out accuracy vs baselines**")
                models = [("GBM · ours", fc["metrics_model"], True),
                          ("Persistence", fc["metrics_persistence"], False),
                          ("4-week avg", fc["metrics_mean"], False)]
                cards = "".join(
                    f"<div class='ps-stat{' best' if best else ''}'>"
                    f"<div class='lbl'>{nm}</div>"
                    f"<div class='big'>{mt['mae']:.2f}<span>MAE</span></div>"
                    f"<div class='mini'>RMSE {mt['rmse']:.2f} · R² {mt['r2']:.2f}</div></div>"
                    for nm, mt, best in models)
                st.markdown(f"<div class='ps-stats'>{cards}</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<span class='caption-dim'>MAE improvement: "
                    f"<b>{fc['mae_lift_vs_persistence_pct']:.1f}%</b> vs persistence · "
                    f"<b>{fc['mae_lift_vs_mean_pct']:.1f}%</b> vs 4-week mean.</span>", unsafe_allow_html=True)
            with cR:
                st.markdown("**What drives the prediction**")
                imp = pd.DataFrame(fc["top_features"]).set_index("feature")
                st.bar_chart(imp["importance"], height=240, color="#38bdf8")

            if not curve.empty:
                st.markdown("**Predicted vs actual - city-wide impact, held-out weeks**")
                cc = curve.copy()
                cc["week"] = pd.to_datetime(cc["week"]).dt.date.astype(str)
                st.line_chart(cc.set_index("week")[["actual", "predicted"]], height=280,
                              color=["#94a3b8", "#38bdf8"])

            st.markdown("**Cells forecast to rise most next week**")
            rising = forecast.sort_values("forecast_delta", ascending=False).head(12) \
                if not forecast.empty else forecast
            rank_list(
                rising, "police_station", "forecast_delta", "#38bdf8",
                sub_fn=lambda r: f"{r['top_violation']} · last wk {r['last_week_impact']:.0f} "
                                 f"→ next {r['pred_next_impact']:.0f}",
                fmt="{:+,.0f}", max_n=12,
                note_fn=lambda r: f"<span class='badge badge-up'>+{r['forecast_change_pct']:.0f}%</span>"
                                  if r['forecast_change_pct'] > 0 else "",
                empty="No cells are forecast to rise next week.")

        if cov_ok and not coverage.empty:
            st.markdown("---")
            section("ROI", "Does it actually help? — operational coverage backtest")
            st.caption(
                "Plan on the past, then measure how much of the held-out future "
                f"({cov['n_eval_weeks']} weeks) each allocation of the same patrol budget sits "
                "on. 'Status quo' = highest-ticket-count cells; 'Oracle' = perfect foresight ceiling.")
            k = st.columns(4)
            k[0].metric("Forecast-driven (ours)", f"{cov['coverage_forecast_pct']:.1f}%")
            k[1].metric("Status-quo footprint", f"{cov['coverage_status_quo_pct']:.1f}%")
            k[2].metric("Gain for same units", f"+{cov['gain_vs_status_quo_pts']:.1f} pts",
                        delta=f"{cov['relative_uplift_pct']:.0f}% relative")
            k[3].metric("Share of achievable gain", f"{cov['share_of_achievable_gain_pct']:.0f}%")
            st.markdown(
                f"<span class='caption-dim'>At <b>{cov['plan_size']} cells</b>, a forecast-driven "
                f"deployment covers <b>{cov['coverage_forecast_pct']:.1f}%</b> of next weeks' real "
                f"congestion-impact vs <b>{cov['coverage_status_quo_pct']:.1f}%</b> for today's "
                f"footprint - closing <b>{cov['share_of_achievable_gain_pct']:.0f}%</b> of the gap to "
                f"perfect foresight ({cov['coverage_oracle_pct']:.1f}%).</span>", unsafe_allow_html=True)
            st.markdown("**Coverage vs patrol budget**")
            st.line_chart(coverage.set_index("budget"), height=300,
                          color=["#94a3b8", "#a78bfa", "#38bdf8", "#4ade80"])

    # ---- EMERGING ----
    with tab_emerge:
        section("WATCH", "Emerging hotspots — early-warning board")
        st.caption(
            "Cells behaving abnormally right now: recent weekly impact surging above the "
            "hex's own robust baseline (median/MAD z-score), cross-checked by an IsolationForest.")
        em = hex_df[hex_df.get("is_emerging", False)].copy() if "is_emerging" in hex_df.columns else pd.DataFrame()
        em = em.sort_values("anomaly_score", ascending=False) if not em.empty else em
        rank_list(
            em, "police_station", "anomaly_score", "#fb7185",
            sub_fn=lambda r: f"{r['top_violation']} · {int(r['violations'])} tickets · "
                             f"surge z {r['surge_z']:.1f} · baseline {r['baseline_impact']:.0f}",
            fmt="{:.0f}",
            note_fn=lambda r: f"<span class='badge badge-up'>↑ {r['recent_week_impact']:.0f}/wk</span>",
            empty="No cells currently cross the emerging-surge threshold.")

    # ---- TABLES ----
    with tab_tables:
        section("RANK", "Hotspot & chokepoint tables")
        left, right = st.columns(2)
        with left:
            st.markdown("**Top carriageway-obstruction hotspots**")
            rank_list(
                hex_df.sort_values("impact", ascending=False), "police_station",
                "impact_score", "#ffb454",
                sub_fn=lambda r: f"{r['top_violation']} · {r['top_vehicle']} · "
                                 f"{int(r['violations'])} tickets · {r['high_severity_share']*100:.0f}% high-sev",
                fmt="{:.0f}")
        with right:
            st.markdown("**Blind spots BTP is currently missing**")
            bs = hex_df[hex_df["blindspot_score"] > 0].sort_values("blindspot_score", ascending=False)
            rank_list(
                bs, "police_station", "blindspot_score", "#ff8c00",
                sub_fn=lambda r: f"{r['top_violation']} · {int(r['violations'])} tickets · "
                                 f"trend ×{r['trend_ratio']:.2f} · {r['high_severity_share']*100:.0f}% high-sev",
                fmt="{:.0f}",
                empty="No active blind spots.")

        st.markdown("**🕸️ Network choke points — highest enforcement ROI**")
        st.caption("Modest local counts, but sited where blockage backs traffic across adjacent cells.")
        choke = hex_df[hex_df["rank_jump"] > 0].sort_values("rank_jump", ascending=False)
        rank_list(
            choke, "police_station", "rank_jump", "#a78bfa",
            sub_fn=lambda r: f"{r['top_violation']} · leverage ×{r['network_leverage']:.1f} · "
                             f"local {r['impact_score']:.0f} · cascade {r['cascade_score']:.0f}",
            fmt="+{:.0f}", suffix=" ranks",
            note_fn=lambda r: f"<span class='badge badge-known'>×{r['network_leverage']:.1f}</span>",
            empty="No network chokepoints detected.")

    # ---- PLAN ----
    with tab_plan:
        units = st.number_input("Patrol units per shift", min_value=3, max_value=25,
                                value=int(meta["units_per_shift"]), step=1)
        section("PLAN", f"Recommended enforcement schedule · {units} units/shift")
        st.caption("Greedy weighted maximum-coverage allocation; each shift reserves a slice for active blind spots.")
        sched = build_schedule(hourly, hex_df, units_per_shift=units)
        for shift_name in SHIFTS:
            s = sched[sched["shift"] == shift_name]
            with st.expander(f"{shift_name} — {len(s)} deployments · "
                             f"expected impact {s['expected_impact'].sum():.0f}"):
                rank_list(
                    s, "police_station", "expected_impact", "#ffb454",
                    sub_fn=lambda r: f"{r['top_violation']} · {r['lat']:.4f}, {r['lon']:.4f}",
                    fmt="{:.0f}", suffix=" impact", max_n=len(s),
                    note_fn=lambda r: (
                        "<span class='badge badge-blind'>BLIND SPOT</span>"
                        if r['primary_target'] == "BLIND SPOT"
                        else "<span class='badge badge-known'>HOTSPOT</span>"),
                    empty="No deployments this shift.")
        st.download_button("⬇️ Download full schedule (CSV)",
                           sched.to_csv(index=False), "enforcement_schedule.csv", "text/csv")

    st.markdown("---")
    st.caption(
        f"Data window {meta['date_min']} → {meta['date_max']} · H3 res {meta['h3_res']}. "
        "Timestamps reflect enforcement logging (UTC→IST); the blind-spot index is an explicit "
        "patrol-bias correction, and the forecast/coverage layers are evaluated on strictly held-out time.")


# ============================================================================
#  ENTRY POINT
# ============================================================================
def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _running_under_streamlit():
    run_dashboard()
elif __name__ == "__main__":
    # python parksight.py [csv_path] [nrows]
    csv_arg = next((a for a in sys.argv[1:] if a.lower().endswith(".csv")), None)
    nrows_arg = next((int(a) for a in sys.argv[1:] if a.isdigit()), None)
    path = find_dataset(csv_arg)
    run_headless(path, nrows=nrows_arg)
