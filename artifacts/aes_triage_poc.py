"""
AES Renewable Asset Performance Triage - Proof of Concept
=========================================================
Solar-only POC. Demonstrates the full analytical core:

    raw signals -> expected baseline (Performance Ratio) -> anomaly band
    -> rule-based cause attribution -> financial ranking -> daily triage table

The baseline is the Performance-Ratio (PR) decomposition:

    expected_generation = theoretical_potential x PR_baseline

where `theoretical_potential` comes from first principles (irradiance +
temperature de-rate) and `PR_baseline` is learned from the asset's own
*healthy* operating history (robust rolling median, leakage-filtered).

Run:  python aes_triage_poc.py
Outputs: triage_output.csv, fig_examples.png, fig_ranking.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 0. Configuration
# ----------------------------------------------------------------------
N_ASSETS = 30
N_DAYS = 45
HOURS = N_DAYS * 24
GAMMA = 0.004           # temp coefficient of power, ~ -0.4 %/degC for c-Si
NOCT = 45               # nominal operating cell temperature (degC)
TODAY_WINDOW_H = 48     # the "last 2 days" the morning triage looks at
PERSIST_WIN = 4         # window of recent daytime hours for the persistence test
PERSIST_MIN = 3         # require >= 3 of the last 4 daytime hours to dip (kills noise)
REGIONS = ["Nordeste", "Sudeste", "Sul"]
OEMS = ["OEM-Alpha", "OEM-Bravo", "OEM-Charlie"]

# ----------------------------------------------------------------------
# 1. Asset metadata
# ----------------------------------------------------------------------
asset_ids = [f"SOLAR_{i:03d}" for i in range(1, N_ASSETS + 1)]
meta = pd.DataFrame({
    "asset_id": asset_ids,
    "capacity_mw": RNG.choice([5, 10, 20, 50, 80, 120, 200], N_ASSETS),
    "region": RNG.choice(REGIONS, N_ASSETS),
    "oem": RNG.choice(OEMS, N_ASSETS),
    # true latent PR each healthy asset operates around
    "true_pr": RNG.uniform(0.78, 0.86, N_ASSETS),
    # contract: ~60% PPA (fixed price), rest merchant (volatile)
    "contract_type": RNG.choice(["PPA", "Merchant"], N_ASSETS, p=[0.6, 0.4]),
    "ppa_rate": RNG.uniform(180, 260, N_ASSETS),   # R$/MWh fixed
}).set_index("asset_id")

# ----------------------------------------------------------------------
# 2. Time axis + shared weather drivers
# ----------------------------------------------------------------------
idx = pd.date_range("2025-04-01", periods=HOURS, freq="h")
hour = idx.hour.values
day = (idx - idx[0]).days

# clear-sky irradiance: half-sine over daylight (6h-18h), seasonal drift
solar_elev = np.clip(np.sin((hour - 6) / 12 * np.pi), 0, None)
seasonal = 1 + 0.05 * np.sin(2 * np.pi * day / 365)
clear_sky = 1000 * solar_elev * seasonal       # W/m2 at plane of array

# daily cloud factor (one per day, shared across fleet for a region)
cloud = RNG.uniform(0.55, 1.0, N_DAYS)[day]
base_irr = clear_sky * cloud
ambient_t = 22 + 8 * solar_elev + 3 * np.sin(2 * np.pi * day / 365)


def cell_temp(irr, amb):
    return amb + (irr / 800.0) * (NOCT - 20)


# ----------------------------------------------------------------------
# 3. Build per-asset telemetry (healthy world first, then inject faults)
# ----------------------------------------------------------------------
rows = []
for aid in asset_ids:
    m = meta.loc[aid]
    irr = base_irr * RNG.uniform(0.97, 1.03, HOURS)        # local sensor noise
    irr = np.clip(irr, 0, None)
    amb = ambient_t + RNG.normal(0, 1.0, HOURS)
    ct = cell_temp(irr, amb)

    potential = m.capacity_mw * (irr / 1000.0) * (1 - GAMMA * (ct - 25))
    potential = np.clip(potential, 0, None)

    pr_series = np.full(HOURS, m.true_pr)
    availability = np.full(HOURS, RNG.uniform(0.985, 1.0))
    inverter_status = np.array(["ok"] * HOURS, dtype=object)
    curtailment_flag = np.zeros(HOURS, dtype=int)

    gen = potential * pr_series
    df = pd.DataFrame({
        "timestamp": idx, "asset_id": aid,
        "irradiance": irr, "ambient_temp": amb, "cell_temp": ct,
        "potential_mwh": potential, "generation_mwh": gen,
        "availability_pct": availability * 100,
        "inverter_status": inverter_status,
        "curtailment_flag": curtailment_flag,
        "price": np.where(m.contract_type == "PPA", m.ppa_rate,
                          RNG.uniform(120, 420, HOURS)),
    })
    rows.append(df)

data = pd.concat(rows, ignore_index=True)
data["region"] = data.asset_id.map(meta["region"])

# add diffuse generation noise everywhere
data["generation_mwh"] *= RNG.normal(1.0, 0.03, len(data)).clip(0.85, 1.15)
data["generation_mwh"] = data["generation_mwh"].clip(lower=0)

# ----------------------------------------------------------------------
# 3b. INJECT GROUND-TRUTH FAULTS (so we can validate detection)
# ----------------------------------------------------------------------
fault_log = {}


def mask(aid, d0, d1):
    return (data.asset_id == aid) & (data.timestamp >= idx[0] + pd.Timedelta(days=d0)) \
                                  & (data.timestamp < idx[0] + pd.Timedelta(days=d1))


# (A) Inverter OUTAGE - abrupt, recent (last 2 days). High capacity asset.
a_out = "SOLAR_005"
mo = mask(a_out, 43, 45) & (data.irradiance > 50)
data.loc[mo, "generation_mwh"] *= 0.45
data.loc[mo, "availability_pct"] = 52.0
data.loc[mo, "inverter_status"] = "fault"
fault_log[a_out] = "Outage"

# (B) Gradual DEGRADATION - slow PR decay over the whole window
a_deg = "SOLAR_012"
md = data.asset_id == a_deg
decay = 1 - 0.18 * (data.loc[md, "timestamp"] - idx[0]).dt.days / N_DAYS
data.loc[md, "generation_mwh"] *= decay.values
fault_log[a_deg] = "Degradation"

# (C) Frozen SENSOR - irradiance stuck at a constant for last 3 days
a_sen = "SOLAR_019"
ms = mask(a_sen, 42, 45)
data.loc[ms, "irradiance"] = 430.0      # frozen, ignores day/night
fault_log[a_sen] = "Sensor/Data issue"

# (D) CURTAILMENT - recent capping with flag set, availability normal
a_cur = "SOLAR_023"
mc = mask(a_cur, 43, 45) & (data.irradiance > 200)
data.loc[mc, "generation_mwh"] *= 0.40
data.loc[mc, "curtailment_flag"] = 1
fault_log[a_cur] = "Curtailment / Market"

# (E) MIXED CAUSE - curtailment on day 1, partial outage on day 2 (same asset).
#     Separated in time so the window contains BOTH causes -> confidence < 1.0.
a_mix = "SOLAR_027"
me1 = mask(a_mix, 43, 44) & (data.irradiance > 200)     # day 1: curtailment
data.loc[me1, "generation_mwh"] *= 0.45
data.loc[me1, "curtailment_flag"] = 1
me2 = mask(a_mix, 44, 45) & (data.irradiance > 50)      # day 2: partial outage
data.loc[me2, "generation_mwh"] *= 0.55
data.loc[me2, "availability_pct"] = 68.0
data.loc[me2, "inverter_status"] = "fault"
fault_log[a_mix] = "Mixed (curtailment + outage)"

# (F) REGIONAL RESOURCE/WEATHER event - a localized weather drop the POA
#     sensors under-captured, hitting EVERY asset in one region for an
#     afternoon. Generation falls but measured irradiance stays normal, so it
#     looks like underperformance; because peers dip TOGETHER it is attributed
#     to weather, not an asset fault (don't dispatch a tech for each one).
reg_assets = meta.index[meta.region == "Nordeste"]      # no asset-specific faults here
mw = (data.asset_id.isin(reg_assets)
      & (data.timestamp >= idx[0] + pd.Timedelta(days=44, hours=11))
      & (data.timestamp < idx[0] + pd.Timedelta(days=44, hours=17)))
data.loc[mw, "generation_mwh"] *= 0.62
fault_log["REGION:Nordeste"] = "Resource/Weather (regional)"

# recompute potential with the (possibly corrupted) irradiance sensor reading
data["potential_mwh"] = (
    data["asset_id"].map(meta["capacity_mw"]) * (data["irradiance"] / 1000.0)
    * (1 - GAMMA * (data["cell_temp"] - 25))
).clip(lower=0)

# ----------------------------------------------------------------------
# 4. BASELINE: learn PR_baseline from healthy history (leakage-filtered)
# ----------------------------------------------------------------------
DAY_IRR = 150        # only learn PR when there is real sun
data["pr_obs"] = np.where(data.potential_mwh > 1e-6,
                          data.generation_mwh / data.potential_mwh, np.nan)

# "healthy" training mask: daylight, available, not curtailed, plausible PR
healthy = (
    (data.irradiance > DAY_IRR)
    & (data.availability_pct > 90)
    & (data.curtailment_flag == 0)
    & (data.pr_obs.between(0.3, 1.1))
)

# baseline PR per asset = robust median over the FIRST 30 days of healthy ops
train_cutoff = idx[0] + pd.Timedelta(days=30)
train = data[healthy & (data.timestamp < train_cutoff)]
pr_base = train.groupby("asset_id")["pr_obs"].median()
pr_lo = train.groupby("asset_id")["pr_obs"].quantile(0.10)   # lower confidence band
data["pr_baseline"] = data.asset_id.map(pr_base)
data["pr_band_lo"] = data.asset_id.map(pr_lo)

# expected generation
data["expected_mwh"] = data.potential_mwh * data.pr_baseline

# ----------------------------------------------------------------------
# 5. ANOMALY DETECTION: sustained shortfall below the healthy PR band.
#    A persistence test kills isolated single-hour noise: a raw dip only
#    counts if >= PERSIST_MIN of the last PERSIST_WIN daytime hours dip.
#    (Real faults span many hours; p10 noise is scattered & isolated.)
# ----------------------------------------------------------------------
data = data.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
DAY = data.irradiance > DAY_IRR
data["raw_under"] = (
    DAY & (data.pr_obs < data.pr_band_lo) & (data.generation_mwh < data.expected_mwh)
).astype(int)

day_rows = data[DAY].copy()
day_rows["roll"] = (day_rows.groupby("asset_id")["raw_under"]
                    .transform(lambda s: s.rolling(PERSIST_WIN, min_periods=1).sum()))
day_rows["persist"] = (day_rows["roll"] >= PERSIST_MIN) & (day_rows["raw_under"] == 1)
data["persist_under"] = False
data.loc[day_rows.index, "persist_under"] = day_rows["persist"].values

impossible = (~data.pr_obs.between(0.0, 1.15)) & DAY      # physically impossible PR
data["underperf"] = data["persist_under"] | impossible

# peer-group signal: fraction of same-region assets ALSO below expected this hour.
# A high fraction means a shared (regional weather/resource) cause, not an
# asset-specific fault -> lets us separate weather from real plant problems.
data["below_exp"] = ((data.generation_mwh < data.expected_mwh * 0.90)
                     & (data.irradiance > DAY_IRR)).astype(int)
data["regional_dip_frac"] = (data.groupby(["region", "timestamp"])["below_exp"]
                             .transform("mean"))

# ----------------------------------------------------------------------
# 6. CAUSE ATTRIBUTION: ordered rules over the auxiliary signals
# ----------------------------------------------------------------------
# precompute irradiance stability (rolling std per asset) for sensor check
data = data.sort_values(["asset_id", "timestamp"])
data["irr_std6"] = (data.groupby("asset_id")["irradiance"]
                    .transform(lambda s: s.rolling(6, min_periods=3).std()))


def attribute(r):
    if not r.underperf:
        return "OK"
    # 1) data/sensor issue first (don't trust a bad input)
    if (r.pr_obs > 1.15) or (r.pr_obs < 0) or (r.irradiance > 300 and r.irr_std6 < 1.0):
        return "Sensor/Data issue"
    # 2) curtailment / market
    if r.curtailment_flag == 1:
        return "Curtailment / Market"
    # 3) outage
    if (r.availability_pct < 90) or (r.inverter_status == "fault"):
        return "Outage"
    # 4) regional resource/weather: same-region peers dip together (shared cause)
    if r.regional_dip_frac >= 0.5:
        return "Resource/Weather"
    # 5) degradation handled at asset level (trend); else escalate
    return "Unexplained"


data["cause"] = data.apply(attribute, axis=1)

# degradation is a SLOW signal - detect at asset level via PR trend
recent = data[data.timestamp >= idx[-1] - pd.Timedelta(hours=TODAY_WINDOW_H)]
for aid, g in data[healthy.values | (data.cause == "Unexplained")].groupby("asset_id"):
    gg = g.dropna(subset=["pr_obs"])
    if len(gg) < 50:
        continue
    t = (gg.timestamp - gg.timestamp.min()).dt.total_seconds().values
    slope = np.polyfit(t, gg.pr_obs.values, 1)[0] * 3600 * 24 * N_DAYS  # PR change/window
    if slope < -0.08:   # persistent decline
        m_recent_unexpl = (data.asset_id == aid) & (data.cause == "Unexplained")
        data.loc[m_recent_unexpl, "cause"] = "Degradation"

# ----------------------------------------------------------------------
# 7. FINANCIAL IMPACT + DAILY TRIAGE AGGREGATION (last 48h)
# ----------------------------------------------------------------------
NEXT_STEP = {
    "Outage": "Dispatch field tech; pull inverter/string fault logs",
    "Curtailment / Market": "Confirm dispatch instruction; split economic vs. recoverable curtailment",
    "Degradation": "Schedule soiling/module inspection; review long-term PR trend",
    "Sensor/Data issue": "Validate POA sensor calibration; check telemetry pipeline freshness",
    "Resource/Weather": "No action - regional resource event; confirm vs. forecast, no dispatch",
    "Unexplained": "Manual review by performance engineer",
}

MATERIALITY_MWH = 5.0    # suppress trivial noise flags below this 48h loss

recent = data[data.timestamp >= idx[-1] - pd.Timedelta(hours=TODAY_WINDOW_H)].copy()
recent["lost_mwh"] = (recent.expected_mwh - recent.generation_mwh).clip(lower=0)
flagged = recent[recent.cause != "OK"].copy()
# don't count "lost" MWh for sensor issues - the expected number is untrustworthy
flagged.loc[flagged.cause == "Sensor/Data issue", "lost_mwh"] = 0.0
flagged["lost_rev"] = flagged.lost_mwh * flagged.price

agg = flagged.groupby("asset_id").agg(
    lost_mwh_48h=("lost_mwh", "sum"),
    est_lost_revenue=("lost_rev", "sum"),
    flagged_hours=("cause", "size"),
    probable_cause=("cause", lambda s: s.value_counts().index[0]),
).reset_index()

conf = (flagged.groupby("asset_id")["cause"]
        .apply(lambda s: s.value_counts().iloc[0] / len(s)))
# NOTE: this is diagnostic CONSISTENCY (share of flagged hours agreeing with the
# dominant cause), NOT a calibrated probability. Calibration would require
# operator-confirmed labels accumulated via the human-in-the-loop (a v2 step).
agg["diagnostic_consistency"] = agg.asset_id.map(conf).round(2)
# secondary (runner-up) cause, for mixed-signal assets
second = (flagged.groupby("asset_id")["cause"]
          .apply(lambda s: s.value_counts().index[1] if s.nunique() > 1 else None))
agg["secondary_cause"] = agg.asset_id.map(second)
agg = agg.join(meta[["region", "capacity_mw", "contract_type"]], on="asset_id")
agg["next_step"] = agg.probable_cause.map(NEXT_STEP)
# when the diagnosis is ambiguous, tell the operator a second cause is present
lc = (agg.diagnostic_consistency < 0.7) & agg.secondary_cause.notna()
agg.loc[lc, "next_step"] = ("[Mixed: also " + agg.loc[lc, "secondary_cause"]
                            + "] " + agg.loc[lc, "next_step"])

# Resource/Weather is a shared, non-actionable event: pull it OUT of the action
# list and report it as a one-line regional summary (don't dispatch for weather).
resource = agg[agg.probable_cause == "Resource/Weather"].copy()
agg = agg[agg.probable_cause != "Resource/Weather"].copy()

# materiality filter: keep data-quality flags always; suppress trivial noise
keep = (agg.lost_mwh_48h >= MATERIALITY_MWH) | (agg.probable_cause == "Sensor/Data issue")
agg = agg[keep].copy()

# data-quality issues have unknown $ impact -> surface them, don't bury at R$0
agg["flag_type"] = np.where(agg.probable_cause == "Sensor/Data issue",
                            "Data quality", "Performance")
agg["est_lost_revenue"] = np.where(agg.flag_type == "Data quality",
                                   np.nan, agg.est_lost_revenue)
# sort: data-quality first (blind spots), then performance by revenue
agg = agg.sort_values(
    ["flag_type", "est_lost_revenue"], ascending=[True, False]
).reset_index(drop=True)

triage = agg[[
    "asset_id", "region", "capacity_mw", "contract_type", "flag_type",
    "probable_cause", "diagnostic_consistency", "lost_mwh_48h", "est_lost_revenue",
    "flagged_hours", "next_step",
]].copy()
triage["lost_mwh_48h"] = triage.lost_mwh_48h.round(1)
triage["est_lost_revenue"] = triage.est_lost_revenue.round(0)
triage.to_csv("triage_output.csv", index=False)

# non-actionable regional resource summary (collapses many assets into one note)
res_summary = (resource.groupby("region")
               .agg(assets=("capacity_mw", "size"),
                    lost_mwh_48h=("lost_mwh_48h", "sum")).reset_index())
if len(res_summary):
    res_summary["lost_mwh_48h"] = res_summary.lost_mwh_48h.round(1)
    res_summary.to_csv("resource_events.csv", index=False)

print("\n=== GROUND-TRUTH FAULTS INJECTED ===")
for k, v in fault_log.items():
    print(f"  {k}: {v}")
print("\n=== RECOVERY CHECK ===")
for aid, truth in fault_log.items():
    if aid.startswith("REGION:"):
        reg = aid.split(":")[1]
        n = int(resource[resource.region == reg].shape[0]) if len(resource) else 0
        print(f"  {aid}: {n} {reg} assets attributed 'Resource/Weather'  truth={truth!r}")
        continue
    row = triage[triage.asset_id == aid]
    if len(row):
        r = row.iloc[0]
        print(f"  {aid}: predicted={r.probable_cause!r}  "
              f"consistency={r.diagnostic_consistency}  truth={truth!r}")
    else:
        print(f"  {aid}: NOT IN TRIAGE (missed)")
n_unexpl = int((triage.probable_cause == "Unexplained").sum())
print(f"\nUnexplained (false-positive tail) rows in triage: {n_unexpl}")
print("\n=== TOP TRIAGE OUTPUT (last 48h) ===")
print(triage.head(12).to_string(index=False))
if len(res_summary):
    print("\n=== NON-ACTIONABLE RESOURCE/WEATHER EVENTS (excluded from action list) ===")
    print(res_summary.to_string(index=False))

# ----------------------------------------------------------------------
# 8. FIGURES
# ----------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

# Fig 1: example assets - expected band vs actual, flagged points
fig, axes = plt.subplots(2, 3, figsize=(13.5, 6), sharex=True)
examples = [(a_out, "Outage"), (a_deg, "Degradation"), (a_sen, "Sensor/Data issue"),
            (a_cur, "Curtailment / Market"), (a_mix, "Mixed (curtail+outage)")]
axl = axes.ravel()
for ax, (aid, label) in zip(axl, examples):
    g = data[data.asset_id == aid].tail(10 * 24)
    ax.plot(g.timestamp, g.expected_mwh, color="#1d9e75", lw=1.2, label="Expected")
    ax.plot(g.timestamp, g.generation_mwh, color="#534ab7", lw=1.0, label="Actual")
    fl = g[g.cause != "OK"]
    ax.scatter(fl.timestamp, fl.generation_mwh, color="#e24b4a", s=10, zorder=5,
               label="Flagged")
    ax.set_title(f"{aid} - {label}", fontsize=9)
    ax.set_ylabel("MWh")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)
axl[-1].axis("off")  # hide the 6th empty panel
fig.suptitle("Expected (PR baseline) vs. actual generation - last 10 days", fontsize=11)
fig.tight_layout()
fig.savefig("fig_examples.png", dpi=130)

# Fig 2: ranked issue list by estimated lost revenue, colored by cause
cause_colors = {
    "Outage": "#e24b4a", "Curtailment / Market": "#ef9f27",
    "Degradation": "#534ab7", "Sensor/Data issue": "#378add",
    "Unexplained": "#888780",
}
top = triage[triage.flag_type == "Performance"].head(10).iloc[::-1]
fig2, ax2 = plt.subplots(figsize=(9, 4.5))
colors = [cause_colors.get(c, "#888780") for c in top.probable_cause]
ax2.barh(top.asset_id, top.est_lost_revenue, color=colors)
ax2.set_xlabel("Estimated lost revenue, last 48h (R$)")
n_dq = int((triage.flag_type == "Data quality").sum())
ax2.set_title(f"Daily triage - performance issues by financial impact "
              f"(+{n_dq} data-quality flag(s) surfaced separately)")
legend = [Patch(color=v, label=k) for k, v in cause_colors.items()
          if k in top.probable_cause.values]
ax2.legend(handles=legend, fontsize=8, loc="lower right")
fig2.tight_layout()
fig2.savefig("fig_ranking.png", dpi=130)

print("\nSaved: triage_output.csv, fig_examples.png, fig_ranking.png")
