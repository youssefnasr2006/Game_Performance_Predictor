"""Create leakage-safe shifted rolling match features from verified data."""
import numpy as np
from common import FEATURED_DATA, load_verified

STAT_COLS = ["Corner kicks", "Expected goals (xG)", "Ball possession", "Shots off target", "Total shots", "Shots on target", "Yellow cards", "Red cards", "Fouls", "Goals for", "Goals against"]
WINDOWS = [3, 5, 10]

df = load_verified()
df["target"] = df["result"].map({"W": 2, "D": 1, "L": 0})
df["points"] = df["result"].map({"W": 3, "D": 1, "L": 0})
df["is_home"] = df["venue"].str.strip().str.lower().eq("home").astype(int)
for col in STAT_COLS:
    if col in df:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")
df["goal_diff"] = df["Goals for"] - df["Goals against"]
df["clean_sheet"] = df["Goals against"].eq(0).astype(float)
df["win_flag"] = df["result"].eq("W").astype(float)
df["draw_flag"] = df["result"].eq("D").astype(float)
df["loss_flag"] = df["result"].eq("L").astype(float)
df["xG_diff"] = df["Expected goals (xG)"] - df["Goals against"]
df["conversion_rate"] = df["Goals for"] / df["Total shots"].replace(0, np.nan)
df["shots_ratio"] = df["Shots on target"] / df["Total shots"].replace(0, np.nan)
df["days_rest"] = df.groupby("Team")["date"].diff().dt.days
df = df.dropna(subset=["date", "target"]).copy()

def prior_mean(series, window):
    return series.shift(1).rolling(window, min_periods=1).mean()

for col in STAT_COLS:
    if col in df:
        safe = col.replace(" ", "_").replace("(", "").replace(")", "")
        for window in WINDOWS:
            df[f"roll{window}_{safe}"] = df.groupby("Team")[col].transform(lambda s, w=window: prior_mean(s, w))

for name, source in {"points":"points", "goal_diff":"goal_diff", "clean_sheet":"clean_sheet", "win_rate":"win_flag", "draw_rate":"draw_flag", "loss_rate":"loss_flag", "xG_diff":"xG_diff", "conversion_rate":"conversion_rate", "shots_ratio":"shots_ratio"}.items():
    for window in WINDOWS:
        df[f"roll{window}_{name}"] = df.groupby("Team")[source].transform(lambda s, w=window: prior_mean(s, w))

def streak(series):
    values, count = [], 0
    for value in series:
        count = count + 1 if value == 2 and count > 0 else 1 if value == 2 else count - 1 if value == 0 and count < 0 else -1 if value == 0 else 0
        values.append(count)
    return pd.Series(values, index=series.index)

df["streak_raw"] = df.groupby("Team")["target"].transform(streak)
df["streak"] = df.groupby("Team")["streak_raw"].shift(1)
df["days_rest"] = df["days_rest"].fillna(7)
opponent_history = df[["Team", "date"]].copy()
opponent_history["opp_def_r5"] = df.groupby("Team")["Goals against"].transform(lambda s: prior_mean(s, 5))
opponent_history["opp_xg_att_r5"] = df.groupby("Team")["Expected goals (xG)"].transform(lambda s: prior_mean(s, 5))
opponent_history = opponent_history.rename(columns={"Team": "Opponent"})
df = df.merge(opponent_history, on=["Opponent", "date"], how="left")
for venue_value, label in [(1, "home"), (0, "away")]:
    mask = df["is_home"].eq(venue_value)
    for source, name in [("Goals for", "gf"), ("Goals against", "ga"), ("points", "pts")]:
        df[f"roll5_{name}_{label}"] = df.groupby("Team")[source].transform(lambda s: s.where(mask).shift(1).rolling(5, min_periods=1).mean())

FEATURED_DATA.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(FEATURED_DATA, index=False)
print(f"Saved {len(df):,} rows to {FEATURED_DATA}")
