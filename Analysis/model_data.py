"""Shared feature matrix and train/test split for all classifiers."""
import pandas as pd
from sklearn.model_selection import train_test_split
from .common import FEATURED_DATA


def get_data(balance=False, seed=42):
    if not FEATURED_DATA.is_file():
        raise FileNotFoundError(
            f"Featured dataset not found: {FEATURED_DATA}\n"
            "Run `python Analysis/10_build_features.py` from the project root first."
        )
    df = pd.read_csv(FEATURED_DATA)
    if "target" not in df:
        raise ValueError(
            f"{FEATURED_DATA.name} has no 'target' column. Rebuild it with "
            "`python Analysis/10_build_features.py`."
        )
    features = [c for c in df.columns if c.startswith("roll")]
    features += [c for c in ["streak", "days_rest", "opp_def_r5", "opp_xg_att_r5", "is_home"] if c in df]
    features = list(dict.fromkeys(c for c in features if c in df))
    X = df[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median()).fillna(0)
    y = df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    if balance:
        from imblearn.over_sampling import SMOTE
        X_train, y_train = SMOTE(random_state=seed).fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test, features
