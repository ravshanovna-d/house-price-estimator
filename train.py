"""Train the house price model, evaluate it, and export it for the web demo.

Usage:  python train.py
Outputs: web/model.onnx, web/meta.json, web/assets/*.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from export_utils import check_onnx, examples, export_onnx, feature_schema, write_meta

ROOT = Path(__file__).parent
WEB, ASSETS = ROOT / "web", ROOT / "web" / "assets"
OPENML_ID = 42165  # "house_prices" = Kaggle House Prices train.csv (Ames, Iowa)
SEED = 42
sys.stdout.reconfigure(encoding="utf-8")  # metric names contain "²"

NEIGHBORHOODS = {
    "Blmngtn": "Bloomington Heights", "Blueste": "Bluestem", "BrDale": "Briardale",
    "BrkSide": "Brookside", "ClearCr": "Clear Creek", "CollgCr": "College Creek",
    "Crawfor": "Crawford", "Edwards": "Edwards", "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT and Rail Road", "MeadowV": "Meadow Village", "Mitchel": "Mitchell",
    "NAmes": "North Ames", "NoRidge": "Northridge", "NPkVill": "Northpark Villa",
    "NridgHt": "Northridge Heights", "NWAmes": "Northwest Ames", "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State Univ.", "Sawyer": "Sawyer", "SawyerW": "Sawyer West",
    "Somerst": "Somerset", "StoneBr": "Stone Brook", "Timber": "Timberland", "Veenker": "Veenker",
}
QUALITY = {"Ex": "Excellent", "Gd": "Good", "TA": "Average", "Fa": "Fair", "Po": "Poor"}

NUM_COLS = ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "YearBuilt",
            "YearRemodAdd", "FullBath", "Fireplaces", "LotArea"]
CAT_COLS = ["Neighborhood", "ExterQual", "KitchenQual", "CentralAir"]
FEATURES = NUM_COLS + CAT_COLS
LABELS = {
    "OverallQual": "Overall quality", "GrLivArea": "Living area (above ground)",
    "TotalBsmtSF": "Basement area", "GarageCars": "Garage size", "YearBuilt": "Year built",
    "YearRemodAdd": "Year remodeled", "FullBath": "Full bathrooms", "Fireplaces": "Fireplaces",
    "LotArea": "Lot size", "Neighborhood": "Neighborhood", "ExterQual": "Exterior quality",
    "KitchenQual": "Kitchen quality", "CentralAir": "Central air conditioning",
}


def load_data() -> pd.DataFrame:
    """Download (and cache in data/) the Ames house prices table from OpenML."""
    df = fetch_openml(data_id=OPENML_ID, as_frame=True, data_home=ROOT / "data").frame
    assert df.shape == (1460, 81), df.shape  # same as Kaggle's train.csv
    df["Neighborhood"] = df["Neighborhood"].map(NEIGHBORHOODS)
    df["ExterQual"] = df["ExterQual"].map(QUALITY)
    df["KitchenQual"] = df["KitchenQual"].map(QUALITY)
    df["CentralAir"] = df["CentralAir"].map({"Y": "Yes", "N": "No"})
    for c in CAT_COLS:
        df[c] = df[c].astype(str)
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Two huge, partially built houses sold far below value (documented by the dataset author).
    return df[~((df["GrLivArea"] > 4000) & (df["SalePrice"] < 300_000))]


def build_models():
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    return {
        "Ridge": Pipeline([("prep", prep), ("model", Ridge(alpha=3.0))]),
        "Random Forest": Pipeline([("prep", prep), ("model", RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1))]),
        "Gradient Boosting": Pipeline([("prep", prep), ("model", GradientBoostingRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=3, subsample=0.8, random_state=SEED))]),
        "HistGradientBoosting": Pipeline([("prep", prep), ("model", HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=10,
            l2_regularization=1.0, random_state=SEED))]),
    }


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    df = remove_outliers(raw)
    X, y = df[FEATURES], np.log1p(df["SalePrice"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # 1) Compare models with 5-fold cross-validation (RMSE on the log price, as on Kaggle)
    cv = KFold(5, shuffle=True, random_state=SEED)
    models = build_models()
    rows = []
    for name, model in models.items():
        r = cross_validate(model, X_train, y_train, cv=cv, n_jobs=-1,
                           scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
        rows.append({"model": name, "CV RMSE (log)": float(-r["test_rmse"].mean()),
                     "CV R²": float(r["test_r2"].mean())})
        print(f"{name:22s} RMSE(log)={rows[-1]['CV RMSE (log)']:.4f}  R2={rows[-1]['CV R²']:.3f}")
    comparison = sorted(rows, key=lambda r: r["CV RMSE (log)"])
    best_name = comparison[0]["model"]
    best = models[best_name].fit(X_train, y_train)

    # 2) Final evaluation on the untouched test set (in dollars)
    pred_log = best.predict(X_test)
    price, pred = np.expm1(y_test), np.expm1(pred_log)
    test_metrics = {
        "RMSE ($)": float(np.sqrt(mean_squared_error(price, pred))),
        "MAE ($)": float(mean_absolute_error(price, pred)),
        "R²": float(r2_score(price, pred)),
        "RMSE (log)": float(np.sqrt(mean_squared_error(y_test, pred_log))),
        "Median error": float(np.median(np.abs(pred - price) / price)),
    }
    print("Test:", {k: round(v, 3) for k, v in test_metrics.items()})

    # 3) Figures
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(price / 1000, pred / 1000, s=14, alpha=0.6, color="#4f46e5")
    lim = [0, max(price.max(), pred.max()) / 1000 * 1.05]
    ax.plot(lim, lim, "--", color="grey"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual price ($k)"); ax.set_ylabel("Predicted price ($k)")
    ax.set_title("Predicted vs actual (test set)")
    fig.tight_layout(); fig.savefig(ASSETS / "predicted_vs_actual.png", dpi=130); plt.close(fig)

    imp = permutation_importance(best, X_test, y_test, scoring="neg_root_mean_squared_error",
                                 n_repeats=10, random_state=SEED, n_jobs=-1)
    imp = pd.Series(imp.importances_mean, index=[LABELS[c] for c in X_test.columns]).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    imp.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Increase in RMSE (log) when shuffled")
    ax.set_title("Most important features")
    fig.tight_layout(); fig.savefig(ASSETS / "feature_importance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="OverallQual", y=df["SalePrice"] / 1000, color="#a5b4fc", ax=ax, fliersize=2)
    ax.set_xlabel("Overall quality (1–10)"); ax.set_ylabel("Sale price ($k)")
    ax.set_title("Price grows steeply with overall quality")
    fig.tight_layout(); fig.savefig(ASSETS / "price_by_quality.png", dpi=130); plt.close(fig)

    med = df.groupby("Neighborhood")["SalePrice"].median().sort_values() / 1000
    fig, ax = plt.subplots(figsize=(6, 6))
    med.plot.barh(ax=ax, color="#4f46e5"); ax.set_xlabel("Median sale price ($k)"); ax.set_ylabel("")
    ax.set_title("Median price by neighborhood")
    fig.tight_layout(); fig.savefig(ASSETS / "price_by_neighborhood.png", dpi=130); plt.close(fig)

    # 4) Export for the browser and verify it matches scikit-learn
    export_onnx(best, X_train, WEB / "model.onnx", classifier=False)
    err = check_onnx(WEB / "model.onnx", best, X_test, classifier=False)
    print(f"ONNX check OK (max diff on log price {err:.2e})")

    helps = {"OverallQual": "1 = very poor, 10 = excellent", "GrLivArea": "sq ft",
             "TotalBsmtSF": "sq ft, 0 = no basement", "GarageCars": "cars", "LotArea": "sq ft",
             "YearRemodAdd": "same as built if never"}
    schema = feature_schema(X_train, LABELS, helps)
    for f in schema:  # show quality options from best to worst
        if f["name"] in ("ExterQual", "KitchenQual"):
            f["options"] = [q for q in QUALITY.values() if q in f["options"]]
    write_meta(WEB / "meta.json", {
        "title": "House Price Estimator",
        "subtitle": ("Estimates the sale price of a house in Ames, Iowa from a dozen facts a buyer "
                     "or agent already knows: size, quality, age, garage and neighborhood."),
        "tags": ["Regression", "scikit-learn", best_name, "Cross-validation", "ONNX"],
        "task": "regression",
        "target_transform": "log1p",
        "output_label": "Estimated sale price",
        "output_format": "currency",
        "result_title": "Estimate",
        "result_note": (f"Typical error is about {test_metrics['Median error'] * 100:.0f}% of the price. "
                        "Try raising Overall quality or changing the Neighborhood."),
        "features": schema,
        "examples": examples(X_test),
        "model_name": best_name,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metric_formats": {"RMSE ($)": "currency", "MAE ($)": "currency", "Median error": "percent"},
        "metrics_note": f"Held-out test set: {len(X_test)} houses never seen during training or model selection.",
        "figures": [
            {"src": "assets/predicted_vs_actual.png", "caption": "Predictions sit close to the diagonal across the price range."},
            {"src": "assets/feature_importance.png", "caption": "Living area and overall quality drive the estimate the most."},
            {"src": "assets/price_by_quality.png", "caption": "Each step in overall quality adds a lot to the price."},
            {"src": "assets/price_by_neighborhood.png", "caption": "Location matters: median prices differ more than 3×."},
        ],
        "steps": [
            f"Loaded the Ames house sales data ({len(raw):,} homes, 79 features) and removed 2 documented outliers.",
            f"Picked {len(FEATURES)} understandable features (quality, size, age, garage, neighborhood…) for the form.",
            "Predicted log(price) so that errors are relative — a $20k miss matters more on a $100k house.",
            "Compared Ridge, Random Forest, Gradient Boosting and HistGradientBoosting with 5-fold cross-validation.",
            f"Picked {best_name} (lowest CV RMSE) and evaluated it once on a held-out 20% test set.",
            "Exported the whole pipeline (scaling + one-hot encoding + model) to ONNX so it runs in the browser.",
        ],
        "dataset": {"name": "Ames House Prices (Kaggle / OpenML 42165)",
                    "url": f"https://www.openml.org/d/{OPENML_ID}", "rows": len(raw)},
        "repo_url": "https://github.com/ravshanovna-d/house-price-estimator",
        "notebook_url": "https://github.com/ravshanovna-d/house-price-estimator/blob/main/notebook.ipynb",
        "footer": "Built by <a href=\"https://github.com/ravshanovna-d\">ravshanovna-d</a>",
    })
    print("Wrote web/model.onnx and web/meta.json")


if __name__ == "__main__":
    main()
