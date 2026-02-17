"""
=============================================================================
Model Trainer - Phase 2
=============================================================================
Walk-forward training and auto-selection of the best ML model for P(win)
prediction. Tests 4 algorithms, selects by EV + Sharpe.

Models tested:
  1. Logistic Regression (baseline)
  2. Random Forest
  3. LightGBM
  4. XGBoost

Walk-forward: 4 months train / 1 month test (rolling)
Selection: Best EV + Sharpe > 1.2 (fallback > 0.8)
Output: best_model.pkl + model_report.json

Usage:
    python scripts/model_trainer.py                    # Full train + select
    python scripts/model_trainer.py --retrain           # Weekly retrain
    python scripts/model_trainer.py --evaluate-only     # Evaluate existing model
=============================================================================
"""

import os
import json
import argparse
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from loguru import logger

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

from scripts.kimi_client import kimi_interpret
from scripts.data_pipeline import compute_features, DATA_DIR, DB_PATH, INSTRUMENTS

from dotenv import load_dotenv

load_dotenv()

MODEL_DIR = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# RULE: Walk-forward 4-month train, 1-month test.
# RULE: Auto-test Logistic, RF, LightGBM, XGBoost.
# RULE: Select best by EV + Sharpe > 1.2 (fallback > 0.8).
# ---------------------------------------------------------------------------

TRAIN_MONTHS = 4
TEST_MONTHS = 1

# Features defined in config/strategy_params.json
FEATURE_COLS = [
    "rsi_14", "atr_14", "vwap_distance_pct", "volume_ratio_20",
    "bb_width_20", "macd_histogram", "adx_14",
    "hour_of_day", "day_of_week",
    "prev_close_return", "open_eq_low", "open_eq_high",
    "vix_level", "news_sentiment_score", "oi_change_pct",
]

# Target: 1 if trade would have been profitable (close > entry after N bars)
TARGET_COL = "target_win"
LOOKAHEAD_BARS = 6  # 6 x 5-min = 30 min lookahead for P(win) label


# =============================================================================
# Label Generation
# =============================================================================
def generate_labels(df: pd.DataFrame, lookahead: int = LOOKAHEAD_BARS) -> pd.DataFrame:
    """
    Create binary target: 1 if price moves up by > 0.08% (cost threshold)
    within the next `lookahead` bars, 0 otherwise.

    RULE: EV calculation accounts for ~0.08% costs/slippage.
    """
    cost_threshold = 0.0008  # 0.08% round-trip costs

    future_max = df["close"].shift(-lookahead).rolling(lookahead).max()
    df[TARGET_COL] = (
        (future_max / df["close"] - 1) > cost_threshold
    ).astype(int)

    # Drop rows where we can't compute the target
    df.dropna(subset=[TARGET_COL], inplace=True)
    return df


# =============================================================================
# Walk-Forward Split
# =============================================================================
def walk_forward_splits(
    df: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward train/test splits.

    RULE: 4-month train, 1-month test, rolling forward.
    Returns list of (train_df, test_df) tuples.
    """
    if df.empty or not hasattr(df.index, "date"):
        logger.warning("DataFrame empty or no datetime index for walk-forward split")
        return []

    # Strip timezone from index to avoid tz-aware vs tz-naive comparison errors
    # (Kite returns tz-aware datetimes with pytz.FixedOffset(330) for IST)
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)

    dates = sorted(df.index.date)
    if not dates:
        return []

    min_date = pd.Timestamp(dates[0])
    max_date = pd.Timestamp(dates[-1])

    splits = []
    current_train_start = min_date

    while True:
        train_end = current_train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > max_date:
            break

        train_mask = (df.index >= current_train_start) & (df.index < train_end)
        test_mask = (df.index >= train_end) & (df.index < test_end)

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(train_df) > 100 and len(test_df) > 20:
            splits.append((train_df, test_df))
            logger.info(
                f"  Split: train {current_train_start.date()} to {train_end.date()} "
                f"({len(train_df)} rows) | test to {test_end.date()} ({len(test_df)} rows)"
            )

        # Slide forward by 1 month
        current_train_start += pd.DateOffset(months=1)

    return splits


# =============================================================================
# Model Definitions
# =============================================================================
def get_models() -> dict:
    """
    Return dict of model_name -> sklearn-compatible estimator.

    RULE: Auto-test Logistic Regression, Random Forest, LightGBM, XGBoost.
    """
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=0.1, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, verbose=-1
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=1.0,  # Auto-balance later
            random_state=42, eval_metric="logloss",
            use_label_encoder=False,
        ),
    }


# =============================================================================
# Evaluation Metrics
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute trading-relevant metrics.

    RULE: Select by EV + Sharpe > 1.2.
    EV = P(win) * avg_win - P(loss) * avg_loss (simplified as accuracy-based).
    Sharpe approximated from prediction confidence distribution.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Trading-specific metrics
    # P(win) from model predictions on positive class
    p_win = y_prob[y_pred == 1].mean() if (y_pred == 1).any() else 0.0

    # Simplified EV: P(win)*1R - P(loss)*1R = 2*accuracy - 1 (scaled)
    # More accurate: use actual P/L from backtest, but this is model evaluation stage
    win_rate = precision  # Precision = how often we're right when we predict positive
    loss_rate = 1 - win_rate
    avg_win_r = 1.0   # Assume 1R target
    avg_loss_r = 1.0   # Assume 1R stop loss

    ev = win_rate * avg_win_r - loss_rate * avg_loss_r

    # Sharpe approximation from prediction returns
    # Each correct prediction = +1R, incorrect = -1R
    simulated_returns = np.where(y_pred == y_true, 1.0, -1.0)
    sharpe = (
        simulated_returns.mean() / (simulated_returns.std() + 1e-8)
        * np.sqrt(252)  # Annualize (trading days)
    )

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "win_rate": round(win_rate, 4),
        "ev": round(ev, 4),
        "sharpe": round(sharpe, 4),
        "p_win_avg": round(p_win, 4),
        "n_predictions": int(y_pred.sum()),
        "n_total": len(y_true),
    }


# =============================================================================
# Training Pipeline
# =============================================================================
def load_training_data() -> pd.DataFrame:
    """Load all feature data from parquet files."""
    all_dfs = []

    feature_files = list(DATA_DIR.glob("features_*.parquet"))
    if not feature_files:
        logger.warning("No feature files found. Run: python scripts/data_pipeline.py --mode features")
        return pd.DataFrame()

    for fp in feature_files:
        df = pd.read_parquet(fp)
        symbol = fp.stem.replace("features_", "")
        df["symbol"] = symbol
        all_dfs.append(df)

    combined = pd.concat(all_dfs, axis=0)
    logger.info(f"Loaded {len(combined)} feature rows from {len(feature_files)} files")
    return combined


def train_and_evaluate() -> dict:
    """
    Full training pipeline: load data, walk-forward train 4 models, select best.

    Returns dict with best model info and all model metrics.
    """
    logger.info("=== MODEL TRAINING: Walk-Forward Pipeline ===")

    # Load data
    df = load_training_data()
    if df.empty:
        return {"error": "No training data available"}

    # Generate labels
    df = generate_labels(df)

    # Filter to only feature columns that exist
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    if len(available_features) < 5:
        return {"error": f"Only {len(available_features)} features available, need >= 5"}

    logger.info(f"Using {len(available_features)} features: {available_features}")

    # Walk-forward splits
    splits = walk_forward_splits(df)
    if not splits:
        return {"error": "Not enough data for walk-forward splits"}

    models = get_models()
    all_results = {}

    for model_name, model in models.items():
        logger.info(f"\n--- Training {model_name} ---")
        split_metrics = []

        for i, (train_df, test_df) in enumerate(splits):
            X_train = train_df[available_features].fillna(0)
            y_train = train_df[TARGET_COL]
            X_test = test_df[available_features].fillna(0)
            y_test = test_df[TARGET_COL]

            # Scale features (important for Logistic Regression)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]

                metrics = compute_metrics(y_test.values, y_pred, y_prob)
                split_metrics.append(metrics)

                logger.info(
                    f"  Split {i}: acc={metrics['accuracy']:.3f} "
                    f"sharpe={metrics['sharpe']:.2f} ev={metrics['ev']:.3f}"
                )
            except Exception as e:
                logger.error(f"  Split {i} failed for {model_name}: {e}")

        if split_metrics:
            avg_metrics = {
                key: round(np.mean([m[key] for m in split_metrics]), 4)
                for key in split_metrics[0].keys()
                if isinstance(split_metrics[0][key], (int, float))
            }
            all_results[model_name] = {
                "avg_metrics": avg_metrics,
                "split_metrics": split_metrics,
                "n_splits": len(split_metrics),
            }
            logger.info(
                f"\n  {model_name} AVG: sharpe={avg_metrics['sharpe']:.2f} "
                f"ev={avg_metrics['ev']:.3f} acc={avg_metrics['accuracy']:.3f}"
            )

    if not all_results:
        return {"error": "All models failed"}

    # --- Select Best Model ---
    # RULE: Choose best by EV + Sharpe > 1.2. Fallback if no model hits 1.2.
    best_model_name = None
    best_score = -999

    for model_name, result in all_results.items():
        sharpe = result["avg_metrics"]["sharpe"]
        ev = result["avg_metrics"]["ev"]
        # Combined score: sharpe + ev (both important)
        score = sharpe + ev

        if sharpe >= 1.2 and score > best_score:
            best_model_name = model_name
            best_score = score

    # Fallback: Sharpe > 0.8
    if best_model_name is None:
        for model_name, result in all_results.items():
            sharpe = result["avg_metrics"]["sharpe"]
            ev = result["avg_metrics"]["ev"]
            score = sharpe + ev
            if sharpe >= 0.8 and score > best_score:
                best_model_name = model_name
                best_score = score

    # Last resort: best overall
    if best_model_name is None:
        best_model_name = max(
            all_results,
            key=lambda k: all_results[k]["avg_metrics"]["sharpe"]
            + all_results[k]["avg_metrics"]["ev"],
        )
        logger.warning("No model hit Sharpe threshold. Using best available.")

    logger.info(f"\n=== BEST MODEL: {best_model_name} (score={best_score:.3f}) ===")

    # Retrain best model on ALL data for production use
    logger.info("Retraining best model on full dataset for production...")
    X_all = df[available_features].fillna(0)
    y_all = df[TARGET_COL]

    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)

    final_model = get_models()[best_model_name]
    final_model.fit(X_all_scaled, y_all)

    # Save model + scaler
    model_path = MODEL_DIR / "best_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Model saved to {model_path}")

    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "best_score": best_score,
        "features": available_features,
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(df),
        "all_results": {
            k: v["avg_metrics"] for k, v in all_results.items()
        },
    }
    meta_path = MODEL_DIR / "model_report.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Report saved to {meta_path}")

    # Kimi 2.5 interpretation (optional, for daily brief)
    try:
        interpretation = kimi_interpret(metadata)
        metadata["kimi_interpretation"] = interpretation
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Kimi 2.5 interpretation:\n{interpretation}")
    except Exception as e:
        logger.warning(f"Kimi interpretation skipped: {e}")

    return metadata


# =============================================================================
# Inference (used by strategy_engine.py)
# =============================================================================
def load_model():
    """Load the best trained model + scaler for inference."""
    model_path = MODEL_DIR / "best_model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    meta_path = MODEL_DIR / "model_report.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No trained model at {model_path}. Run model_trainer.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    with open(meta_path) as f:
        metadata = json.load(f)

    return model, scaler, metadata


def predict(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on feature DataFrame.

    Returns DataFrame with added columns: 'p_win', 'prediction', 'regime_label'.
    """
    model, scaler, metadata = load_model()
    feature_cols = metadata["features"]

    available = [c for c in feature_cols if c in features_df.columns]
    X = features_df[available].fillna(0)
    X_scaled = scaler.transform(X)

    features_df["p_win"] = model.predict_proba(X_scaled)[:, 1]
    features_df["prediction"] = model.predict(X_scaled)

    # Regime detection: simple volatility-based classification
    # 0 = neutral, 1 = trending (high ADX), 2 = ranging (low ADX)
    if "adx_14" in features_df.columns:
        features_df["regime_label"] = np.where(
            features_df["adx_14"] > 25, 1,  # Trending
            np.where(features_df["adx_14"] < 15, 2, 0)  # Ranging / Neutral
        )
    else:
        features_df["regime_label"] = 0

    return features_df


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="NSE Paper Trading - Model Trainer")
    parser.add_argument("--retrain", action="store_true", help="Retrain on latest data")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate existing model")
    args = parser.parse_args()

    if args.evaluate_only:
        model, scaler, meta = load_model()
        logger.info(f"Current model: {meta['best_model']}")
        logger.info(f"Metrics: {json.dumps(meta['all_results'], indent=2)}")
    else:
        result = train_and_evaluate()
        if "error" in result:
            logger.error(f"Training failed: {result['error']}")
            sys.exit(1)
        logger.info(f"Training complete. Best: {result['best_model']}")


if __name__ == "__main__":
    import sys
    main()
