"""ML-based asset lifecycle extrapolation for infrastructure assets.

Approach
--------
Production energy-asset datasets are proprietary and unavailable in a public
portfolio repo.  This module therefore uses a *knowledge-encoded synthetic
training dataset* — a technique common in industrial ML when labelled ground-
truth is scarce.  Domain expertise (sector-typical lifespans, geographic
capacity factors, economic-level adjustments) is encoded as the data-generation
rules, and scikit-learn models learn the feature-to-outcome mappings from 300+
synthetic reference assets that are statistically consistent with real-world
energy-infrastructure literature.

Three models are trained and applied to every asset in the silver asset_master:

  1. ``lifecycle_stage_clf``  — RandomForestClassifier
       Predicts which stage of its operational lifecycle an asset is in:
       planning | construction | operating | decommissioning | retired

  2. ``retirement_year_reg``  — GradientBoostingRegressor
       Predicts the calendar year in which the asset is most likely to be
       decommissioned, based on sector lifespan norms, geographic durability
       factors, and economic-level maintenance capacity.

  3. ``capacity_factor_reg``  — RandomForestRegressor
       Predicts the annual energy output factor (0–1), i.e. actual output
       divided by theoretical maximum, driven by local solar irradiance,
       wind speed, altitude, and regulatory environment.

All models are trained fresh on every pipeline run (appropriate for demo scale).
A production deployment would persist fitted models as artefacts.

Feature inputs
--------------
Per asset, the feature vector combines:
  - asset attributes: capacity_mw, sector_encoded
  - geographic enrichment (from reference_data/country_attributes.csv):
      latitude, longitude, altitude_avg_m, territorial_type_encoded,
      gdp_tier, solar_irradiance_kwh_m2_yr, wind_speed_avg_ms,
      regulatory_stability_score
  - lifecycle signal (from gold ownership_lifecycle):
      total_appearances, presence_rate, reliability_score,
      snapshot_count_available, consecutive_appearances_current

Outputs
-------
``gold/dw/asset_lifecycle_predictions.parquet`` — one row per asset, all
prediction outputs plus the feature inputs for full explainability.  Also
written to ``gold/entity_lakehouse.duckdb`` as ``ml_asset_lifecycle_predictions``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.preprocessing import LabelEncoder

from .contracts import validate_dataframe
from .observability import get_langfuse

logger = logging.getLogger(__name__)

# Model version tag written to every prediction row so consumers can track
# which training configuration produced a result.
_MODEL_VERSION = "v1.1-synthetic-300"

# Lifecycle stage labels and their approximate age-fraction boundaries relative
# to the asset's typical lifespan.
_LIFECYCLE_STAGES = [
    "planning",
    "construction",
    "operating",
    "decommissioning",
    "retired",
]

# Columns that form the feature matrix for all three models.
_FEATURE_COLS = [
    "capacity_mw",
    "sector_encoded",
    "latitude",
    "longitude",
    "altitude_avg_m",
    "territorial_type_encoded",
    "gdp_tier",
    "solar_irradiance_kwh_m2_yr",
    "wind_speed_avg_ms",
    "regulatory_stability_score",
    "total_appearances",
    "presence_rate",
    "reliability_score",
    "typical_lifespan_years",
]

# Baseline calendar year used as a reference anchor when commissioning year is
# unknown (treated as the midpoint of the asset's observed snapshot range).
_REFERENCE_YEAR = 2025


# ---------------------------------------------------------------------------
# Reference data loaders
# ---------------------------------------------------------------------------


def _load_country_attributes(reference_root: Path) -> dict[str, dict]:
    """Return a dict keyed by country_code with all attribute columns."""
    path = reference_root / "country_attributes.csv"
    df = pd.read_csv(path, dtype={"country_code": str})
    return {row["country_code"]: row.to_dict() for _, row in df.iterrows()}


def _load_sector_lifecycle(reference_root: Path) -> dict[str, dict]:
    """Return a dict keyed by sector with lifecycle parameter columns."""
    path = reference_root / "sector_lifecycle.csv"
    df = pd.read_csv(path, dtype={"sector": str})
    return {row["sector"]: row.to_dict() for _, row in df.iterrows()}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

_TERRITORIAL_TYPE_ENCODING = {
    "island": 1,
    "coastal": 2,
    "mixed": 3,
    "inland": 4,
}

_ECONOMIC_LEVEL_ENCODING = {
    "low": 1,
    "lower_middle": 2,
    "upper_middle": 3,
    "high": 4,
}

_SECTOR_ENCODING = {
    "solar": 1,
    "wind": 2,
    "storage": 3,
}


def _encode_territorial_type(value: str) -> int:
    return _TERRITORIAL_TYPE_ENCODING.get(value, 3)


def _build_sector_encoding(sector_params: dict[str, dict]) -> dict[str, int]:
    """Build a stable sector-to-integer mapping derived from the loaded sector params.

    Sectors are sorted alphabetically and assigned codes starting at 1, making
    the mapping deterministic and consistent between synthetic training data
    generation and real-asset feature enrichment.

    Because this function derives the encoding from sector_params rather than a
    hard-coded dict, adding a new sector row to reference_data/sector_lifecycle.csv
    with a training_weight value is sufficient to include it in both encoding
    and training without touching ml.py.
    """
    return {sector: idx + 1 for idx, sector in enumerate(sorted(sector_params))}


def _encode_sector(value: str, encoding: dict[str, int]) -> int:
    """Look up the integer code for a sector.  Returns 0 for unknown sectors."""
    return encoding.get(value, 0)


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------


def _enrich_asset_features(
    asset_master: pd.DataFrame,
    ownership_lifecycle: pd.DataFrame,
    country_attrs: dict[str, dict],
    sector_params: dict[str, dict],
) -> pd.DataFrame:
    """Join assets with geographic, economic, and lifecycle signal features.

    Unknown countries fall back to global medians so predictions still run.
    Assets missing from ownership_lifecycle (no ownership history) receive
    neutral lifecycle signal values.
    """
    # Compute global medians for unknown-country fallback.
    all_latitudes = [v["latitude_centroid"] for v in country_attrs.values()]
    all_longitudes = [v["longitude_centroid"] for v in country_attrs.values()]
    all_altitudes = [v["altitude_avg_m"] for v in country_attrs.values()]
    all_irradiances = [v["solar_irradiance_kwh_m2_yr"] for v in country_attrs.values()]
    all_winds = [v["wind_speed_avg_ms"] for v in country_attrs.values()]
    all_gdp_tiers = [v["gdp_tier"] for v in country_attrs.values()]
    all_reg_scores = [v["regulatory_stability_score"] for v in country_attrs.values()]
    fallback_country = {
        "latitude_centroid": float(np.median(all_latitudes)),
        "longitude_centroid": float(np.median(all_longitudes)),
        "altitude_avg_m": float(np.median(all_altitudes)),
        "territorial_type": "mixed",
        "economic_level": "upper_middle",
        "gdp_tier": int(np.median(all_gdp_tiers)),
        "solar_irradiance_kwh_m2_yr": float(np.median(all_irradiances)),
        "wind_speed_avg_ms": float(np.median(all_winds)),
        "regulatory_stability_score": float(np.median(all_reg_scores)),
    }

    # Build per-lifecycle-key summary from ownership_lifecycle.
    lifecycle_by_asset: dict[str, dict] = {}
    for _, row in ownership_lifecycle.iterrows():
        asset_id = row["asset_id"]
        existing = lifecycle_by_asset.get(asset_id)
        if existing is None or row["reliability_score"] > existing["reliability_score"]:
            lifecycle_by_asset[asset_id] = {
                "total_appearances": int(row["total_appearances"]),
                "presence_rate": float(row["presence_rate"]),
                "reliability_score": float(row["reliability_score"]),
                "consecutive_appearances_current": int(
                    row["consecutive_appearances_current"]
                ),
            }

    rows: list[dict] = []
    sector_encoding = _build_sector_encoding(sector_params)
    for _, asset in asset_master.iterrows():
        country_code = str(asset["asset_country"])
        sector = str(asset["asset_sector"])

        geo = country_attrs.get(country_code, fallback_country)
        if sector not in sector_params:
            supported = sorted(sector_params)
            raise ValueError(
                f"Unsupported asset sector '{sector}' for asset '{asset['asset_id']}'. "
                f"Supported sectors: {supported}. "
                "To add a new sector, add a row to reference_data/sector_lifecycle.csv "
                "with a training_weight value — the encoding and training weights are "
                "derived from that file automatically."
            )
        sp = sector_params[sector]
        lc = lifecycle_by_asset.get(
            str(asset["asset_id"]),
            {
                "total_appearances": 1,
                "presence_rate": 1.0,
                "reliability_score": 0.5,
                "consecutive_appearances_current": 1,
            },
        )

        rows.append(
            {
                "asset_id": str(asset["asset_id"]),
                "asset_name": str(asset["asset_name"]),
                "asset_country": country_code,
                "asset_sector": sector,
                "capacity_mw": float(asset["capacity_mw"]),
                "sector_encoded": _encode_sector(sector, sector_encoding),
                # Geographic features
                "latitude": float(geo["latitude_centroid"]),
                "longitude": float(geo["longitude_centroid"]),
                "altitude_avg_m": float(geo["altitude_avg_m"]),
                "territorial_type": str(geo["territorial_type"]),
                "territorial_type_encoded": _encode_territorial_type(
                    str(geo["territorial_type"])
                ),
                "economic_level": str(geo["economic_level"]),
                "gdp_tier": int(geo["gdp_tier"]),
                "solar_irradiance_kwh_m2_yr": float(geo["solar_irradiance_kwh_m2_yr"]),
                "wind_speed_avg_ms": float(geo["wind_speed_avg_ms"]),
                "regulatory_stability_score": float(geo["regulatory_stability_score"]),
                # Sector lifecycle parameters
                "typical_lifespan_years": float(sp.get("typical_lifespan_years", 25)),
                # Ownership lifecycle signal
                "total_appearances": int(lc["total_appearances"]),
                "presence_rate": float(lc["presence_rate"]),
                "reliability_score": float(lc["reliability_score"]),
                "consecutive_appearances_current": int(
                    lc["consecutive_appearances_current"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic training data generation
# ---------------------------------------------------------------------------


def _generate_synthetic_training_data(
    country_attrs: dict[str, dict],
    sector_params: dict[str, dict],
    n_samples: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labelled synthetic training dataset encoding domain knowledge.

    Each row represents a plausible reference energy infrastructure asset.
    Labels are derived deterministically from the feature values plus controlled
    noise — they are not arbitrary random values but reflect real-world
    relationships documented in energy transition literature.

    Sector distribution is driven by the training_weight column in
    sector_lifecycle.csv.  Weights are normalised to sum to 1.0, so adding a
    new sector row with a training_weight value automatically includes it in
    training without changes to this function.

    Commissioning years span 2000-2030 so future-commissioned assets
    (commissioning_year > _REFERENCE_YEAR) are included and the classifier
    learns the planning label.
    """
    rng = np.random.default_rng(seed)

    # Derive sectors and normalised weights from sector_params so that adding a
    # new row to sector_lifecycle.csv is sufficient to include it in training.
    sector_encoding = _build_sector_encoding(sector_params)
    sectors = sorted(sector_params.keys())  # consistent with _build_sector_encoding
    raw_weights = [float(sector_params[s].get("training_weight", 1.0)) for s in sectors]
    total_weight = sum(raw_weights)
    sector_weights = [w / total_weight for w in raw_weights]

    country_codes = list(country_attrs.keys())
    country_list = [country_attrs[cc] for cc in country_codes]

    rows: list[dict] = []

    for _ in range(n_samples):
        # --- Sample sector and country ---
        sector = rng.choice(sectors, p=sector_weights)
        country_idx = rng.integers(0, len(country_list))
        geo = country_list[country_idx]
        sp = sector_params[sector]

        capacity_mw = float(
            rng.choice([35, 50, 90, 100, 120, 150, 200, 220, 300, 500])
            * (1.0 + rng.normal(0, 0.05))
        )
        capacity_mw = max(1.0, round(capacity_mw, 1))

        # --- Commissioning year: 2000-2030 so future-commissioned assets
        # (commissioning_year > _REFERENCE_YEAR) are included and the
        # classifier learns the planning label.  rng.integers is upper-bound
        # exclusive, so 2031 produces values up to and including 2030. ---
        commissioning_year = int(rng.integers(2000, 2031))

        # --- Lifespan: sector typical + geographic/economic adjustment ---
        base_lifespan = float(sp["typical_lifespan_years"])
        economic_bonus_col = f"economic_level_bonus_{geo['economic_level']}"
        economic_bonus = float(sp.get(economic_bonus_col, 0.0)) * 5.0  # scale to years
        altitude_adj = (
            float(geo["altitude_avg_m"])
            * float(sp.get("altitude_sensitivity", 0.0))
            * 10.0
        )
        lifespan_noise = rng.normal(0, 1.5)
        operational_lifespan = max(
            float(sp["min_lifespan_years"]),
            min(
                float(sp["max_lifespan_years"]),
                base_lifespan + economic_bonus + altitude_adj + lifespan_noise,
            ),
        )

        retirement_year = commissioning_year + int(round(operational_lifespan))
        asset_age_years = _REFERENCE_YEAR - commissioning_year
        total_lifespan = retirement_year - commissioning_year

        # --- Lifecycle stage: derived from asset age vs. total lifespan ---
        if commissioning_year > _REFERENCE_YEAR:
            lifecycle_stage = "planning"
        elif asset_age_years < float(sp["construction_years"]):
            lifecycle_stage = "construction"
        elif retirement_year <= _REFERENCE_YEAR:
            lifecycle_stage = "retired"
        elif asset_age_years >= total_lifespan - float(sp["decommissioning_years"]):
            lifecycle_stage = "decommissioning"
        else:
            lifecycle_stage = "operating"

        # --- Capacity factor: sector base + geographic modifiers ---
        base_cf = float(sp["base_capacity_factor"])
        irradiance_adj = float(geo["solar_irradiance_kwh_m2_yr"]) * float(
            sp.get("irradiance_sensitivity", 0.0)
        )
        wind_adj = float(geo["wind_speed_avg_ms"]) * float(
            sp.get("wind_sensitivity", 0.0)
        )
        altitude_cf_adj = float(geo["altitude_avg_m"]) * float(
            sp.get("altitude_sensitivity", 0.0)
        )
        econ_cf_bonus = float(sp.get(economic_bonus_col, 0.0))
        cf_noise = rng.normal(0, 0.015)
        capacity_factor = float(
            np.clip(
                base_cf
                + irradiance_adj
                + wind_adj
                + altitude_cf_adj
                + econ_cf_bonus
                + cf_noise,
                0.03,
                0.80,
            )
        )

        # --- Simulate lifecycle signal (presence/reliability) ---
        years_observed = min(3, max(1, _REFERENCE_YEAR - commissioning_year))
        total_appearances = rng.integers(1, years_observed + 1)
        presence_rate = float(total_appearances) / float(years_observed)
        reliability_score = float(
            np.clip(presence_rate * (0.8 + rng.random() * 0.2), 0.1, 1.0)
        )

        rows.append(
            {
                # Features
                "capacity_mw": capacity_mw,
                "sector_encoded": _encode_sector(sector, sector_encoding),
                "latitude": float(geo["latitude_centroid"]),
                "longitude": float(geo["longitude_centroid"]),
                "altitude_avg_m": float(geo["altitude_avg_m"]),
                "territorial_type_encoded": _encode_territorial_type(
                    str(geo["territorial_type"])
                ),
                "gdp_tier": int(geo["gdp_tier"]),
                "solar_irradiance_kwh_m2_yr": float(geo["solar_irradiance_kwh_m2_yr"]),
                "wind_speed_avg_ms": float(geo["wind_speed_avg_ms"]),
                "regulatory_stability_score": float(geo["regulatory_stability_score"]),
                "total_appearances": int(total_appearances),
                "presence_rate": float(round(presence_rate, 3)),
                "reliability_score": float(round(reliability_score, 3)),
                "typical_lifespan_years": base_lifespan,
                # Labels
                "lifecycle_stage": lifecycle_stage,
                "retirement_year": retirement_year,
                "capacity_factor": round(capacity_factor, 4),
                "commissioning_year": commissioning_year,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def _train_models(
    training_data: pd.DataFrame,
    seed: int = 42,
) -> tuple[dict, LabelEncoder]:
    """Fit the three scikit-learn models on the synthetic training set.

    Returns a dict of fitted estimators and the LabelEncoder used for the
    lifecycle stage classifier (needed to decode predictions back to strings).
    """
    X = training_data[_FEATURE_COLS].values

    # --- 1. Lifecycle stage classifier ---
    le = LabelEncoder()
    y_stage = le.fit_transform(training_data["lifecycle_stage"])
    stage_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    stage_clf.fit(X, y_stage)

    # --- 2. Retirement year regressor ---
    y_retirement = training_data["retirement_year"].values.astype(float)
    retirement_reg = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    retirement_reg.fit(X, y_retirement)

    # --- 3. Capacity factor regressor ---
    y_cf = training_data["capacity_factor"].values.astype(float)
    cf_reg = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    cf_reg.fit(X, y_cf)

    models = {
        "lifecycle_stage_clf": stage_clf,
        "retirement_year_reg": retirement_reg,
        "capacity_factor_reg": cf_reg,
    }
    logger.info(
        "Trained 3 ML models on %d synthetic reference assets "
        "(lifecycle stage classes: %s).",
        len(training_data),
        list(le.classes_),
    )
    return models, le


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _predict_for_assets(
    enriched_assets: pd.DataFrame,
    models: dict,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    """Run all three models against the enriched real-asset feature matrix.

    Commissioning year is back-calculated from the predicted retirement year
    and the sector's typical lifespan so it is always internally consistent.
    Remaining years is clamped to [0, max_lifespan] to avoid implausible values.
    """
    if enriched_assets.empty:
        return pd.DataFrame()

    X = enriched_assets[_FEATURE_COLS].values

    # --- Stage predictions with per-class probabilities ---
    stage_probs = models["lifecycle_stage_clf"].predict_proba(X)
    stage_indices = np.argmax(stage_probs, axis=1)
    stage_labels = label_encoder.inverse_transform(stage_indices)
    stage_confidences = stage_probs[np.arange(len(stage_probs)), stage_indices]

    # --- Retirement year predictions (clamp to plausible range) ---
    raw_retirement = models["retirement_year_reg"].predict(X)
    retirement_years = np.clip(raw_retirement, 2025, 2080).astype(int)

    # --- Capacity factor predictions (convert to percentage, clamp 1-80%) ---
    raw_cf = models["capacity_factor_reg"].predict(X)
    cf_pct = np.clip(raw_cf * 100.0, 1.0, 80.0)

    results = enriched_assets[
        [
            "asset_id",
            "asset_name",
            "asset_country",
            "asset_sector",
            "capacity_mw",
            "latitude",
            "longitude",
            "altitude_avg_m",
            "territorial_type",
            "economic_level",
            "gdp_tier",
            "solar_irradiance_kwh_m2_yr",
            "wind_speed_avg_ms",
            "regulatory_stability_score",
            "typical_lifespan_years",
        ]
    ].copy()

    results["predicted_lifecycle_stage"] = stage_labels
    results["lifecycle_stage_confidence"] = np.round(stage_confidences, 4)
    results["estimated_retirement_year"] = retirement_years
    results["estimated_commissioning_year"] = results[
        "estimated_retirement_year"
    ] - results["typical_lifespan_years"].astype(int)
    results["predicted_remaining_years"] = np.maximum(
        0.0,
        np.round(
            (results["estimated_retirement_year"] - _REFERENCE_YEAR).astype(float), 1
        ),
    )
    results["predicted_capacity_factor_pct"] = np.round(cf_pct, 2)
    results["model_version"] = _MODEL_VERSION

    # Enforce integer types for year columns.
    results["estimated_commissioning_year"] = results[
        "estimated_commissioning_year"
    ].astype("int64")
    results["estimated_retirement_year"] = results["estimated_retirement_year"].astype(
        "int64"
    )
    results["gdp_tier"] = results["gdp_tier"].astype("int64")

    return results.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_ml_predictions(
    gold_root: Path,
    silver_outputs: dict[str, pd.DataFrame],
    gold_outputs: dict[str, pd.DataFrame],
    reference_root: Path,
    contract_paths: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    """Orchestrate the full ML extrapolation pipeline.

    Steps:
      1. Load country attributes and sector lifecycle parameters from
         ``reference_root/``.
      2. Enrich the silver ``asset_master`` with geographic and economic
         features plus lifecycle signal from the gold ``ownership_lifecycle``.
      3. Generate a deterministic synthetic training dataset.
      4. Train three models on the synthetic data.
      5. Run inference against the real enriched assets.
      6. Validate the output against the contract schema.
      7. Write ``gold/dw/asset_lifecycle_predictions.parquet`` and register
         in DuckDB.

    Returns a dict containing the predictions DataFrame under the key
    ``"asset_lifecycle_predictions"``.
    """
    logger.info("Starting ML asset lifecycle extrapolation.")

    # Initialise Langfuse objects up-front but guard against any SDK/network
    # error so telemetry failures can never abort pipeline execution.
    # If setup fails, lf/span fall back to the no-op stubs from observability.py
    # so every span.end() / lf.flush() call in except/finally is always valid.
    try:
        lf = get_langfuse()
        trace = lf.trace(name="sklearn_build_ml_predictions")
        span = trace.span(
            name="build_ml_predictions",
            input={
                "asset_count": len(silver_outputs.get("asset_master", [])),
                "seed": 42,
                "backend": __import__("os").environ.get("ML_BACKEND", "sklearn"),
            },
        )
    except Exception:
        logger.debug("Langfuse setup failed; tracing disabled for this run.", exc_info=True)
        from entity_data_lakehouse.observability import _NoOpLangfuse, _NoOpSpan, _NoOpTrace
        lf = _NoOpLangfuse()
        trace = _NoOpTrace()
        span = _NoOpSpan()

    try:
        # --- 1. Load reference data ---
        country_attrs = _load_country_attributes(reference_root)
        sector_params = _load_sector_lifecycle(reference_root)
        logger.info(
            "Loaded %d country attribute records and %d sector lifecycle profiles.",
            len(country_attrs),
            len(sector_params),
        )

        # --- 2. Enrich real assets with geographic + lifecycle features ---
        asset_master = silver_outputs["asset_master"]
        ownership_lifecycle = gold_outputs.get("ownership_lifecycle", pd.DataFrame())

        enriched = _enrich_asset_features(
            asset_master, ownership_lifecycle, country_attrs, sector_params
        )
        logger.info(
            "Enriched %d real assets with geographic and lifecycle features.", len(enriched)
        )

        # --- 3. Generate synthetic training data ---
        training_data = _generate_synthetic_training_data(
            country_attrs=country_attrs,
            sector_params=sector_params,
            n_samples=300,
            seed=42,
        )
        stage_dist = training_data["lifecycle_stage"].value_counts().to_dict()
        logger.info(
            "Generated %d synthetic training samples. Lifecycle stage distribution: %s",
            len(training_data),
            stage_dist,
        )

        # --- 4. Train models ---
        models, label_encoder = _train_models(training_data, seed=42)

        # --- 5. Predict for real assets ---
        predictions = _predict_for_assets(enriched, models, label_encoder)
        logger.info(
            "Generated lifecycle predictions for %d assets. Stages: %s",
            len(predictions),
            predictions["predicted_lifecycle_stage"].value_counts().to_dict()
            if not predictions.empty
            else {},
        )

        # --- 5a. Optional LoRA lifecycle-stage override ---
        # When ML_BACKEND=lora is set and a trained adapter exists, override only
        # predicted_lifecycle_stage and lifecycle_stage_confidence.  All other
        # columns (retirement year, capacity factor, commissioning year, etc.)
        # remain unchanged so the contract and integration-test row counts are
        # not affected.
        import os as _os

        _backend = _os.environ.get("ML_BACKEND")
        if _backend == "lora":
            _trusted_root = gold_root.parent / "models"
            _default_adapter = _trusted_root / "lifecycle_lora_adapter"
            _raw_adapter = Path(_os.environ.get("LORA_ADAPTER_PATH", str(_default_adapter)))
            try:
                from entity_data_lakehouse.ml_lora import validate_adapter_dir
                _adapter_dir = validate_adapter_dir(_raw_adapter, _trusted_root)
            except ValueError as _ve:
                logger.warning(
                    "ML_BACKEND=lora set but adapter path validation failed: %s; "
                    "falling back to sklearn predictions.",
                    _ve,
                )
            else:
                from entity_data_lakehouse.ml_lora import predict_lifecycle_lora_batch

                if len(enriched) != len(predictions):
                    raise ValueError(
                        f"LoRA override row mismatch: enriched={len(enriched)} "
                        f"vs predictions={len(predictions)}"
                    )
                logger.info(
                    "ML_BACKEND=lora: overriding lifecycle stage column for %d assets "
                    "using adapter at %s.",
                    len(predictions),
                    _adapter_dir,
                )
                batch_results = predict_lifecycle_lora_batch(
                    enriched,
                    adapter_dir=_adapter_dir,
                    parent_trace=span,
                )
                if len(batch_results) != len(predictions):
                    raise ValueError(
                        f"LoRA batch returned {len(batch_results)} results for "
                        f"{len(predictions)} rows — length mismatch."
                    )
                _lora_success_count = 0
                _failed_indices: list[int] = []
                for i, result in enumerate(batch_results):
                    if result is None:
                        _failed_indices.append(i)
                        continue
                    stage, conf = result
                    predictions.iat[
                        i, predictions.columns.get_loc("predicted_lifecycle_stage")
                    ] = stage
                    predictions.iat[
                        i, predictions.columns.get_loc("lifecycle_stage_confidence")
                    ] = conf
                    predictions.iat[
                        i, predictions.columns.get_loc("model_version")
                    ] = predictions.iat[
                        i, predictions.columns.get_loc("model_version")
                    ] + "+lora"
                    _lora_success_count += 1
                if _failed_indices:
                    _preview = _failed_indices[:5]
                    logger.warning(
                        "LoRA override: %d/%d rows failed (indices: %s%s); "
                        "retained sklearn predictions.",
                        len(_failed_indices),
                        len(predictions),
                        _preview,
                        "..." if len(_failed_indices) > 5 else "",
                    )

        # --- 6. Validate against contract ---
        validate_dataframe(predictions, contract_paths["asset_lifecycle_predictions"])

        # --- 7. Write Parquet ---
        dw_root = gold_root / "dw"
        dw_root.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(dw_root / "asset_lifecycle_predictions.parquet", index=False)
        logger.info("Wrote asset_lifecycle_predictions.parquet to %s", dw_root)

        try:
            span.end(output={"prediction_rows": len(predictions)})
        except Exception:
            logger.debug("Langfuse span.end() failed on success path; ignoring.", exc_info=True)
        return {"asset_lifecycle_predictions": predictions}

    except Exception as _exc:
        # End the span with error status so Langfuse captures failure cases.
        # Any exception that reaches here is re-raised after telemetry cleanup.
        try:
            span.end(
                output={"error": str(_exc)},
                level="ERROR",
                status_message=str(_exc),
            )
        except Exception:
            logger.debug("Langfuse span.end(error) failed; ignoring.", exc_info=True)
        raise

    finally:
        # Always flush buffered telemetry — whether the pipeline succeeded or
        # failed.  Without this, Langfuse traces for failed runs are silently
        # discarded because the process exits before the background thread drains.
        try:
            lf.flush()
        except Exception:
            logger.debug("Langfuse flush() failed; ignoring.", exc_info=True)
