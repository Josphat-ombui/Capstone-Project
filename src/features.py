"""Pre-release predictors for the batch classifier.

Only information known before release is used. Assay, dissolution, impurity, actual yield and
rejection reason are measured at or after release, so they are never predictors.
"""
import pandas as pd

NUMERIC_FEATURES = [
    "equipment_age_years", "maintenance_days_ago", "temp_avg", "temp_variance", "pressure_avg",
    "pressure_variance", "humidity_avg", "raw_material_quality", "operator_experience_years",
    "production_hours", "batch_size",
]
MAINTENANCE_FEATURES = ["maintenance_due_soon", "maintenance_critical"]   # reference: Current
DOSAGE_LEVELS = ["Capsule", "Film-coated Tablet", "Oral Suspension", "Syrup", "Oral Liquid", "Bolus",
                 "Cream", "Injection"]                                     # reference: Tablet
SECTION_LEVELS = ["Oral Liquids", "Creams & Ointments", "Injectables"]     # reference: Oral Solids

FEATURE_LABELS = {
    "equipment_age_years": "Equipment age (years)",
    "maintenance_days_ago": "Days since maintenance",
    "temp_avg": "Average temperature",
    "temp_variance": "Temperature variance",
    "pressure_avg": "Average pressure",
    "pressure_variance": "Pressure variance",
    "humidity_avg": "Average humidity",
    "raw_material_quality": "Raw material quality",
    "operator_experience_years": "Operator experience (years)",
    "production_hours": "Production hours",
    "batch_size": "Batch size",
    "maintenance_due_soon": "Maintenance: Due Soon",
    "maintenance_critical": "Maintenance: Critical",
    **{f"dosage_{d}": f"Dosage: {d}" for d in DOSAGE_LEVELS},
    **{f"section_{s}": f"Section: {s.replace('&', 'and')}" for s in SECTION_LEVELS},
}


def feature_columns():
    """Names of the model inputs, in a fixed order."""
    return (NUMERIC_FEATURES + MAINTENANCE_FEATURES + [f"dosage_{d}" for d in DOSAGE_LEVELS]
            + [f"section_{s}" for s in SECTION_LEVELS])


def build_features(batches):
    """Model input matrix (missing numeric values are left for the model step to impute)."""
    X = batches[NUMERIC_FEATURES].astype(float).copy()
    X["maintenance_due_soon"] = (batches["maintenance_status"] == "Due Soon").astype(int)
    X["maintenance_critical"] = (batches["maintenance_status"] == "Critical").astype(int)
    for d in DOSAGE_LEVELS:
        X[f"dosage_{d}"] = (batches["dosage_form"] == d).astype(int)
    for s in SECTION_LEVELS:
        X[f"section_{s}"] = (batches["manufacturing_section"] == s).astype(int)
    return X[feature_columns()]
