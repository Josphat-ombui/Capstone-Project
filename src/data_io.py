"""Reading raw files and reading/writing processed tables."""
import pandas as pd

from src.config import DATA_RAW, DATA_PROCESSED, RAW_FILES

# Columns read as text so codes such as "SUP_001" keep their leading zeros.
_TEXT_COLUMNS = {"inventory": {"Supplier_ID": str}, "sales": {"Supplier_ID": str}}


def load_raw(key):
    """Read one raw file exactly as supplied ('batch', 'maintenance', 'sales' or 'inventory')."""
    return pd.read_csv(DATA_RAW / RAW_FILES[key], dtype=_TEXT_COLUMNS.get(key))


def _parquet_available():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return True
        except ImportError:
            return False


def save_processed(df, name):
    """Write a processed table to data/processed/ (Parquet when available, otherwise pickle)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if _parquet_available():
        df.to_parquet(DATA_PROCESSED / f"{name}.parquet")
    else:
        df.to_pickle(DATA_PROCESSED / f"{name}.pkl")


def load_processed(name):
    """Read a processed table. If it has not been built yet, run the step that builds it."""
    for suffix, reader in ((".parquet", pd.read_parquet), (".pkl", pd.read_pickle)):
        path = DATA_PROCESSED / f"{name}{suffix}"
        if path.exists():
            try:
                return reader(path)
            except ImportError:
                continue
    if name in ("batches", "sales", "inventory", "quality_register"):
        from src.cleaning import run_cleaning
        return run_cleaning(save=True)[name]
    if name == "alerts_test_period":
        from src.pipeline import build_risk_outputs
        return build_risk_outputs()
    raise FileNotFoundError(f"No processed table called '{name}' in {DATA_PROCESSED}")
