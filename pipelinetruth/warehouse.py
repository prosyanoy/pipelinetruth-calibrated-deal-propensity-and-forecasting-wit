"""DuckDB warehouse: runs the plain-SQL transform layers and exposes the feature table."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .config import SQL_DIR, Config

SQL_FILES = ["01_staging.sql", "02_data_quality.sql", "03_features.sql"]


def build_warehouse(cfg: Config | None = None, raw_dir: Path | None = None, db_path: Path | str | None = None) -> duckdb.DuckDBPyConnection:
    """Run all SQL layers against the raw CSVs and return an open connection."""
    cfg = cfg or Config()
    raw_dir = Path(raw_dir or cfg.raw_dir)
    db_path = str(db_path or cfg.db_path)
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    params = {"${RAW_DIR}": raw_dir.resolve().as_posix(), "${SNAPSHOT_DAYS}": str(int(cfg.snapshot_days))}
    for name in SQL_FILES:
        sql = (SQL_DIR / name).read_text()
        for k, v in params.items():
            sql = sql.replace(k, v)
        con.execute(sql)
    return con


def load_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("SELECT * FROM features ORDER BY snapshot_date, opp_id").df()
    for c in ("created_date", "snapshot_date", "closed_date"):
        df[c] = pd.to_datetime(df[c])
    df["dq_error"] = df["dq_error"].fillna(False).astype(bool)
    return df


def dq_report(con: duckdb.DuckDBPyConnection) -> dict:
    return {
        "rules": con.execute("SELECT * FROM dq_summary").df(),
        "table_checks": con.execute("SELECT * FROM dq_table_checks").df(),
        "n_opps": int(con.execute("SELECT count(*) FROM stg_opportunities").fetchone()[0]),
        "n_quarantined": int(con.execute("SELECT count(*) FROM dq_opp WHERE dq_error").fetchone()[0]),
    }
