"""Configuration read from environment variables with safe local defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SQL_DIR = ROOT / "sql"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Config:
    data_dir: Path = field(default_factory=lambda: Path(_env("PT_DATA_DIR", "data")))
    artifact_dir: Path = field(default_factory=lambda: Path(_env("PT_ARTIFACT_DIR", "artifacts")))
    seed: int = field(default_factory=lambda: int(_env("PT_SEED", "42")))
    n_opps: int = field(default_factory=lambda: int(_env("PT_N_OPPS", "7000")))
    snapshot_days: int = field(default_factory=lambda: int(_env("PT_SNAPSHOT_DAYS", "14")))
    interval_level: float = field(default_factory=lambda: float(_env("PT_INTERVAL_LEVEL", "0.8")))
    # Synthetic world: opportunities are created in [start, as_of]; outcomes are known up to as_of.
    start_date: pd.Timestamp = pd.Timestamp("2022-01-01")
    as_of: pd.Timestamp = pd.Timestamp("2025-06-30")
    drift_start: pd.Timestamp = pd.Timestamp("2024-07-01")
    max_cycle_days: int = 180

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "pipelinetruth.duckdb"

    @property
    def model_dir(self) -> Path:
        return self.artifact_dir / "model"

    @property
    def mlflow_uri(self) -> str:
        return _env("MLFLOW_TRACKING_URI", f"sqlite:///{self.artifact_dir.as_posix()}/mlflow.db")
