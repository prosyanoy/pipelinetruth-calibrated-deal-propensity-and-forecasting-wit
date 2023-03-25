import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pytest

from pipelinetruth.config import Config
from pipelinetruth.synth import generate, write_raw
from pipelinetruth.warehouse import build_warehouse, load_features

N_SMALL = 3000


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("pt")
    return Config(data_dir=root / "data", artifact_dir=root / "artifacts", n_opps=N_SMALL, seed=3)


@pytest.fixture(scope="session")
def raw_tables(small_cfg):
    tables = generate(small_cfg)
    write_raw(tables, small_cfg.raw_dir)
    return tables


@pytest.fixture(scope="session")
def warehouse(small_cfg, raw_tables):
    con = build_warehouse(small_cfg, db_path=":memory:")
    yield con
    con.close()


@pytest.fixture(scope="session")
def feat(warehouse):
    return load_features(warehouse)


@pytest.fixture(scope="session")
def pipeline_out(small_cfg, raw_tables):
    from pipelinetruth.pipeline import run
    return run(small_cfg, regenerate=False, use_mlflow=False)
