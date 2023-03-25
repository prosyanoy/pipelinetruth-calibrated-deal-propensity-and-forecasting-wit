"""FastAPI scoring service: calibrated win probability, reason codes and trust flags per deal."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import Config
from .scoring import score_frame


class Deal(BaseModel):
    """Day-N snapshot of a deal. Every field except ``opp_id`` is optional; absent values score as missing."""
    model_config = ConfigDict(extra="forbid")
    opp_id: str | None = None
    amount: float | None = Field(None, description="Deal amount; missing or <= 0 is a data-quality error and the deal is rejected")
    segment: str | None = None
    region: str | None = None
    industry: str | None = None
    lead_source: str | None = None
    age_days: int | None = Field(None, description="Days since creation; deals younger than the snapshot window are not scoreable")
    n_activities_snap: float | None = None
    n_meetings_snap: float | None = None
    n_contacts_snap: float | None = None
    days_since_last_activity: float | None = None
    stage_ord: int | None = Field(None, ge=1, le=4, description="1=Qualify 2=Develop 3=Propose 4=Negotiate")
    expected_cycle_days: float | None = None
    close_date_stale: int | None = Field(None, ge=0, le=1)
    account_prior_deals: float | None = None
    account_prior_wins: float | None = None
    owner_prior_deals: float | None = None
    owner_prior_win_rate: float | None = None
    days_to_quarter_end: float | None = None
    created_qtr: int | None = Field(None, ge=1, le=4)


class ScoreRequest(BaseModel):
    deals: list[Deal] = Field(min_length=1, max_length=1000)


def _load(model_dir: Path, artifact_dir: Path) -> dict:
    bundle_path = model_dir / "bundle.joblib"
    if not bundle_path.exists():
        return {}
    meta = json.loads((model_dir / "meta.json").read_text())
    card = artifact_dir / "model_card.md"
    pri = artifact_dir / "priorities.csv"
    return {"bundle": joblib.load(bundle_path), "meta": meta, "card": card.read_text() if card.exists() else "",
            "priorities": pd.read_csv(pri) if pri.exists() else pd.DataFrame()}


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.update(_load(cfg.model_dir, cfg.artifact_dir))
        yield

    app = FastAPI(title="PipelineTruth scoring API", version="0.1.0", lifespan=lifespan)

    def need_model():
        if "bundle" not in state:
            raise HTTPException(503, "No trained model found. Run `make train` (python -m pipelinetruth.pipeline) first.")
        return state

    @app.get("/health")
    def health():
        ok = "bundle" in state
        return {"status": "ok" if ok else "no_model", "model_version": state["meta"]["version"] if ok else None,
                "drift_status": state["meta"]["drift_status"] if ok else None}

    @app.post("/score")
    def score(req: ScoreRequest):
        st = need_model()
        bundle, meta = st["bundle"], st["meta"]
        snap = int(meta["snapshot_days"])
        df = pd.DataFrame([d.model_dump() for d in req.deals])
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["age_days"] = pd.to_numeric(df["age_days"], errors="coerce")
        df["log_amount"] = np.where(df["amount"] > 0, np.log(df["amount"].where(df["amount"] > 0, 1.0)), np.nan)
        results = []
        rejected: dict[int, str] = {}
        for i, d in df.iterrows():
            if pd.isna(d["amount"]) or d["amount"] <= 0:
                rejected[i] = "data-quality error: amount missing or non-positive"
            elif pd.notna(d["age_days"]) and d["age_days"] < snap:
                rejected[i] = f"too new: needs at least {snap} days of history for a snapshot score"
        ok_idx = [i for i in df.index if i not in rejected]
        scored = score_frame(bundle, df.loc[ok_idx], caution_segments=set(meta.get("caution_segments", [])),
                             drift_status=meta["drift_status"], dq_rules_col=None) if ok_idx else None
        pos = {i: k for k, i in enumerate(ok_idx)}
        for i, d in enumerate(req.deals):
            if i in rejected:
                results.append({"opp_id": d.opp_id, "status": "rejected", "p_win": None, "reason_codes": [],
                                "trust": "do_not_use", "trust_flags": [rejected[i]]})
                continue
            r = scored.iloc[pos[i]]
            results.append({"opp_id": d.opp_id, "status": "scored", "p_win": round(float(r["p_win"]), 4),
                            "expected_value": round(float(r["p_win"]) * d.amount, 2), "reason_codes": r["reasons"],
                            "trust": r["trust"], "trust_flags": r["trust_flags"]})
        return {"model_version": meta["version"], "drift_status": meta["drift_status"], "results": results}

    @app.get("/priorities")
    def priorities(top: int = Query(20, ge=1, le=500), segment: str | None = None):
        st = need_model()
        pr = st["priorities"]
        if segment:
            pr = pr[pr["segment"] == segment]
        return {"model_version": st["meta"]["version"], "n_total": len(pr),
                "deals": json.loads(pr.head(top).to_json(orient="records"))}

    @app.get("/model-card", response_class=PlainTextResponse)
    def model_card():
        st = need_model()
        if not st["card"]:
            raise HTTPException(404, "model card not found")
        return st["card"]

    return app


app = create_app()
