"""FastAPI serving layer for the I2I recommender (this folder's engine)."""
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import serving
from database import upsert_user, get_user, get_user_interactions
from cache import cache

app = FastAPI(title="TableMind (I2I) API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _startup():
    serving.load_all()


class Filters(BaseModel):
    max_price: Optional[int] = None
    serves_alcohol: Optional[bool] = False
    has_outdoor_seating: Optional[bool] = False
    is_wheelchair: Optional[bool] = False
    kid_friendly: Optional[bool] = False
    has_live_music: Optional[bool] = False
    accepts_reservations: Optional[bool] = False
    vegan: Optional[bool] = False
    vegetarian: Optional[bool] = False
    gluten_free: Optional[bool] = False
    halal: Optional[bool] = False
    category: Optional[str] = None


class OnboardRequest(BaseModel):
    name: Optional[str] = None
    cuisines: Optional[List[str]] = []
    filters: Optional[Filters] = None
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    max_miles: Optional[float] = 10
    n: Optional[int] = 20


class RecommendRequest(BaseModel):
    user_id: Optional[str] = None
    cuisines: Optional[List[str]] = []
    filters: Optional[Filters] = None
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    max_miles: Optional[float] = None
    n: Optional[int] = 20


class GroupMember(BaseModel):
    user_id: Optional[str] = None
    name: str
    cuisines: Optional[List[str]] = []
    filters: Optional[Dict[str, Any]] = {}
    price_max: Optional[int] = 4


class GroupRequest(BaseModel):
    members: List[GroupMember]
    strategy: Optional[str] = "least_misery"
    user_lat: Optional[float] = None
    user_lon: Optional[float] = None
    max_miles: Optional[float] = None
    n: Optional[int] = 10


class InteractionRequest(BaseModel):
    user_id: str
    gmap_id: str
    rating: Optional[float] = 5.0


class FeedbackRequest(BaseModel):
    user_id: str
    gmap_id: str
    sentiment: str  # "up" | "down"


@app.post("/users/create")
def create_user(name: Optional[str] = None):
    uid = str(uuid.uuid4())[:8]
    upsert_user(uid, name)
    return {"user_id": uid, "name": name}


@app.get("/users/{user_id}")
def user_info(user_id: str):
    u = get_user(user_id)
    if not u:
        raise HTTPException(404, "User not found")
    return u


@app.post("/onboard")
def onboard(req: OnboardRequest):
    uid = str(uuid.uuid4())[:8]
    prefs = {"cuisines": req.cuisines or [],
             "filters": req.filters.dict() if req.filters else {}}
    upsert_user(uid, req.name, prefs)
    results = serving.recommend(
        user_id=uid, prefs=prefs,
        filters=req.filters.dict() if req.filters else {},
        user_lat=req.user_lat, user_lon=req.user_lon,
        max_miles=req.max_miles, n=req.n)
    return {"user_id": uid, "results": results}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    prefs = {"cuisines": req.cuisines or []}
    # merge stored prefs for known users
    if req.user_id:
        u = get_user(req.user_id)
        if u and u.get("preferences"):
            stored = u["preferences"]
            if not prefs["cuisines"]:
                prefs["cuisines"] = stored.get("cuisines", [])
    results = serving.recommend(
        user_id=req.user_id, prefs=prefs,
        filters=req.filters.dict() if req.filters else {},
        user_lat=req.user_lat, user_lon=req.user_lon,
        max_miles=req.max_miles, n=req.n)
    return {"results": results}


@app.post("/recommend/group")
def recommend_group(req: GroupRequest):
    results = serving.group_recommend(
        members=[m.dict() for m in req.members],
        strategy=req.strategy,
        user_lat=req.user_lat, user_lon=req.user_lon,
        max_miles=req.max_miles, n=req.n)
    return {"results": results, "strategy": req.strategy}


@app.post("/interactions")
def interaction(req: InteractionRequest):
    serving.log_like(req.user_id, req.gmap_id, req.rating)
    return {"status": "ok"}


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if req.sentiment not in ("up", "down"):
        raise HTTPException(400, "sentiment must be 'up' or 'down'")
    result = serving.record_feedback(req.user_id, req.gmap_id, req.sentiment)
    return {"status": "ok", **result}


@app.get("/health")
def health():
    return {"status": "ok"}
