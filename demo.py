"""
demo.py — Gradio front-end for the I2I recommender (this folder's engine).

Tabs:
  ✨ New here   — cold-start onboarding: collect cuisines / location / price /
                  dietary+amenities, get recommendations + map instantly, then
                  "like" places to warm up the profile.
  🧑 For You    — returning user_id -> personalised recommendations + map.
  👥 Group      — the USP: recommend for a whole group with 3 strategies.
"""
import gradio as gr
import requests
import folium

API_URL = "http://localhost:8000"

CUISINE_OPTIONS = [
    "Mexican", "Italian", "Chinese", "Japanese", "Indian", "American", "Thai",
    "Mediterranean", "Korean", "Vietnamese", "Sushi", "Pizza", "Burger",
    "Seafood", "Vegan", "BBQ", "Cafe", "Bar",
]

# offline-safe city -> (lat, lon); Nominatim used if reachable, else this table
CITY_COORDS = {
    "los angeles": (34.0522, -118.2437), "san francisco": (37.7749, -122.4194),
    "san diego": (32.7157, -117.1611), "sacramento": (38.5816, -121.4944),
    "san jose": (37.3382, -121.8863), "oakland": (37.8044, -122.2712),
    "fresno": (36.7378, -119.7871), "long beach": (33.7701, -118.1937),
    "santa monica": (34.0195, -118.4912), "pasadena": (34.1478, -118.1445),
    "irvine": (33.6846, -117.8265), "anaheim": (33.8366, -117.9143),
}


def resolve_city(city):
    if not city or not city.strip():
        return 34.0522, -118.2437
    key = city.lower().strip()
    for name, coords in CITY_COORDS.items():
        if name in key or key in name:
            return coords
    try:
        from geopy.geocoders import Nominatim
        loc = Nominatim(user_agent="tablemind_i2i").geocode(city, timeout=4)
        if loc:
            return loc.latitude, loc.longitude
    except Exception:
        pass
    return 34.0522, -118.2437


# ─── styling ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');
:root{--bg:#08080f;--surface:#10101a;--surface2:#18182a;--accent:#ff6b35;--accent2:#ffd166;--text:#ede8f0;--text-dim:#7a7a99;--border:#22223a;--success:#4ade80;--radius:14px;}
body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;font-family:'DM Sans',sans-serif!important;}
h1,h2,h3{font-family:'Playfair Display',serif!important;}
.gr-button{background:linear-gradient(135deg,#ff6b35,#ff4500)!important;color:#fff!important;border:none!important;border-radius:var(--radius)!important;font-weight:500!important;padding:12px 28px!important;box-shadow:0 4px 15px rgba(255,107,53,.3)!important;}
.gr-button:hover{transform:translateY(-2px)!important;}
.gr-input,.gr-dropdown,.gr-textbox textarea,.gr-textbox input,.gr-slider input{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;border-radius:10px!important;}
.gr-box,.gr-panel,.gr-group{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:var(--radius)!important;}
.gr-tab-nav button{background:transparent!important;color:var(--text-dim)!important;border:none!important;font-size:14px!important;padding:10px 20px!important;}
.gr-tab-nav button.selected{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
label{color:var(--text-dim)!important;font-size:12px!important;letter-spacing:.5px!important;text-transform:uppercase!important;}
.gr-checkbox label{text-transform:none!important;font-size:13px!important;color:var(--text)!important;}
::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
"""

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#08080f 0%,#160820 40%,#080f18 100%);padding:40px 40px 30px;border-bottom:1px solid #22223a;">
  <div style="display:flex;align-items:center;gap:18px;">
    <div style="width:52px;height:52px;background:linear-gradient(135deg,#ff6b35,#ffd166);border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:26px;box-shadow:0 6px 24px rgba(255,107,53,.35);">🍽</div>
    <div>
      <h1 style="font-size:34px;font-weight:900;margin:0;color:#ede8f0;letter-spacing:-1px;">TableMind</h1>
      <p style="margin:2px 0 0;color:#7a7a99;font-size:11px;letter-spacing:2px;text-transform:uppercase;">Semantic Restaurant Discovery · I2I Engine</p>
    </div>
  </div>
  <p style="color:#9a9ab8;max-width:600px;line-height:1.7;font-size:15px;margin:14px 0 0;">
    Tell us what you like and we'll find your spot — from a cold start, or for a whole group.
    Powered by 567-dim review embeddings over 80k California restaurants.
  </p>
</div>
"""


# ─── rendering ───────────────────────────────────────────────────────────────
def make_map(restaurants, center_lat=None, center_lon=None):
    pts = [r for r in restaurants if r.get("latitude") and r.get("longitude")]
    if not pts:
        return "<div style='color:#7a7a99;text-align:center;padding:60px;'>No map data.</div>"
    clat = center_lat or sum(r["latitude"] for r in pts) / len(pts)
    clon = center_lon or sum(r["longitude"] for r in pts) / len(pts)
    m = folium.Map(location=[clat, clon], zoom_start=12, tiles="CartoDB dark_matter")
    colors = ["#ff6b35", "#ffd166", "#4ade80", "#8888ff", "#ff88aa", "#88ffee"]
    for i, r in enumerate(pts[:20]):
        color = "#ff6b35" if i == 0 else colors[i % len(colors)]
        size = 14 if i == 0 else (10 if i < 3 else 7)
        dist = f"<div style='color:#888;font-size:11px;margin-top:4px'>📍 {round(r.get('distance_miles') or 0,1)} mi</div>" if r.get("distance_miles") else ""
        bd = ""
        if r.get("member_breakdown"):
            bd = "<div style='margin-top:6px;border-top:1px solid #eee;padding-top:4px'>" + "".join(
                f"<div style='font-size:11px;color:#666'>{b['name']}: {int(b['score']*100)}%</div>"
                for b in r["member_breakdown"]) + "</div>"
        popup = f"""<div style="font-family:sans-serif;min-width:200px;max-width:240px">
          <div style="font-weight:700;font-size:14px;margin-bottom:3px">#{i+1} {r.get('name','')}</div>
          <div style="color:#666;font-size:11px;margin-bottom:6px">{str(r.get('address',''))[:55]}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap">
            <span style="background:#fff3e0;color:#e65100;padding:2px 7px;border-radius:8px;font-size:11px">⭐ {round(r.get('avg_rating') or 0,1)}</span>
            <span style="background:#e8f5e9;color:#2e7d32;padding:2px 7px;border-radius:8px;font-size:11px">{r.get('price') or 'N/A'}</span>
            <span style="background:#e3f2fd;color:#1565c0;padding:2px 7px;border-radius:8px;font-size:11px">{int((r.get('final_score') or 0)*100)}% match</span>
          </div>{dist}{bd}</div>"""
        folium.CircleMarker([r["latitude"], r["longitude"]], radius=size, color=color,
                            fill=True, fill_color=color, fill_opacity=0.9,
                            popup=folium.Popup(popup, max_width=260),
                            tooltip=f"#{i+1} {r.get('name','')}").add_to(m)
    if center_lat and center_lon:
        folium.Marker([center_lat, center_lon],
                      icon=folium.DivIcon(html='<div style="font-size:20px">📍</div>'),
                      tooltip="Your location").add_to(m)
    return m._repr_html_()


def format_results_html(restaurants, show_breakdown=False):
    if not restaurants:
        return "<div style='color:#7a7a99;text-align:center;padding:60px;font-size:15px'>No restaurants found. Try a wider radius or fewer filters.</div>"
    cards = []
    for i, r in enumerate(restaurants):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"<span style='font-size:13px;color:#7a7a99'>#{i+1}</span>"
        price = r.get("price") or "N/A"
        rating = round(r.get("avg_rating") or 0, 1)
        cat = r.get("category", "")
        dist = f"<span style='background:#1c1c2e;padding:3px 9px;border-radius:8px;font-size:12px'>📍 {round(r.get('distance_miles') or 0,1)} mi</span>" if r.get("distance_miles") else ""
        pct = int((r.get("final_score") or 0) * 100)
        col = "#4ade80" if pct >= 70 else ("#ffd166" if pct >= 40 else "#ff6b35")
        bd = ""
        if show_breakdown and r.get("member_breakdown"):
            rows = "".join(
                f"<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #22223a;font-size:12px'>"
                f"<span style='color:#9a9ab8'>{b['name']}</span><span style='color:#ffd166;font-weight:500'>{int(b['score']*100)}%</span></div>"
                for b in r["member_breakdown"])
            bd = f"<div style='margin-top:12px;padding:10px 12px;background:#08080f;border-radius:10px;border:1px solid #22223a'><div style='font-size:10px;color:#7a7a99;margin-bottom:6px;letter-spacing:1px;text-transform:uppercase'>Per-member match</div>{rows}</div>"
        hl = "border-left:3px solid #ff6b35;" if i == 0 else ""
        cards.append(f"""<div style="background:#10101a;border:1px solid #22223a;border-radius:14px;padding:18px 20px;margin-bottom:10px;{hl}">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:5px">
            <span style="font-size:20px">{medal}</span>
            <span style="font-family:'Playfair Display',serif;font-size:17px;font-weight:700;color:#ede8f0">{r.get('name','')}</span>
          </div>
          <div style="color:#7a7a99;font-size:12px;margin:0 0 10px 30px">{str(r.get('address',''))[:70]}</div>
          <div style="display:flex;gap:6px;flex-wrap:wrap;padding-left:30px">
            <span style="background:#18182a;padding:3px 9px;border-radius:8px;font-size:12px">⭐ {rating}</span>
            <span style="background:#18182a;padding:3px 9px;border-radius:8px;font-size:12px">{price}</span>
            <span style="background:#18182a;padding:3px 9px;border-radius:8px;font-size:12px">{cat}</span>{dist}
          </div>
          <div style="margin-top:10px">
            <div style="display:flex;justify-content:space-between;font-size:11px;color:#7a7a99;margin-bottom:3px"><span style="text-transform:uppercase">Match</span><span style="color:{col};font-weight:500">{pct}%</span></div>
            <div style="background:#18182a;border-radius:4px;height:3px"><div style="background:linear-gradient(90deg,#ff6b35,#ffd166);width:{pct}%;height:3px;border-radius:4px"></div></div>
          </div>{bd}</div>""")
    return f"""<div style="max-height:680px;overflow-y:auto;padding-right:6px">
      <div style="color:#7a7a99;font-size:11px;margin-bottom:14px;letter-spacing:1px;text-transform:uppercase">{len(restaurants)} recommendations</div>
      {''.join(cards)}</div>"""


def _filters_dict(price, alcohol, outdoor, wheelchair, kid, music, resv, vegan, veg, gf):
    f = {}
    if price != "Any":
        f["max_price"] = ["$", "$$", "$$$", "$$$$"].index(price) + 1
    for flag, key in [(alcohol, "serves_alcohol"), (outdoor, "has_outdoor_seating"),
                      (wheelchair, "is_wheelchair"), (kid, "kid_friendly"),
                      (music, "has_live_music"), (resv, "accepts_reservations"),
                      (vegan, "vegan"), (veg, "vegetarian"), (gf, "gluten_free")]:
        if flag:
            f[key] = True
    return f


def _picks(results):
    """Build like + dislike pickers: choices for both + a {label: gmap_id} map."""
    labels, m = [], {}
    for i, r in enumerate(results):
        lbl = f"{i+1}. {str(r.get('name',''))[:42]}"
        labels.append(lbl)
        m[lbl] = r["gmap_id"]
    return (gr.update(choices=labels, value=[]),
            gr.update(choices=labels, value=[]), m)


def _empty_picks():
    return (gr.update(choices=[], value=[]),
            gr.update(choices=[], value=[]), {})


def _source_tag(results):
    src = results[0].get("source", "") if results else ""
    return {"history": "personalised from your history",
            "interactions": "personalised from your likes 👍",
            "cold_start": "based on your stated tastes",
            "popularity": "popular picks"}.get(src, src)


SORT_OPTIONS = ["✨ Best match", "📍 Nearest", "⭐ Top rated"]


def _sort_results(results, sort_by):
    """Re-order the API results client-side. API returns best-match order."""
    rs = list(results or [])
    if sort_by == "📍 Nearest":
        rs.sort(key=lambda r: (r.get("distance_miles") is None,
                               r.get("distance_miles") if r.get("distance_miles") is not None else 1e9))
    elif sort_by == "⭐ Top rated":
        rs.sort(key=lambda r: (r.get("avg_rating") or 0), reverse=True)
    else:  # ✨ Best match — similarity / match score
        rs.sort(key=lambda r: (r.get("final_score") or 0), reverse=True)
    return rs


# ─── login ────────────────────────────────────────────────────────────────────
# Two paths, no mode toggle: type an ID to sign back in, or a name to create a
# new account. ID wins if both are filled.
def _welcome_new(uid, name):
    nm = (name or "").strip() or "there"
    return (f"<div style='background:linear-gradient(135deg,rgba(255,107,53,.12),transparent);"
            f"border:1px solid rgba(255,107,53,.3);border-left:3px solid #ff6b35;border-radius:12px;"
            f"padding:14px 18px'><div style='color:#ff6b35;font-weight:600;font-size:15px'>"
            f"✨ Welcome, {nm}!</div><div style='color:#9a9ab8;font-size:13px;margin-top:5px'>"
            f"Your User ID is <code style='color:#ffd166;font-size:14px'>{uid}</code> — "
            f"save it to log back in next time. Now pick your tastes and hit "
            f"<b>Find restaurants</b>.</div></div>")


def _welcome_back(uid, name):
    nm = name or uid
    return (f"<div style='background:linear-gradient(135deg,rgba(74,222,128,.1),transparent);"
            f"border:1px solid rgba(74,222,128,.3);border-left:3px solid #4ade80;border-radius:12px;"
            f"padding:14px 18px'><div style='color:#4ade80;font-weight:600;font-size:15px'>"
            f"👋 Welcome back, {nm}!</div><div style='color:#9a9ab8;font-size:13px;margin-top:4px'>"
            f"Signed in as <code>{uid}</code>. Find restaurants below — your likes/dislikes keep "
            f"shaping your picks.</div></div>")


def login_fn(name, uid_in):
    """Returns: active_uid, status HTML, app-panel visibility update."""
    hide, show = gr.update(visible=False), gr.update(visible=True)
    uid = (uid_in or "").strip()
    name = (name or "").strip()

    # returning: an ID was entered -> verify it
    if uid:
        try:
            r = requests.get(f"{API_URL}/users/{uid}", timeout=15)
            if r.status_code != 200:
                return "", (f"<div style='color:#ff6b35;padding:14px'>⚠️ ID <code>{uid}</code> not found. "
                            f"Check it, or leave it blank and enter a name to start fresh.</div>"), hide
            u = r.json()
        except Exception as e:
            return "", f"<div style='color:#ff6b35;padding:14px'>⚠️ {e}</div>", hide
        return uid, _welcome_back(uid, u.get("name")), show

    # new: no ID -> create an account (name optional)
    try:
        r = requests.post(f"{API_URL}/users/create",
                          params={"name": name or None}, timeout=15)
        new_uid = r.json().get("user_id", "")
    except Exception as e:
        return "", f"<div style='color:#ff6b35;padding:14px'>⚠️ {e}</div>", hide
    return new_uid, _welcome_new(new_uid, name), show


# ─── recommend (cold-start for new users, personalised for returning, same call) ─
def _do_recommend(user_id, cuisines, city, radius, price, alcohol, outdoor,
                  wheelchair, kid, music, resv, vegan, veg, gf):
    lat, lon = resolve_city(city)
    filters = _filters_dict(price, alcohol, outdoor, wheelchair, kid, music, resv, vegan, veg, gf)
    resp = requests.post(f"{API_URL}/recommend", json={
        "user_id": user_id.strip(), "cuisines": cuisines or [], "filters": filters,
        "user_lat": lat, "user_lon": lon, "max_miles": float(radius), "n": 20,
    }, timeout=60)
    return resp.json().get("results", []), lat, lon


def _render(results, lat, lon, sort_by, banner=""):
    """Sort + build all six discover outputs from a raw results list."""
    shown = _sort_results(results, sort_by)
    like_p, dislike_p, pmap = _picks(shown)
    return (banner + format_results_html(shown), make_map(shown, lat, lon),
            like_p, dislike_p, pmap, results)


def discover_fn(user_id, cuisines, city, radius, price, alcohol, outdoor, wheelchair,
                kid, music, resv, vegan, veg, gf, sort_by):
    if not user_id or not user_id.strip():
        return ("<div style='color:#ff6b35;padding:16px'>⚠️ Log in first.</div>", "",
                *_empty_picks(), [])
    try:
        results, lat, lon = _do_recommend(user_id, cuisines, city, radius, price, alcohol,
                                          outdoor, wheelchair, kid, music, resv, vegan, veg, gf)
    except Exception as e:
        return f"<div style='color:#ff6b35;padding:16px'>⚠️ {e}</div>", "", *_empty_picks(), []
    if not results:
        return ("<div style='color:#7a7a99;padding:20px'>No results. Try a wider radius or fewer filters.</div>",
                "", *_empty_picks(), [])
    banner = f"<div style='color:#7a7a99;font-size:12px;margin-bottom:10px'>🧠 {_source_tag(results)}</div>"
    return _render(results, lat, lon, sort_by, banner)


def resort_fn(results, sort_by):
    """Re-render existing results under a new sort — no API call."""
    if not results:
        return "", "", *_empty_picks()
    shown = _sort_results(results, sort_by)
    like_p, dislike_p, pmap = _picks(shown)
    return format_results_html(shown), make_map(shown), like_p, dislike_p, pmap


# ─── feedback: record 👍/👎, update profile, then re-fetch ─────────────────────
def _send_fb(user_id, labels, state_map, sentiment):
    sent = 0
    for lbl in (labels or []):
        gid = state_map.get(lbl)
        if not gid:
            continue
        try:
            requests.post(f"{API_URL}/feedback",
                          json={"user_id": user_id, "gmap_id": gid, "sentiment": sentiment},
                          timeout=15)
            sent += 1
        except Exception:
            pass
    return sent


def refine_fn(user_id, liked, disliked, state_map, cuisines, city, radius, price,
              alcohol, outdoor, wheelchair, kid, music, resv, vegan, veg, gf, sort_by):
    if not user_id or not user_id.strip():
        return "<div style='color:#ff6b35;padding:16px'>⚠️ Log in and get recommendations first.</div>", "", *_empty_picks(), []
    if not liked and not disliked:
        return "<div style='color:#ffd166;padding:16px'>Tick at least one spot you'd go to (👍) or want to avoid (👎), then save.</div>", "", *_empty_picks(), []
    uid = user_id.strip()
    state_map = state_map or {}
    up = _send_fb(uid, liked, state_map, "up")
    down = _send_fb(uid, disliked, state_map, "down")
    # re-fetch — feedback now steers the query vector (likes pull toward, dislikes push away)
    try:
        results, lat, lon = _do_recommend(uid, cuisines, city, radius, price, alcohol,
                                          outdoor, wheelchair, kid, music, resv, vegan, veg, gf)
    except Exception as e:
        return f"<div style='color:#ff6b35;padding:16px'>⚠️ {e}</div>", "", *_empty_picks(), []
    note = (f"<div style='color:#4ade80;font-size:12px;margin-bottom:10px'>"
            f"👍 {up} liked · 👎 {down} avoided — profile updated, recommendations refreshed.</div>")
    return _render(results, lat, lon, sort_by, note)


# ─── Group (USP) ─────────────────────────────────────────────────────────────
def group_fn(strategy, city, radius, price,
             m1n, m1c, m1v, m1a, m1o, m2n, m2c, m2v, m2a, m2o,
             m3n, m3c, m3v, m3a, m3o, include_m3):
    lat, lon = resolve_city(city)
    pmax = ["$", "$$", "$$$", "$$$$"].index(price) + 1 if price != "Any" else 4
    members = []
    rows = [(m1n, m1c, m1v, m1a, m1o), (m2n, m2c, m2v, m2a, m2o)]
    if include_m3:
        rows.append((m3n, m3c, m3v, m3a, m3o))
    for nm, cz, vg, al, od in rows:
        if not nm or not nm.strip():
            continue
        f = {}
        if vg:
            f["vegan"] = True
        if al:
            f["serves_alcohol"] = True
        if od:
            f["has_outdoor_seating"] = True
        members.append({"name": nm.strip(), "cuisines": cz or [], "filters": f, "price_max": pmax})
    if len(members) < 2:
        return "<div style='color:#ff6b35;padding:16px'>⚠️ Add at least 2 members.</div>", ""
    smap = {"🤝 Least Misery — nobody has a bad time": "least_misery",
            "⚖️ Average — best for the group": "average",
            "🎉 Most Pleasure — make someone's night": "most_pleasure"}
    strat = smap.get(strategy, "least_misery")
    try:
        resp = requests.post(f"{API_URL}/recommend/group", json={
            "members": members, "strategy": strat,
            "user_lat": lat, "user_lon": lon, "max_miles": float(radius), "n": 10,
        }, timeout=60)
        results = resp.json().get("results", [])
    except Exception as e:
        return f"<div style='color:#ff6b35;padding:16px'>⚠️ {e}</div>", ""
    if not results:
        return "<div style='color:#7a7a99;padding:20px'>No spot satisfies everyone's constraints. Try relaxing dietary/amenity needs or widening the radius.</div>", ""
    labels = {"least_misery": ("🤝", "Least Misery", "Nobody has a bad time — ranked by the lowest individual match.", "#ff6b35"),
              "average": ("⚖️", "Average", "Maximising the whole group's happiness.", "#ffd166"),
              "most_pleasure": ("🎉", "Most Pleasure", "Best possible night for at least one person.", "#4ade80")}
    e, nm, desc, c = labels[strat]
    banner = f"<div style='background:linear-gradient(135deg,{c}18,transparent);border:1px solid {c}40;border-left:3px solid {c};border-radius:12px;padding:14px 18px;margin-bottom:14px'><div style='display:flex;align-items:center;gap:8px'><span style='font-size:18px'>{e}</span><span style='font-weight:600;color:{c}'>{nm}</span><span style='color:#7a7a99;font-size:12px'>· Group of {len(members)}</span></div><div style='color:#9a9ab8;font-size:13px;margin-top:3px'>{desc}</div></div>"
    return banner + format_results_html(results, show_breakdown=True), make_map(results, lat, lon)


# ─── amenity widget helper ──────────────────────────────────────────────────────
def _amenities():
    with gr.Row():
        a = gr.Checkbox(label="🍺 Bar / Alcohol")
        o = gr.Checkbox(label="🌳 Outdoor seating")
    with gr.Row():
        w = gr.Checkbox(label="♿ Wheelchair access")
        k = gr.Checkbox(label="👶 Kid friendly")
    with gr.Row():
        mu = gr.Checkbox(label="🎵 Live music")
        rv = gr.Checkbox(label="📅 Reservations")
    with gr.Row():
        vg = gr.Checkbox(label="🌱 Vegan")
        ve = gr.Checkbox(label="🥗 Vegetarian")
        g = gr.Checkbox(label="🌾 Gluten-free")
    return a, o, w, k, mu, rv, vg, ve, g


def build_ui():
    with gr.Blocks(title="TableMind — I2I Restaurant Discovery") as demo:
        gr.HTML(HEADER_HTML)
        with gr.Tabs():

            # ── Discover (login + recommend; new & returning users together) ──
            with gr.Tab("🍽 Discover"):
                # ---- login (no toggle: name = new account, ID = sign back in) ----
                with gr.Group():
                    gr.HTML("<div style='font-family:Playfair Display,serif;font-size:19px;color:#ede8f0;padding:4px 2px'>Sign in</div>"
                            "<div style='color:#9a9ab8;font-size:13px;margin:0 0 8px'>Coming back? Enter your <b>User ID</b>. New here? Leave it blank, type a <b>name</b>, and we'll create your account.</div>")
                    dc_id = gr.Textbox(label="🔑 Returning — your User ID", placeholder="e.g. a3f9c1b2")
                    dc_name = gr.Textbox(label="✨ New — your name", placeholder="e.g. Alex")
                    dc_login_btn = gr.Button("Continue →", variant="primary")
                    dc_login_status = gr.HTML()
                dc_uid = gr.Textbox(visible=False)  # resolved active user id

                # ---- app panel (revealed after login) ----
                with gr.Group(visible=False) as dc_app:
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=320):
                            dc_cuisines = gr.CheckboxGroup(CUISINE_OPTIONS, label="🍽 Cuisines you love")
                            with gr.Group():
                                dc_city = gr.Textbox(label="📍 City or area", value="Los Angeles")
                                dc_radius = gr.Slider(1, 50, value=10, step=1, label="Search radius (miles)")
                                dc_price = gr.Radio(["Any", "$", "$$", "$$$", "$$$$"], value="Any", label="Max price")
                            with gr.Group():
                                gr.HTML("<div style='font-size:11px;color:#7a7a99;padding:6px 0'>✨ DIETARY & AMENITIES</div>")
                                dc_a, dc_o, dc_w, dc_k, dc_mu, dc_rv, dc_vg, dc_ve, dc_g = _amenities()
                            dc_btn = gr.Button("Find restaurants →", variant="primary", size="lg")
                            with gr.Group():
                                gr.HTML("<div style='font-size:11px;color:#7a7a99;padding:6px 0'>RATE — teach us your taste</div>")
                                dc_likes = gr.CheckboxGroup([], label="👍 Spots you'd go to")
                                dc_dislikes = gr.CheckboxGroup([], label="👎 Spots to avoid")
                                dc_like_btn = gr.Button("Save feedback & personalise", variant="secondary")
                            dc_pickmap = gr.State({})
                            dc_results = gr.State([])
                        with gr.Column(scale=2):
                            dc_sort = gr.Radio(SORT_OPTIONS, value=SORT_OPTIONS[0], label="Sort by")
                            dc_res = gr.HTML()
                            dc_map = gr.HTML()

                _disc_inputs = [dc_uid, dc_cuisines, dc_city, dc_radius, dc_price,
                                dc_a, dc_o, dc_w, dc_k, dc_mu, dc_rv, dc_vg, dc_ve, dc_g, dc_sort]
                _disc_outputs = [dc_res, dc_map, dc_likes, dc_dislikes, dc_pickmap, dc_results]

                dc_login_btn.click(login_fn, [dc_name, dc_id],
                                   [dc_uid, dc_login_status, dc_app])
                dc_btn.click(discover_fn, inputs=_disc_inputs, outputs=_disc_outputs)
                dc_sort.change(resort_fn, inputs=[dc_results, dc_sort],
                               outputs=[dc_res, dc_map, dc_likes, dc_dislikes, dc_pickmap])
                dc_like_btn.click(
                    refine_fn,
                    inputs=[dc_uid, dc_likes, dc_dislikes, dc_pickmap, dc_cuisines, dc_city,
                            dc_radius, dc_price, dc_a, dc_o, dc_w, dc_k, dc_mu, dc_rv,
                            dc_vg, dc_ve, dc_g, dc_sort],
                    outputs=_disc_outputs)

            # ── Group (USP) ──
            with gr.Tab("👥 Group — Our Speciality"):
                gr.HTML("<div style='background:linear-gradient(135deg,rgba(255,107,53,.08),rgba(255,209,102,.05));border:1px solid rgba(255,107,53,.2);border-radius:14px;padding:18px 22px;margin:8px 0 20px'><div style='font-family:Playfair Display,serif;font-size:20px;color:#ede8f0'>Group Recommendation Engine</div><div style='color:#9a9ab8;font-size:14px;margin-top:6px;line-height:1.6'>One's vegan, another wants a bar, a third's bringing kids? Add everyone's tastes and constraints and we'll find one place that works for the whole group.</div></div>")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=340):
                        g_strategy = gr.Radio(
                            ["🤝 Least Misery — nobody has a bad time",
                             "⚖️ Average — best for the group",
                             "🎉 Most Pleasure — make someone's night"],
                            value="🤝 Least Misery — nobody has a bad time", label="🎯 Strategy")
                        with gr.Group():
                            g_city = gr.Textbox(label="📍 City or area", value="Los Angeles")
                            g_radius = gr.Slider(1, 50, value=15, step=1, label="Radius (miles)")
                            g_price = gr.Radio(["Any", "$", "$$", "$$$", "$$$$"], value="Any", label="Max price for everyone")
                        gr.HTML("<div style='font-family:Playfair Display,serif;font-size:16px;color:#ede8f0;margin:14px 0 8px'>Members</div>")
                        with gr.Group():
                            gr.HTML("<div style='color:#ff6b35;font-size:13px;font-weight:500;padding:4px 0'>● Member 1</div>")
                            m1n = gr.Textbox(label="Name", placeholder="Maya")
                            m1c = gr.CheckboxGroup(CUISINE_OPTIONS, label="Cuisines")
                            with gr.Row():
                                m1v = gr.Checkbox(label="🌱 Vegan"); m1a = gr.Checkbox(label="🍺 Bar"); m1o = gr.Checkbox(label="🌳 Outdoor")
                        with gr.Group():
                            gr.HTML("<div style='color:#ffd166;font-size:13px;font-weight:500;padding:4px 0'>● Member 2</div>")
                            m2n = gr.Textbox(label="Name", placeholder="James")
                            m2c = gr.CheckboxGroup(CUISINE_OPTIONS, label="Cuisines")
                            with gr.Row():
                                m2v = gr.Checkbox(label="🌱 Vegan"); m2a = gr.Checkbox(label="🍺 Bar"); m2o = gr.Checkbox(label="🌳 Outdoor")
                        include_m3 = gr.Checkbox(label="➕ Add a 3rd member", value=False)
                        with gr.Group(visible=False) as m3box:
                            gr.HTML("<div style='color:#4ade80;font-size:13px;font-weight:500;padding:4px 0'>● Member 3</div>")
                            m3n = gr.Textbox(label="Name", placeholder="Ravi")
                            m3c = gr.CheckboxGroup(CUISINE_OPTIONS, label="Cuisines")
                            with gr.Row():
                                m3v = gr.Checkbox(label="🌱 Vegan"); m3a = gr.Checkbox(label="🍺 Bar"); m3o = gr.Checkbox(label="🌳 Outdoor")
                        include_m3.change(lambda x: gr.update(visible=x), include_m3, m3box)
                        g_btn = gr.Button("Find Group Spots →", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        g_res = gr.HTML()
                        g_map = gr.HTML()
                g_btn.click(group_fn,
                            inputs=[g_strategy, g_city, g_radius, g_price,
                                    m1n, m1c, m1v, m1a, m1o, m2n, m2c, m2v, m2a, m2o,
                                    m3n, m3c, m3v, m3a, m3o, include_m3],
                            outputs=[g_res, g_map])

    return demo


if __name__ == "__main__":
    import os
    # Env overrides: GRADIO_SERVER_PORT (default 7860) and SHARE=1 -> public *.gradio.live link.
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    share = os.environ.get("SHARE", "0") not in ("0", "", "false", "False")
    # Gradio 6 moved `css` from Blocks() to launch(); fall back for older 4.x.
    ui = build_ui()
    try:
        ui.launch(server_name="0.0.0.0", server_port=port, share=share, css=CSS)
    except TypeError:
        ui.css = CSS
        ui.launch(server_name="0.0.0.0", server_port=port, share=share)
