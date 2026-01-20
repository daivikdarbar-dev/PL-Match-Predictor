import streamlit as st
import requests
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Match Predictor")
st.caption("Automated, rule-based model using live league data")

# -------------------- API CONFIG --------------------
API_KEY = st.secrets["RAPIDAPI_KEY"]

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com"
}

# -------------------- LEAGUE MAP --------------------
LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61
}

CURRENT_SEASON = 2024

# -------------------- DATA FETCH --------------------
@st.cache_data(ttl=3600)
def fetch_standings(league_id, season):
    url = "https://api-football-v1.p.rapidapi.com/v3/standings"
    params = {"league": league_id, "season": season}

    response = requests.get(url, headers=HEADERS, params=params, timeout=20)
    data = response.json()

    standings = data["response"][0]["league"]["standings"][0]

    teams = {}
    for row in standings:
        team = row["team"]["name"]
        played = row["all"]["played"]
        gf = row["all"]["goals"]["for"]
        ga = row["all"]["goals"]["against"]

        teams[team] = {
            "position": row["rank"],
            "gf": gf / max(played, 1),
            "ga": ga / max(played, 1)
        }

    return teams

# -------------------- PREDICTION LOGIC --------------------
def predict_match(home, away):
    score = 0

    score += (away["position"] - home["position"]) * 0.6
    score += (home["gf"] - away["gf"]) * 1.3
    score += (away["ga"] - home["ga"]) * 1.0
    score += 0.4  # home advantage

    home_prob = 1 / (1 + np.exp(-score))
    away_prob = 1 - home_prob
    draw_prob = 0.22

    total = home_prob + away_prob + draw_prob
    home_prob = home_prob / total * 100
    draw_prob = draw_prob / total * 100
    away_prob = away_prob / total * 100

    confidence = "High" if abs(home_prob - away_prob) > 15 else "Medium"

    return {
        "home": home_prob,
        "draw": draw_prob,
        "away": away_prob,
        "confidence": confidence
    }

# -------------------- UI --------------------
league_name = st.selectbox("Competition", LEAGUES.keys())
league_id = LEAGUES[league_name]

teams_data = fetch_standings(league_id, CURRENT_SEASON)

if not teams_data:
    st.error("Unable to load league data.")
    st.stop()

teams = sorted(teams_data.keys())

col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", teams)
away_team = col2.selectbox("✈️ Away Team", teams, index=1)

if st.button("🔮 Predict Match", type="primary", use_container_width=True):
    home = teams_data[home_team]
    away = teams_data[away_team]

    result = predict_match(home, away)

    st.markdown("---")
    st.subheader("Prediction")

    c1, c2, c3 = st.columns(3)
    c1.metric(home_team, f"{result['home']:.1f}%")
    c2.metric("Draw", f"{result['draw']:.1f}%")
    c3.metric(away_team, f"{result['away']:.1f}%")

    st.markdown(f"**Confidence:** {result['confidence']}")

    st.markdown("---")
    st.caption("Rule-based model. No betting odds. No black-box ML.")
