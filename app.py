import streamlit as st
import requests
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Football Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Match Predictor")
st.caption("Competition-aware, automated, rule-based model")

# =========================
# API CONFIG
# =========================
API_HOST = "api-football-v1.p.rapidapi.com"
API_KEY = st.secrets["API_FOOTBALL_KEY"]

HEADERS = {
    "X-RapidAPI-Key": API_KEY,
    "X-RapidAPI-Host": API_HOST
}

# =========================
# COMPETITIONS
# =========================
LEAGUES = {
    "Premier League": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61
}

SEASON = 2024

# =========================
# DATA FETCHING
# =========================
@st.cache_data(ttl=3600)
def fetch_standings(league_id):
    url = f"https://{API_HOST}/v3/standings"
    params = {"league": league_id, "season": SEASON}

    r = requests.get(url, headers=HEADERS, params=params)
    data = r.json()

    standings = data["response"][0]["league"]["standings"][0]
    teams = {}

    for t in standings:
        played = t["all"]["played"]

        teams[t["team"]["name"]] = {
            "position": t["rank"],
            "points_per_game": t["points"] / max(played, 1),
            "gf": t["all"]["goals"]["for"] / max(played, 1),
            "ga": t["all"]["goals"]["against"] / max(played, 1),
            "home_ppg": (
                (t["home"]["wins"] * 3 + t["home"]["draws"])
                / max(t["home"]["played"], 1)
            ),
            "away_ppg": (
                (t["away"]["wins"] * 3 + t["away"]["draws"])
                / max(t["away"]["played"], 1)
            )
        }

    return teams


# =========================
# MODEL
# =========================
def predict_match(home, away):
    score = 0

    # League position
    score += (away["position"] - home["position"]) * 1.8

    # Attack vs defence
    score += (home["gf"] - away["ga"]) * 1.5
    score -= (away["gf"] - home["ga"]) * 1.5

    # Home advantage
    score += (home["home_ppg"] - away["away_ppg"]) * 2

    # Overall consistency
    score += (home["points_per_game"] - away["points_per_game"]) * 2

    def sigmoid(x):
        return 1 / (1 + np.exp(-x / 5))

    home_win = sigmoid(score) * 100
    away_win = sigmoid(-score) * 100
    draw = max(0, 100 - home_win - away_win)

    return {
        "home": home_win,
        "draw": draw,
        "away": away_win,
        "confidence": "High" if abs(score) > 6 else "Medium"
    }


# =========================
# UI
# =========================
league_name = st.selectbox("Select Competition", list(LEAGUES.keys()))
league_id = LEAGUES[league_name]

teams_data = fetch_standings(league_id)

teams = sorted(teams_data.keys())

col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", teams)
away_team = col2.selectbox("✈️ Away Team", teams, index=1)

if home_team == away_team:
    st.warning("Please select two different teams.")
    st.stop()

if st.button("🔮 Predict Match", use_container_width=True):
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

    with st.expander("Why this result?"):
        st.write(
            "- League position\n"
            "- Goals scored vs conceded\n"
            "- Home vs away performance\n"
            "- Points per game in this competition"
        )
