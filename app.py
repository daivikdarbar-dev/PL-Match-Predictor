import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
import re

# PAGE CONFIG
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Premier League Match Predictor")
st.caption("Transparent, rule-based model using public FBref data")

# DATA SCRAPING

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

@st.cache_data(ttl=1800)
def fetch_pl_table():
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    response = requests.get(url, headers=HEADERS, timeout=15)
    soup = BeautifulSoup(response.content, "html.parser")

    table = soup.find("table", class_="stats_table")
    rows = table.tbody.find_all("tr")

    data = {}

    for row in rows:
        team = row.find("th", {"data-stat": "team"}).text.strip()

        def val(stat):
            cell = row.find("td", {"data-stat": stat})
            return int(cell.text.strip()) if cell and cell.text.strip() != "" else 0

        data[team] = {
            "position": val("rank"),
            "played": val("games"),
            "gf": val("goals_for"),
            "ga": val("goals_against"),
            "points": val("points"),
            "form": parse_form(row)
        }

    return data

def parse_form(row):
    form_cell = row.find("td", {"data-stat": "last_5"})
    if not form_cell:
        return 50

    form = form_cell.text.strip()
    pts = form.count("W") * 3 + form.count("D")
    return (pts / 15) * 100

# MODEL

def predict_match(home, away):
    weights = {
        "form": 0.30,
        "position": 0.15,
        "attack": 0.20,
        "defense": 0.15,
        "home_adv": 0.20
    }

    score = (
        (home["form"] - away["form"]) * weights["form"] +
        (away["position"] - home["position"]) * weights["position"] +
        (home["gf"] - away["gf"]) * weights["attack"] +
        (away["ga"] - home["ga"]) * weights["defense"] +
        10 * weights["home_adv"]
    )

    home_win = 1 / (1 + np.exp(-score / 10))
    away_win = 1 - home_win

    draw = max(0.18, 0.30 - abs(score) * 0.01)

    total = home_win + away_win + draw

    return {
        "home": home_win / total * 100,
        "draw": draw / total * 100,
        "away": away_win / total * 100,
        "confidence": (
            "High" if abs(score) > 15 else
            "Medium" if abs(score) > 7 else
            "Low"
        )
    }

# UI

teams_data = fetch_pl_table()
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
    st.caption("Model notes: rule-based weights, no betting odds, no hidden ML")

