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
@st.cache_data(ttl=3600)
def fetch_pl_table():
    url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(response.text, "html.parser")

    # FBref tables are hidden inside HTML comments
    comments = soup.find_all(
        string=lambda text: isinstance(text, str) and "stats_table" in text
    )

    table_html = None
    for comment in comments:
        if "Premier League" in comment:
            table_html = comment
            break

    if table_html is None:
        return None

    table_soup = BeautifulSoup(table_html, "html.parser")
    table = table_soup.find("table", class_="stats_table")

    teams = {}

    for row in table.find("tbody").find_all("tr"):
        try:
            team = row.find("th").text.strip()
            cells = row.find_all("td")

            position = int(cells[0].text)
            played = int(cells[2].text)
            gf = int(cells[5].text)
            ga = int(cells[6].text)

            teams[team] = {
                "position": position,
                "gf": gf / max(played, 1),
                "ga": ga / max(played, 1)
            }
        except:
            continue

    return teams

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

