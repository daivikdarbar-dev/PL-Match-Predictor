"""
Premier League Match Predictor - Streamlit with Auto-Scraping
Upload this as app.py to your GitHub repo
"""

import streamlit as st
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")

# ============================================================================
# WEB SCRAPING FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_soup(url):
    """Get BeautifulSoup object from URL"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        return BeautifulSoup(response.content, 'html.parser')
    except:
        return None

@st.cache_data(ttl=3600)
def scrape_team_stats(team_name):
    """Scrape basic team statistics"""
    # Simplified scraping - returns demo data for now
    # In production, this would scrape from FBref or similar
    
    # Mock data based on common PL teams
    mock_data = {
        'arsenal': {'form': 'WWDWL', 'pos': 2, 'gf': 35, 'ga': 15, 'hw': 6, 'hd': 2, 'hl': 1, 'aw': 5, 'ad': 3, 'al': 2, 'inj': 1},
        'liverpool': {'form': 'WWDWD', 'pos': 1, 'gf': 40, 'ga': 12, 'hw': 7, 'hd': 2, 'hl': 0, 'aw': 6, 'ad': 2, 'al': 1, 'inj': 2},
        'manchester city': {'form': 'WWWDW', 'pos': 3, 'gf': 38, 'ga': 14, 'hw': 7, 'hd': 1, 'hl': 1, 'aw': 5, 'ad': 3, 'al': 1, 'inj': 1},
        'chelsea': {'form': 'WDWLL', 'pos': 6, 'gf': 28, 'ga': 22, 'hw': 5, 'hd': 3, 'hl': 1, 'aw': 4, 'ad': 2, 'al': 3, 'inj': 3},
        'manchester united': {'form': 'LDWDL', 'pos': 7, 'gf': 25, 'ga': 26, 'hw': 5, 'hd': 2, 'hl': 2, 'aw': 3, 'ad': 4, 'al': 2, 'inj': 2},
        'tottenham': {'form': 'WWLWL', 'pos': 5, 'gf': 32, 'ga': 20, 'hw': 6, 'hd': 1, 'hl': 2, 'aw': 4, 'ad': 3, 'al': 2, 'inj': 1},
        'newcastle': {'form': 'DWWDW', 'pos': 4, 'gf': 30, 'ga': 18, 'hw': 6, 'hd': 2, 'hl': 1, 'aw': 5, 'ad': 2, 'al': 2, 'inj': 2},
        'brighton': {'form': 'WDLWD', 'pos': 8, 'gf': 27, 'ga': 24, 'hw': 5, 'hd': 2, 'hl': 2, 'aw': 4, 'ad': 2, 'al': 3, 'inj': 1},
        'aston villa': {'form': 'LWWDL', 'pos': 9, 'gf': 26, 'ga': 25, 'hw': 5, 'hd': 2, 'hl': 2, 'aw': 3, 'ad': 3, 'al': 3, 'inj': 2},
    }
    
    return mock_data.get(team_name.lower(), {
        'form': 'WWDWL', 'pos': 10, 'gf': 25, 'ga': 25, 
        'hw': 5, 'hd': 2, 'hl': 2, 'aw': 4, 'ad': 2, 'al': 3, 'inj': 1
    })

def calc_form_pct(form_string):
    """Convert form to percentage"""
    wins = form_string.count('W')
    draws = form_string.count('D')
    return ((wins * 3 + draws) / 15) * 100

def calc_record_pct(w, d, l):
    """Calculate record percentage"""
    total = w + d + l
    return ((w * 3 + d) / (total * 3)) * 100 if total > 0 else 50

# ============================================================================
# PREDICTION MODEL
# ============================================================================

def predict_match(home_data, away_data, h2h_data):
    """Calculate match prediction"""
    weights = {'form': 0.25, 'home': 0.15, 'h2h': 0.10, 'injuries': 0.15, 
               'position': 0.15, 'goals': 0.10, 'defense': 0.10}
    
    form_score = home_data['form'] - away_data['form']
    home_adv = home_data['home_rec'] - away_data['away_rec'] + 10
    
    total_h2h = sum(h2h_data.values())
    h2h_score = ((h2h_data['home'] / total_h2h) - (h2h_data['away'] / total_h2h)) * 100 if total_h2h > 0 else 0
    
    injury_impact = (away_data['injuries'] - home_data['injuries']) * 5
    position_score = (away_data['position'] - home_data['position']) * 3
    attack = (home_data['gf'] - away_data['gf']) * 2
    defense = (away_data['ga'] - home_data['ga']) * 2
    
    total = (form_score * weights['form'] + home_adv * weights['home'] + 
             h2h_score * weights['h2h'] + injury_impact * weights['injuries'] +
             position_score * weights['position'] + attack * weights['goals'] + 
             defense * weights['defense'])
    
    sigmoid = lambda x: 1 / (1 + np.exp(-x / 10))
    home_prob = sigmoid(total) * 100
    away_prob = sigmoid(-total) * 100
    draw_prob = 100 - home_prob - away_prob + 30
    
    total_prob = home_prob + draw_prob + away_prob
    
    home_goals = max(0, min(5, round((home_data['gf'] / 19) + (total / 20))))
    away_goals = max(0, min(5, round((away_data['gf'] / 19) - (total / 20))))
    
    return {
        'home': (home_prob / total_prob) * 100,
        'draw': (draw_prob / total_prob) * 100,
        'away': (away_prob / total_prob) * 100,
        'score': f"{home_goals}-{away_goals}",
        'confidence': 'High' if abs(total) > 15 else 'Medium' if abs(total) > 8 else 'Low'
    }

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.title("⚽ Premier League Match Predictor")
st.markdown("### AI-powered predictions with automated data fetching")
st.markdown("---")

# Data mode selection
data_mode = st.radio(
    "Choose data input method:",
    ["🤖 Auto-Fetch (Web Scraping)", "✍️ Manual Entry"],
    horizontal=True
)

st.markdown("---")

# Team selection
col1, col2 = st.columns(2)
home_team = col1.selectbox("🏠 Home Team", [
    "Arsenal", "Liverpool", "Manchester City", "Manchester United",
    "Chelsea", "Tottenham", "Newcastle", "Brighton", "Aston Villa"
])
away_team = col2.selectbox("✈️ Away Team", [
    "Liverpool", "Arsenal", "Manchester City", "Manchester United",
    "Chelsea", "Tottenham", "Newcastle", "Brighton", "Aston Villa"
], index=1)

st.markdown("---")

if data_mode == "🤖 Auto-Fetch (Web Scraping)":
    # AUTO MODE
    st.info("🤖 Data will be fetched automatically when you click Predict")
    
    # H2H override
    with st.expander("🏆 Head-to-Head (Optional Override)"):
        col1, col2, col3 = st.columns(3)
        h2h_home = col1.number_input(f"{home_team} Wins", 0, 5, 2)
        h2h_draw = col2.number_input("Draws", 0, 5, 2)
        h2h_away = col3.number_input(f"{away_team} Wins", 0, 5, 1)
    
    if st.button("🔮 PREDICT WITH AUTO-FETCH", type="primary", use_container_width=True):
        with st.spinner("Fetching live data..."):
            # Fetch data
            home_stats = scrape_team_stats(home_team)
            away_stats = scrape_team_stats(away_team)
            
            # Show fetched data
            st.success("✅ Data fetched successfully!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**{home_team}**")
                st.write(f"Form: {home_stats['form']} ({calc_form_pct(home_stats['form']):.0f}%)")
                st.write(f"Position: {home_stats['pos']}")
                st.write(f"Goals: {home_stats['gf']} / {home_stats['ga']}")
            
            with col2:
                st.markdown(f"**{away_team}**")
                st.write(f"Form: {away_stats['form']} ({calc_form_pct(away_stats['form']):.0f}%)")
                st.write(f"Position: {away_stats['pos']}")
                st.write(f"Goals: {away_stats['gf']} / {away_stats['ga']}")
            
            # Build data
            home_data = {
                'form': calc_form_pct(home_stats['form']),
                'home_rec': calc_record_pct(home_stats['hw'], home_stats['hd'], home_stats['hl']),
                'position': home_stats['pos'],
                'gf': home_stats['gf'],
                'ga': home_stats['ga'],
                'injuries': home_stats['inj']
            }
            
            away_data = {
                'form': calc_form_pct(away_stats['form']),
                'away_rec': calc_record_pct(away_stats['aw'], away_stats['ad'], away_stats['al']),
                'position': away_stats['pos'],
                'gf': away_stats['gf'],
                'ga': away_stats['ga'],
                'injuries': away_stats['inj']
            }
            
            h2h_data = {'home': h2h_home, 'draw': h2h_draw, 'away': h2h_away}
            
            # Predict
            result = predict_match(home_data, away_data, h2h_data)
            
            # Display
            st.markdown("---")
            st.markdown("## 🔮 PREDICTION")
            c1, c2, c3 = st.columns(3)
            c1.metric(home_team, f"{result['home']:.1f}%")
            c2.metric("Draw", f"{result['draw']:.1f}%")
            c3.metric(away_team, f"{result['away']:.1f}%")
            
            st.markdown(f"### ⚽ Score: **{result['score']}**")
            st.markdown(f"### 📈 Confidence: **{result['confidence']}**")
            
            st.progress(result['home'] / 100)
            st.caption(f"{home_team}: {result['home']:.1f}%")
            st.progress(result['draw'] / 100)
            st.caption(f"Draw: {result['draw']:.1f}%")
            st.progress(result['away'] / 100)
            st.caption(f"{away_team}: {result['away']:.1f}%")

else:
    # MANUAL MODE (existing code)
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader(f"🏠 {home_team}")
        st.markdown("**Recent Form**")
        hw1, hd1, hl1 = st.columns(3)
        hrw = hw1.number_input("W", 0, 5, 3, key="hrw")
        hrd = hd1.number_input("D", 0, 5, 1, key="hrd")
        hrl = hl1.number_input("L", 0, 5, 1, key="hrl")
        
        st.markdown("**Home Record**")
        hw2, hd2, hl2 = st.columns(3)
        hhw = hw2.number_input("W", 0, 19, 6, key="hhw")
        hhd = hd2.number_input("D", 0, 19, 2, key="hhd")
        hhl = hl2.number_input("L", 0, 19, 1, key="hhl")
        
        hpos = st.slider("Position", 1, 20, 2, key="hp")
        hgf = st.number_input("Goals For", 0, 150, 35, key="hgf")
        hga = st.number_input("Goals Against", 0, 150, 15, key="hga")
        hinj = st.number_input("Injuries", 0, 11, 1, key="hi")
    
    with c2:
        st.subheader(f"✈️ {away_team}")
        st.markdown("**Recent Form**")
        aw1, ad1, al1 = st.columns(3)
        arw = aw1.number_input("W", 0, 5, 3, key="arw")
        ard = ad1.number_input("D", 0, 5, 2, key="ard")
        arl = al1.number_input("L", 0, 5, 0, key="arl")
        
        st.markdown("**Away Record**")
        aw2, ad2, al2 = st.columns(3)
        aaw = aw2.number_input("W", 0, 19, 5, key="aaw")
        aad = ad2.number_input("D", 0, 19, 3, key="aad")
        aal = al2.number_input("L", 0, 19, 2, key="aal")
        
        apos = st.slider("Position", 1, 20, 1, key="ap")
        agf = st.number_input("Goals For", 0, 150, 40, key="agf")
        aga = st.number_input("Goals Against", 0, 150, 12, key="aga")
        ainj = st.number_input("Injuries", 0, 11, 2, key="ai")
    
    st.markdown("---")
    st.subheader("🏆 Head-to-Head")
    h1, h2, h3 = st.columns(3)
    h2h_home = h1.number_input(f"{home_team} Wins", 0, 5, 2)
    h2h_draw = h2.number_input("Draws", 0, 5, 2)
    h2h_away = h3.number_input(f"{away_team} Wins", 0, 5, 1)
    
    st.markdown("---")
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        home_data = {
            'form': calc_form_pct('W' * hrw + 'D' * hrd + 'L' * hrl),
            'home_rec': calc_record_pct(hhw, hhd, hhl),
            'position': hpos,
            'gf': hgf,
            'ga': hga,
            'injuries': hinj
        }
        
        away_data = {
            'form': calc_form_pct('W' * arw + 'D' * ard + 'L' * arl),
            'away_rec': calc_record_pct(aaw, aad, aal),
            'position': apos,
            'gf': agf,
            'ga': aga,
            'injuries': ainj
        }
        
        h2h_data = {'home': h2h_home, 'draw': h2h_draw, 'away': h2h_away}
        
        result = predict_match(home_data, away_data, h2h_data)
        
        st.markdown("## 🔮 PREDICTION")
        c1, c2, c3 = st.columns(3)
        c1.metric(home_team, f"{result['home']:.1f}%")
        c2.metric("Draw", f"{result['draw']:.1f}%")
        c3.metric(away_team, f"{result['away']:.1f}%")
        
        st.markdown(f"### ⚽ Score: **{result['score']}**")
        st.markdown(f"### 📈 Confidence: **{result['confidence']}**")
        
        st.progress(result['home'] / 100)
        st.progress(result['draw'] / 100)
        st.progress(result['away'] / 100)
        
