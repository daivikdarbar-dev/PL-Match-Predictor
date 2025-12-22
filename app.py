import streamlit as st
import numpy as np

#CONFIG

st.set_page_config(page_title="PL Match Predictor", page_icon="⚽", layout="wide")

#Prediction Model

def predict_match(home_data, away_data, h2h_data):
    """Calculate match prediction"""
    weights = {'form': 0.25, 'home': 0.15, 'h2h': 0.10, 'injuries': 0.15, 
               'position': 0.15, 'goals': 0.10, 'defense': 0.10}
    
    # Calculate scores
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
    
    # Convert to probabilities
    sigmoid = lambda x: 1 / (1 + np.exp(-x / 10))
    home_prob = sigmoid(total) * 100
    away_prob = sigmoid(-total) * 100
    draw_prob = 100 - home_prob - away_prob + 30
    
    total_prob = home_prob + draw_prob + away_prob
    
    # Predicted score
    home_goals = max(0, min(5, round((home_data['gf'] / 19) + (total / 20))))
    away_goals = max(0, min(5, round((away_data['gf'] / 19) - (total / 20))))
    
    return {
        'home': (home_prob / total_prob) * 100,
        'draw': (draw_prob / total_prob) * 100,
        'away': (away_prob / total_prob) * 100,
        'score': f"{home_goals}-{away_goals}",
        'confidence': 'High' if abs(total) > 15 else 'Medium' if abs(total) > 8 else 'Low'
    }

def calc_form(w, d, l):
    """Calculate form percentage"""
    return ((w * 3 + d) / 15) * 100 if (w + d + l) <= 5 else 50

def calc_record(w, d, l):
    """Calculate home/away record percentage"""
    total = w + d + l
    return ((w * 3 + d) / (total * 3)) * 100 if total > 0 else 50

#Main App

st.title("⚽ Premier League Match Predictor")
st.markdown("---")

# Team Names
c1, c2 = st.columns(2)
home_team = c1.text_input("🏠 Home Team", "Arsenal")
away_team = c2.text_input("✈️ Away Team", "Liverpool")

st.markdown("---")

# Data Entry
c1, c2 = st.columns(2)

with c1:
    st.subheader(f"🏠 {home_team}")
    st.markdown("**Recent Form (Last 5)**")
    hw, hd, hl = st.columns(3)
    hrw = hw.number_input("W", 0, 5, 3, key="hrw")
    hrd = hd.number_input("D", 0, 5, 1, key="hrd")
    hrl = hl.number_input("L", 0, 5, 1, key="hrl")
    
    st.markdown("**Home Record**")
    hw, hd, hl = st.columns(3)
    hhw = hw.number_input("W", 0, 19, 6, key="hhw")
    hhd = hd.number_input("D", 0, 19, 2, key="hhd")
    hhl = hl.number_input("L", 0, 19, 1, key="hhl")
    
    hpos = st.slider("Position", 1, 20, 2, key="hp")
    hgf = st.number_input("Goals For", 0, 150, 35, key="hgf")
    hga = st.number_input("Goals Against", 0, 150, 15, key="hga")
    hinj = st.number_input("Injuries", 0, 11, 1, key="hi")

with c2:
    st.subheader(f"✈️ {away_team}")
    st.markdown("**Recent Form (Last 5)**")
    aw, ad, al = st.columns(3)
    arw = aw.number_input("W", 0, 5, 3, key="arw")
    ard = ad.number_input("D", 0, 5, 2, key="ard")
    arl = al.number_input("L", 0, 5, 0, key="arl")
    
    st.markdown("**Away Record**")
    aw, ad, al = st.columns(3)
    aaw = aw.number_input("W", 0, 19, 5, key="aaw")
    aad = ad.number_input("D", 0, 19, 3, key="aad")
    aal = al.number_input("L", 0, 19, 2, key="aal")
    
    apos = st.slider("Position", 1, 20, 1, key="ap")
    agf = st.number_input("Goals For", 0, 150, 40, key="agf")
    aga = st.number_input("Goals Against", 0, 150, 12, key="aga")
    ainj = st.number_input("Injuries", 0, 11, 2, key="ai")

st.markdown("---")
st.subheader("🏆 Head-to-Head (Last 5)")
h1, h2, h3 = st.columns(3)
h2h_home = h1.number_input(f"{home_team} Wins", 0, 5, 2)
h2h_draw = h2.number_input("Draws", 0, 5, 2)
h2h_away = h3.number_input(f"{away_team} Wins", 0, 5, 1)

st.markdown("---")

# Predict
if st.button("🔮 PREDICT", type="primary", use_container_width=True):
    # Build data
    home_data = {
        'form': calc_form(hrw, hrd, hrl),
        'home_rec': calc_record(hhw, hhd, hhl),
        'position': hpos,
        'gf': hgf,
        'ga': hga,
        'injuries': hinj
    }
    
    away_data = {
        'form': calc_form(arw, ard, arl),
        'away_rec': calc_record(aaw, aad, aal),
        'position': apos,
        'gf': agf,
        'ga': aga,
        'injuries': ainj
    }
    
    h2h_data = {'home': h2h_home, 'draw': h2h_draw, 'away': h2h_away}
    
    # Get prediction
    result = predict_match(home_data, away_data, h2h_data)
    
    # Display
    st.markdown("## 🔮 PREDICTION")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{home_team}", f"{result['home']:.1f}%")
    c2.metric("Draw", f"{result['draw']:.1f}%")
    c3.metric(f"{away_team}", f"{result['away']:.1f}%")
    
    st.markdown(f"### ⚽ Score: **{result['score']}**")
    st.markdown(f"### 📈 Confidence: **{result['confidence']}**")
    
    # Visual bars
    st.progress(result['home'] / 100)
    st.progress(result['draw'] / 100)
    st.progress(result['away'] / 100)
