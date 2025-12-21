"""
Premier League Match Predictor - Streamlit Dashboard

INSTALLATION (in Google Colab):
!pip install streamlit

HOW TO RUN IN COLAB:
1. Save this code as a file: predictor.py
2. Run: !streamlit run predictor.py & npx localtunnel --port 8501

OR LOCALLY:
streamlit run predictor.py
"""

import streamlit as st
import numpy as np

# Page Config

st.set_page_config(
    page_title="PL Match Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3em;
        background-color: #00ff87;
        color: black;
        font-weight: bold;
        font-size: 1.2em;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

#Prediction Model

class PLMatchPredictor:
    def __init__(self):
        self.weights = {
            'form': 0.25,
            'home_advantage': 0.15,
            'head_to_head': 0.10,
            'injuries': 0.15,
            'table_position': 0.15,
            'goals': 0.10,
            'defense': 0.10
        }
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x / 10))
    
    def predict_match(self, home_data, away_data, h2h_data):
        # Calculate scores
        form_score = home_data['recent_form'] - away_data['recent_form']
        home_adv_score = home_data['home_record'] - away_data['away_record'] + 10
        
        total_h2h = h2h_data['home_wins'] + h2h_data['draws'] + h2h_data['away_wins']
        h2h_score = ((h2h_data['home_wins'] / total_h2h) - (h2h_data['away_wins'] / total_h2h)) * 100 if total_h2h > 0 else 0
        
        injury_score = ((away_data['key_injuries'] + away_data['suspensions']) - 
                       (home_data['key_injuries'] + home_data['suspensions']))
        injury_impact = injury_score * 5
        
        position_score = (away_data['table_position'] - home_data['table_position']) * 3
        attack_score = (home_data['goals_scored'] - away_data['goals_scored']) * 2
        defense_score = (away_data['goals_conceded'] - home_data['goals_conceded']) * 2
        
        # Weighted total
        total_score = (
            (form_score * self.weights['form']) +
            (home_adv_score * self.weights['home_advantage']) +
            (h2h_score * self.weights['head_to_head']) +
            (injury_impact * self.weights['injuries']) +
            (position_score * self.weights['table_position']) +
            (attack_score * self.weights['goals']) +
            (defense_score * self.weights['defense'])
        )
        
        # Convert to probabilities
        home_win_prob = self.sigmoid(total_score) * 100
        away_win_prob = self.sigmoid(-total_score) * 100
        draw_prob = 100 - home_win_prob - away_win_prob + 30
        
        # Normalize
        total = home_win_prob + draw_prob + away_win_prob
        
        return {
            'home_win_prob': (home_win_prob / total) * 100,
            'draw_prob': (draw_prob / total) * 100,
            'away_win_prob': (away_win_prob / total) * 100,
            'confidence': 'High' if abs(total_score) > 15 else 'Medium' if abs(total_score) > 8 else 'Low',
            'predicted_score': self._predict_score(home_data, away_data, total_score)
        }
    
    def _predict_score(self, home_data, away_data, advantage):
        home_goals = max(0, round((home_data['goals_scored'] / 19) + (advantage / 20)))
        away_goals = max(0, round((away_data['goals_scored'] / 19) - (advantage / 20)))
        return f"{min(home_goals, 5)}-{min(away_goals, 5)}"

#Main App

def main():
    # Header
    st.title("⚽ Premier League Match Predictor")
    st.markdown("### AI-powered predictions using multiple factors")
    st.markdown("---")
    
    # Team Names
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.text_input("🏠 Home Team", value="Arsenal", key="home_team")
    with col2:
        away_team = st.text_input("✈️ Away Team", value="Liverpool", key="away_team")
    
    st.markdown("---")
    
    # Main Data Entry
    col_home, col_away = st.columns(2)
    
    # ========== HOME TEAM DATA ==========
    with col_home:
        st.markdown(f"### 🏠 {home_team} Data")
        
        st.markdown("**📊 Recent Form (Last 5 matches)**")
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            home_recent_wins = st.number_input("Wins", 0, 5, 3, key="hrw")
        with h_col2:
            home_recent_draws = st.number_input("Draws", 0, 5, 1, key="hrd")
        with h_col3:
            home_recent_losses = st.number_input("Losses", 0, 5, 1, key="hrl")
        
        st.markdown("**🏠 Home Record (This Season)**")
        h_col1, h_col2, h_col3 = st.columns(3)
        with h_col1:
            home_home_wins = st.number_input("Home Wins", 0, 19, 6, key="hhw")
        with h_col2:
            home_home_draws = st.number_input("Home Draws", 0, 19, 2, key="hhd")
        with h_col3:
            home_home_losses = st.number_input("Home Losses", 0, 19, 1, key="hhl")
        
        st.markdown("**📈 Season Stats**")
        home_position = st.slider("League Position", 1, 20, 2, key="hp")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            home_goals_scored = st.number_input("Goals Scored", 0, 150, 35, key="hgs")
        with h_col2:
            home_goals_conceded = st.number_input("Goals Conceded", 0, 150, 15, key="hgc")
        
        st.markdown("**🏥 Team News**")
        h_col1, h_col2 = st.columns(2)
        with h_col1:
            home_injuries = st.number_input("Key Injuries", 0, 11, 1, key="hi")
        with h_col2:
            home_suspensions = st.number_input("Suspensions", 0, 11, 0, key="hs")
    
    # ========== AWAY TEAM DATA ==========
    with col_away:
        st.markdown(f"### ✈️ {away_team} Data")
        
        st.markdown("**📊 Recent Form (Last 5 matches)**")
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            away_recent_wins = st.number_input("Wins", 0, 5, 3, key="arw")
        with a_col2:
            away_recent_draws = st.number_input("Draws", 0, 5, 2, key="ard")
        with a_col3:
            away_recent_losses = st.number_input("Losses", 0, 5, 0, key="arl")
        
        st.markdown("**✈️ Away Record (This Season)**")
        a_col1, a_col2, a_col3 = st.columns(3)
        with a_col1:
            away_away_wins = st.number_input("Away Wins", 0, 19, 5, key="aaw")
        with a_col2:
            away_away_draws = st.number_input("Away Draws", 0, 19, 3, key="aad")
        with a_col3:
            away_away_losses = st.number_input("Away Losses", 0, 19, 2, key="aal")
        
        st.markdown("**📈 Season Stats**")
        away_position = st.slider("League Position", 1, 20, 1, key="ap")
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            away_goals_scored = st.number_input("Goals Scored", 0, 150, 40, key="ags")
        with a_col2:
            away_goals_conceded = st.number_input("Goals Conceded", 0, 150, 12, key="agc")
        
        st.markdown("**🏥 Team News**")
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            away_injuries = st.number_input("Key Injuries", 0, 11, 2, key="ai")
        with a_col2:
            away_suspensions = st.number_input("Suspensions", 0, 11, 1, key="as")
    
    st.markdown("---")
    
    # ========== HEAD TO HEAD ==========
    st.markdown("### 🏆 Head-to-Head (Last 5 meetings)")
    h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
    with h2h_col1:
        h2h_home_wins = st.number_input(f"{home_team} Wins", 0, 5, 2, key="h2h_hw")
    with h2h_col2:
        h2h_draws = st.number_input("Draws", 0, 5, 2, key="h2h_d")
    with h2h_col3:
        h2h_away_wins = st.number_input(f"{away_team} Wins", 0, 5, 1, key="h2h_aw")
    
    st.markdown("---")
    
    # ========== PREDICT BUTTON ==========
    if st.button("🔮 PREDICT MATCH", type="primary"):
        # Calculate form percentages
        home_form_points = (home_recent_wins * 3) + home_recent_draws
        home_form = (home_form_points / 15) * 100
        
        home_total_games = home_home_wins + home_home_draws + home_home_losses
        if home_total_games > 0:
            home_record_points = (home_home_wins * 3) + home_home_draws
            home_record = (home_record_points / (home_total_games * 3)) * 100
        else:
            home_record = 50
        
        away_form_points = (away_recent_wins * 3) + away_recent_draws
        away_form = (away_form_points / 15) * 100
        
        away_total_games = away_away_wins + away_away_draws + away_away_losses
        if away_total_games > 0:
            away_record_points = (away_away_wins * 3) + away_away_draws
            away_record = (away_record_points / (away_total_games * 3)) * 100
        else:
            away_record = 50
        
        # Build data
        home_data = {
            'recent_form': home_form,
            'home_record': home_record,
            'table_position': home_position,
            'goals_scored': home_goals_scored,
            'goals_conceded': home_goals_conceded,
            'key_injuries': home_injuries,
            'suspensions': home_suspensions
        }
        
        away_data = {
            'recent_form': away_form,
            'away_record': away_record,
            'table_position': away_position,
            'goals_scored': away_goals_scored,
            'goals_conceded': away_goals_conceded,
            'key_injuries': away_injuries,
            'suspensions': away_suspensions
        }
        
        h2h_data = {
            'home_wins': h2h_home_wins,
            'draws': h2h_draws,
            'away_wins': h2h_away_wins
        }
        
        # Get prediction
        predictor = PLMatchPredictor()
        result = predictor.predict_match(home_data, away_data, h2h_data)
        
        # Display Results
        st.markdown("---")
        st.markdown(f"## 🔮 Prediction: {home_team} vs {away_team}")
        
        # Probabilities
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"🏠 {home_team} Win", f"{result['home_win_prob']:.1f}%")
        with col2:
            st.metric("🤝 Draw", f"{result['draw_prob']:.1f}%")
        with col3:
            st.metric(f"✈️ {away_team} Win", f"{result['away_win_prob']:.1f}%")
        
        # Progress bars for visual representation
        st.markdown("**Visual Probabilities:**")
        st.progress(result['home_win_prob'] / 100)
        st.caption(f"{home_team} Win: {result['home_win_prob']:.1f}%")
        
        st.progress(result['draw_prob'] / 100)
        st.caption(f"Draw: {result['draw_prob']:.1f}%")
        
        st.progress(result['away_win_prob'] / 100)
        st.caption(f"{away_team} Win: {result['away_win_prob']:.1f}%")
        
        # Most likely outcome
        probs = {
            f'{home_team} Win': result['home_win_prob'],
            'Draw': result['draw_prob'],
            f'{away_team} Win': result['away_win_prob']
        }
        most_likely = max(probs, key=probs.get)
        
        # Final prediction box
        st.markdown(f"""
        <div class="prediction-box">
            <h2 style="margin:0;">⚽ Predicted Score: {result['predicted_score']}</h2>
            <h3 style="margin:0.5rem 0;">🎯 Most Likely: {most_likely}</h3>
            <p style="margin:0;">📈 Confidence: {result['confidence']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Model Weights:** Form (25%) • Home Advantage (15%) • Injuries (15%) • Position (15%) • H2H (10%) • Attack (10%) • Defense (10%)
    """)

if __name__ == "__main__":
    main()
