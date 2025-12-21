# Premier League Match Predictor ⚽

## Overview

This project is an interactive Streamlit web application that predicts the outcome of an English Premier League match.
It uses multiple match-related factors such as recent form, home advantage, injuries, league position, goals scored, and head-to-head results to estimate win probabilities and a likely scoreline.

The project is designed for learning, experimentation, and demonstrating applied football analytics through a clean and simple interface.

---

## Features

* Interactive Streamlit dashboard
* Predicts match outcome: Home Win, Draw, or Away Win
* Probability breakdown with visual indicators
* Predicted scoreline
* Confidence level for each prediction
* Fully customizable inputs for both teams

---

## Prediction Logic

The prediction is based on a weighted scoring model that evaluates:

* Recent form (last 5 matches)
* Home and away performance
* League table position
* Goals scored and conceded
* Key injuries and suspensions
* Head-to-head record

The combined score is converted into probabilities using a sigmoid function.

### Model Weights

* Form: 25%
* Home Advantage: 15%
* Injuries & Suspensions: 15%
* League Position: 15%
* Head-to-Head: 10%
* Attack (Goals Scored): 10%
* Defense (Goals Conceded): 10%

---

## Output

The app provides:

* Home win probability
* Draw probability
* Away win probability
* Predicted scoreline
* Confidence level (Low / Medium / High)

All predictions depend on historical trends and user-provided inputs.

---

## Tech Stack

* Python
* Streamlit
* NumPy

---

## Project Structure

```
├── .streamlit/
├── app.py
├── requirements.txt
└── README.md
```

* `app.py` — Main Streamlit application
* `requirements.txt` — Project dependencies
* `.streamlit/` — Streamlit configuration files

---

## How to Run

### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Use Case

This project is suitable for:

* Football analytics practice
* Machine learning demonstrations
* Academic submissions
* Portfolio projects
* Interactive sports data exploration

It is not intended for betting or financial decision-making.

---

## Limitations

* Uses a rule-based weighted model, not a trained ML model
* Depends on manual user inputs
* Does not use live or real-time match data

---

## Future Improvements

* Train models on historical Premier League datasets
* Add real-time data integration
* Compare multiple prediction models
* Deploy as a public web application

---

## Disclaimer

This project is for educational purposes only. Predictions are not guaranteed to be accurate.

---

