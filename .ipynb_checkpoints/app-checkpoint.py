import streamlit as st
import joblib
import pandas as pd


#Ladda in modeller och scaler från vår träningsdata
model = joblib.load("Modell_LR.pkt")
scaler = joblib.load("scaler.pkt")

# ladda in dataset och sortera lagen
df = pd.read_csv("df_final.csv")
teams = sorted(df["homeTeam"].unique())

st.title("OUR AMAZING COOL PERFECT MATCH PREDICTOR")

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

#Hämtar in stats för lagen
def get_team_stats(team):

    team_matches = df[(df["homeTeam"] == team) | (df["awayTeam"] == team)]

    latest_match = team_matches.iloc[-1]

    return latest_match

# Hämta in fetures som vi vill använda kan ändre denna senare
features2 = [
    "lastSeasonDivisions",

    "winLoseLast3",
    "winLoseLast5",
    "winLoseLast10",
    "winLoseLast20",
    "winLoseLast38",
    "winLoseLast76",
    
    "deltaShotsOnGoalLast3",
    "deltaShotsOnGoalLast5",
    "deltaShotsOnGoalLast10",
    "deltaShotsOnGoalLast20",
    "deltaShotsOnGoalLast38",
    "deltaShotsOnGoalLast76",
    
    "deltaShotsAgainstGoalLast3",
    "deltaShotsAgainstGoalLast5",
    "deltaShotsAgainstGoalLast10",
    "deltaShotsAgainstGoalLast20",
    "deltaShotsAgainstGoalLast38",
    "deltaShotsAgainstGoalLast76",
    
    "deltaCornersLast3",
    "deltaCornersLast5",
    "deltaCornersLast10",
    "deltaCornersLast20",
    "deltaCornersLast38",
    "deltaCornersLast76",
    
    "deltaFormLast3",
    "deltaFormLast5",
    "deltaFormLast10",
    "deltaFormLast20",
    "deltaFormLast38",
    "deltaFormLast76",
    
    "deltaYellowLast3",
    "deltaYellowLast5",
    "deltaYellowLast10",
    "deltaYellowLast20",
    "deltaYellowLast38",
    "deltaYellowLast76",
    
    "deltaRedLast3",
    "deltaRedLast5",
    "deltaRedLast10",
    "deltaRedLast20",
    "deltaRedLast38",
    "deltaRedLast76",
    
    "deltaFoulsLast3",
    "deltaFoulsLast5",
    "deltaFoulsLast10",
    "deltaFoulsLast20",
    "deltaFoulsLast38",
    "deltaFoulsLast76",
    
    "deltaGoalDiffLast3",
    "deltaGoalDiffLast5",
    "deltaGoalDiffLast10",
    "deltaGoalDiffLast20",
    "deltaGoalDiffLast38",
    "deltaGoalDiffLast76"
]


# Här bygger vi up med hjälp av alla fetures.
def build_features(home, away):

    home_data = get_team_stats(home)
    away_data = get_team_stats(away)

    features = {}

    for f in features2:

        home_val = home_data[f] if f in home_data else 0
        away_val = away_data[f] if f in away_data else 0

        features[f] = home_val - away_val

    return pd.DataFrame([features])


#Knapp för att räkna ut oddsen för matcherna.
if st.button("Predict Match"):

    X = build_features(home_team, away_team)

    X_scaled = scaler.transform(X)

    probs = model.predict_proba(X_scaled)[0]

    st.subheader("Prediction")

    st.write(f"{home_team} win probability: {probs[0]:.2f}")
    st.write(f"Draw probability: {probs[1]:.2f}")
    st.write(f"{away_team} win probability: {probs[2]:.2f}")

