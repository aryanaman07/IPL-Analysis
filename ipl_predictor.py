import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load datasets
matches = pd.read_csv(r"C:\Users\Aryanaman\Desktop\EDA\matches.csv")
deliveries = pd.read_csv(r"C:\Users\Aryanaman\Desktop\EDA\deliveries.csv")

st.title("🏏 IPL Match Winner Predictor")

# Drop rows with missing winner
matches = matches.dropna(subset=['winner'])

# Encode categorical columns
label_encoders = {}
for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']:
    le = LabelEncoder()
    matches[col] = le.fit_transform(matches[col])
    label_encoders[col] = le

# Features and target
features = ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']
X = matches[features]
y = matches['winner']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI Inputs
st.subheader("Enter Match Details")

team1 = st.selectbox("Team 1", label_encoders['team1'].classes_)
team2 = st.selectbox("Team 2", label_encoders['team2'].classes_)
toss_winner = st.selectbox("Toss Winner", label_encoders['toss_winner'].classes_)
toss_decision = st.selectbox("Toss Decision", label_encoders['toss_decision'].classes_)
venue = st.selectbox("Venue", label_encoders['venue'].classes_)

if st.button("Predict Winner"):
    input_df = pd.DataFrame({
        'team1': [label_encoders['team1'].transform([team1])[0]],
        'team2': [label_encoders['team2'].transform([team2])[0]],
        'toss_winner': [label_encoders['toss_winner'].transform([toss_winner])[0]],
        'toss_decision': [label_encoders['toss_decision'].transform([toss_decision])[0]],
        'venue': [label_encoders['venue'].transform([venue])[0]]
    })

    prediction = model.predict(input_df)[0]
    winner = label_encoders['winner'].inverse_transform([prediction])[0]
    st.success(f"🏆 Predicted Winner: {winner}")

st.caption("Built with IPL data. Replace 'matches.csv' and 'deliveries.csv' with full datasets for accurate prediction.")
