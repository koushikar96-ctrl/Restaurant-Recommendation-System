# ai_mode.py
import streamlit as st

def run_ai_mode():
    st.header("AI Prediction — tell me about your preferences")
    c1, c2 = st.columns(2)
    cuisine = c1.multiselect("Which cuisines do you prefer?", ["North Indian","South Indian","Chinese","Italian","Mexican","Desserts"])
    diet = c2.multiselect("Diet preference", ["Vegetarian","Non-Vegetarian","Vegan","Jain"])
    location = st.selectbox("Location preference", ["Any","Downtown","Suburb","Near Campus"])
    online = st.selectbox("Do you want online ordering?", ["Any","Yes","No"])
    book = st.selectbox("Need table booking option?", ["Any","Yes","No"])
    rating = st.slider("Minimum rating you'd accept", 0.0, 5.0, 3.5, 0.1)
    budget = st.selectbox("Budget", ["Any","Low","Medium","High"])

    if st.button("Submit Preferences"):
        prefs = {
            "cuisine": cuisine,
            "diet": diet,
            "location": [] if location=="Any" else [location],
            "online_order": online,
            "book_table": book,
            "min_rating": rating,
            "budget": budget
        }
        st.write("Thanks — processing recommendations...")
        return prefs
    else:
        st.stop()
