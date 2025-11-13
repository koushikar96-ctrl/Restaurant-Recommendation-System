# app.py
import streamlit as st
from recommender import RestaurantRecommender
from ai_mode import run_ai_mode
from utils import load_data, style_background

st.set_page_config(page_title="Restaurant Recommender", layout="wide", page_icon="🍽️")

# Optional: make background look nicer (if you provide assets/bg.jpg)
style_background()

st.title("🍽️ Restaurant Recommendation System")

# Load data
df = load_data("data/restaurants_sample.csv")

# Initialize recommender
rec = RestaurantRecommender(df)

# Sidebar - Filters
st.sidebar.header("Filters")
cuisine = st.sidebar.multiselect("Cuisine", options=sorted(df['cuisine'].unique()), default=[])
diet = st.sidebar.multiselect("Diet Type", options=sorted(df['diet_type'].unique()), default=[])
location = st.sidebar.multiselect("Location", options=sorted(df['location'].unique()), default=[])
online_order = st.sidebar.selectbox("Online Order", options=["Any","Yes","No"])
book_table = st.sidebar.selectbox("Book Table", options=["Any","Yes","No"])
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
budget = st.sidebar.selectbox("Budget", options=["Any","Low","Medium","High"])

st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Filter Mode", "AI Prediction Mode"])
st.sidebar.markdown("---")
show_top10 = st.sidebar.checkbox("Show Top 10 only", value=True)
st.sidebar.markdown("")

# Get recommendations
if mode == "AI Prediction Mode":
    if st.button("Start AI Prediction"):
        user_prefs = run_ai_mode()
        # run_ai_mode returns a dict of filter preferences
        filtered = rec.filter_by_preferences(user_prefs)
        st.success("AI prediction completed — showing recommendations below.")
    else:
        st.info("Click 'Start AI Prediction' to answer a few AI questions.")
        filtered = df.copy()
else:
    if st.button("Get Recommendations"):
        opts = {
            "cuisine": cuisine,
            "diet": diet,
            "location": location,
            "online_order": online_order,
            "book_table": book_table,
            "min_rating": min_rating,
            "budget": budget
        }
        filtered = rec.filter_by_preferences(opts)
    else:
        st.info("Use the filters and click **Get Recommendations**.")
        filtered = df.copy()

# Optionally apply clustering and show cluster visual
if not filtered.empty:
    with st.expander("Recommendation Summary & Visualizations", expanded=True):
        st.write(f"Found **{len(filtered)}** matching restaurants.")
        if show_top10:
            display_df = rec.rank_and_select(filtered, top_n=10)
        else:
            display_df = rec.rank_and_select(filtered, top_n=None)

        st.dataframe(display_df.reset_index(drop=True))

        # Clustering visualization
        if st.button("Show Clusters"):
            fig = rec.plot_clusters(filtered)
            st.pyplot(fig)

        # Rating distribution
        st.write("Rating distribution of the results:")
        st.bar_chart(display_df['rate'].value_counts().sort_index())

        # Cuisine counts
        st.write("Cuisine distribution:")
        st.bar_chart(display_df['cuisine'].value_counts())

else:
    st.warning("No restaurants match the current filters / AI preferences.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ — Streamlit + scikit-learn")

