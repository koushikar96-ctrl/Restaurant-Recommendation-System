# Restaurant Recommendation System

Streamlit-based restaurant recommender with filtering, AI questionnaire mode, and basic ML features (KMeans clustering, CART example).

## Run locally
1. Create virtual env:
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt

markdown
Copy code
2. Put `data/restaurants_sample.csv` in `data/` (already included)
3. Run:
streamlit run app.py



## Features
- Filter by cuisine, diet, rating, budget, location, online order, book table
- AI Prediction Mode (quick questionnaire)
- Top 10 results option
- Clustering visualization (KMeans)
