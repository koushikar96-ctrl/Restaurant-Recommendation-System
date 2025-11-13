#clustering logic
# recommender.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

class RestaurantRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare()

    def _prepare(self):
        # cleaning / simple conversions
        self.df['rate'] = pd.to_numeric(self.df['rate'], errors='coerce').fillna(0)
        self.df['votes'] = pd.to_numeric(self.df['votes'], errors='coerce').fillna(0)
        self.df['approx_cost'] = pd.to_numeric(self.df['approx_cost'], errors='coerce').fillna(0)
        # budget category
        self.df['budget_cat'] = pd.cut(self.df['approx_cost'], bins=[-1,200,500,10000], labels=["Low","Medium","High"])

    def filter_by_preferences(self, prefs):
        # prefs can be dict with lists or single values
        df = self.df.copy()

        # Accept both AI dicts and sidebar dicts
        cuisine = prefs.get('cuisine') if isinstance(prefs.get('cuisine'), list) else prefs.get('cuisine', [])
        diet = prefs.get('diet') if isinstance(prefs.get('diet'), list) else prefs.get('diet', [])
        location = prefs.get('location') if isinstance(prefs.get('location'), list) else prefs.get('location', [])
        online_order = prefs.get('online_order', "Any")
        book_table = prefs.get('book_table', "Any")
        min_rating = prefs.get('min_rating', 0)
        budget = prefs.get('budget', "Any")

        if cuisine:
            df = df[df['cuisine'].isin(cuisine)]
        if diet:
            df = df[df['diet_type'].isin(diet)]
        if location:
            df = df[df['location'].isin(location)]
        if online_order in ("Yes","No"):
            df = df[df['online_order'] == online_order]
        if book_table in ("Yes","No"):
            df = df[df['book_table'] == book_table]
        if min_rating:
            df = df[df['rate'] >= float(min_rating)]
        if budget in ("Low","Medium","High"):
            df = df[df['budget_cat'] == budget]

        return df.sort_values(by=['rate','votes'], ascending=[False, False])

    def rank_and_select(self, df, top_n=10):
        # simple scoring: normalized rate * 0.7 + normalized votes * 0.3
        df2 = df.copy()
        if df2.empty:
            return df2
        df2['rate_norm'] = (df2['rate'] - df2['rate'].min()) / (df2['rate'].max() - df2['rate'].min() + 1e-9)
        df2['votes_norm'] = (df2['votes'] - df2['votes'].min()) / (df2['votes'].max() - df2['votes'].min() + 1e-9)
        df2['score'] = df2['rate_norm']*0.7 + df2['votes_norm']*0.3
        df2 = df2.sort_values('score', ascending=False)
        if top_n:
            return df2.head(top_n)[['name','location','cuisine','diet_type','rate','votes','approx_cost','score']].reset_index(drop=True)
        return df2[['name','location','cuisine','diet_type','rate','votes','approx_cost','score']].reset_index(drop=True)

    def plot_clusters(self, df, n_clusters=3):
        # simple cluster on (rate, approx_cost, votes)
        if df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5,0.5,"No data",ha='center')
            return fig
        X = df[['rate','approx_cost','votes']].fillna(0).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        k = min(n_clusters, len(df))
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(Xs)

        fig, ax = plt.subplots(figsize=(7,5))
        scatter = ax.scatter(Xs[:,0], Xs[:,1], c=labels, cmap='tab10', s=50, alpha=0.8)
        ax.set_xlabel("Scaled rate")
        ax.set_ylabel("Scaled approx_cost")
        ax.set_title("KMeans clusters (rate vs cost)")
        legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend1)
        return fig

    # A simple stub for CART classification example (not required for filters)
    def train_cart_example(self):
        # We'll make a toy target: high_rating (>4.0)
        df = self.df.copy()
        df['high_rating'] = (df['rate'] >= 4.0).astype(int)
        features = pd.get_dummies(df[['cuisine','diet_type']], drop_first=True)
        # add numeric features
        features['votes'] = df['votes']
        features['approx_cost'] = df['approx_cost']
        y = df['high_rating']
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(features.fillna(0), y)
        return clf, features.columns
