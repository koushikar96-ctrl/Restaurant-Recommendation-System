import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import random

# Page setup
st.set_page_config(page_title="Restaurant Explorer", layout="wide")

# Cached data loading with clustering
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\Koushika\Downloads\zomato_dataset02.csv', encoding='latin1')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.dropna(subset=['name', 'location', 'cuisine', 'rate', 'approx_cost', 'votes'], inplace=True)
    
    # Data cleaning
    df['votes'] = pd.to_numeric(df['votes'], errors='coerce').fillna(0).astype(int)
    df['votes'] = df['votes'].apply(lambda x: max(x, 0))
    
    df['numeric_rate'] = df['rate'].apply(lambda x: float(x.split('/')[0]) if isinstance(x, str) and '/' in x else float(x) if isinstance(x, float) else 0.0)
    df['numeric_rate'] = df['numeric_rate'].clip(0, 5)
    
    df['approx_cost'] = pd.to_numeric(df['approx_cost'], errors='coerce')
    df['approx_cost'] = df['approx_cost'].fillna(df['approx_cost'].median())
    
    # Classification - Budget categories
    df['budget'] = pd.cut(df['approx_cost'], bins=[0, 300, 700, np.inf], labels=['Low', 'Medium', 'High'])
    
    # Classification - Popularity categories
    df['popularity'] = pd.cut(df['votes'], bins=[0, 100, 500, np.inf], labels=['Low', 'Medium', 'High'])
    
    # Clustering - Restaurant segments
    scaler = StandardScaler()
    cluster_features = df[['numeric_rate', 'approx_cost', 'votes']].fillna(0)
    cluster_features_scaled = scaler.fit_transform(cluster_features)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(cluster_features_scaled)
    df['segment'] = df['cluster'].map({
        0: 'Budget-Friendly',
        1: 'High-End',
        2: 'Popular Mid-Range',
        3: 'Underrated Gems'
    })
    
    return df

# Page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

if 'main_option' not in st.session_state:
    st.session_state.main_option = 'Explore'

# -------------------------
# HOME PAGE
# -------------------------
def show_home():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1504674900247-0877df9cc836");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        .overlay-box {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 50px;
            border-radius: 15px;
            margin: 100px auto;
            width: 70%;
            text-align: center;
        }
        </style>
        <div class="overlay-box">
            <h1 style="font-size: 64px;">🍴 Restaurant Explorer</h1>
            <p style="font-size: 20px;">Your gateway to discovering amazing restaurants and personalized dishes.</p>
        </div>
        """, unsafe_allow_html=True
    )
    if st.button("🚀 Get Started"):
        st.session_state.page = 'main'
        st.rerun()

# -------------------------
# MAIN PAGE
# -------------------------
def show_main_page():
    st.sidebar.header("Choose an Option")
    options = ["Explore Restaurants", "Insights", "AI Dish Prediction", "AI Restaurant Prediction"]
    selected_option = st.sidebar.radio("Navigate to:", options)
    st.session_state.main_option = selected_option

    if selected_option == "Explore Restaurants":
        show_explore_page()
    elif selected_option == "Insights":
        show_insights_page()
    elif selected_option == "AI Dish Prediction":
        show_dish_prediction()
    elif selected_option == "AI Restaurant Prediction":
        show_ai_restaurant_prediction()

# -------------------------
# EXPLORE PAGE
# -------------------------
def show_explore_page():
    df = load_data()
    st.header("🔍 Explore Restaurants")
    st.sidebar.header("Filter Options")

    # Filter options
    search_name = st.sidebar.text_input("Search by restaurant name:")
    cuisine_options = df['cuisine'].unique()
    selected_cuisines = st.sidebar.multiselect("Choose cuisines:", cuisine_options, default=cuisine_options[:3])
    rating_threshold = st.sidebar.slider("Minimum rating:", 0.0, 5.0, 3.0, 0.1)
    cost_range = st.sidebar.slider("Cost range (₹):", 0, int(df['approx_cost'].max()), (0, int(df['approx_cost'].max())))
    selected_budget = st.sidebar.selectbox("Budget:", ['Any', 'Low', 'Medium', 'High'])
    popularity_filter = st.sidebar.selectbox("Popularity:", ['Any', 'Low', 'Medium', 'High'])
    diet_type = st.sidebar.selectbox("Diet Type:", ['Any', 'veg', 'nonveg', 'vegan'])
    online_order = st.sidebar.selectbox("Online Order:", ['Any', 'Yes', 'No'])
    book_table = st.sidebar.selectbox("Table Booking:", ['Any', 'Yes', 'No'])

    # Filter dataset
    filtered = df.copy()
    
    if selected_cuisines:
        filtered = filtered[filtered['cuisine'].isin(selected_cuisines)]
    filtered = filtered[filtered['numeric_rate'] >= rating_threshold]
    filtered = filtered[filtered['approx_cost'].between(cost_range[0], cost_range[1])]
    
    if selected_budget != 'Any':
        filtered = filtered[filtered['budget'] == selected_budget]
    if popularity_filter != 'Any':
        filtered = filtered[filtered['popularity'] == popularity_filter]
    if diet_type != 'Any':
        filtered = filtered[filtered['diet_type'] == diet_type]
    if online_order != 'Any':
        filtered = filtered[filtered['online_order'] == online_order]
    if book_table != 'Any':
        filtered = filtered[filtered['book_table'] == book_table]
        
    if search_name:
        filtered = filtered[filtered['name'].str.contains(search_name, case=False)]

    st.sidebar.header("Recommendation Options")
    top_10 = st.sidebar.checkbox("Show Top 10 Recommendations", value=False)
    sort_by = st.sidebar.selectbox("Sort by:", ['Rating', 'Cost', 'Popularity'])

    # Button to display recommendations
    if st.sidebar.button("✨ Get Recommendations"):
        st.subheader(f"🍽️ Restaurants Found ({len(filtered)} Results)")
        if not filtered.empty:
            if sort_by == 'Rating':
                filtered = filtered.sort_values('numeric_rate', ascending=False)
            elif sort_by == 'Cost':
                filtered = filtered.sort_values('approx_cost')
            elif sort_by == 'Popularity':
                filtered = filtered.sort_values('votes', ascending=False)
                
            if top_10:
                filtered = filtered.head(10)

            for _, row in filtered.iterrows():
                with st.expander(f"{row['name']} - ⭐ {row['numeric_rate']:.1f}"):
                    st.markdown(f"""
                    **📍 Location:** {row['location']}  
                    **🍱 Cuisine:** {row['cuisine']}  
                    **🥗 Diet Type:** {row['diet_type']}  
                    **⭐ Rating:** {row['rate']}  
                    **💰 Cost:** ₹{row['approx_cost']}  
                    **🗳️ Votes:** {row['votes']}  
                    **💵 Budget:** {row['budget']}  
                    **🔥 Popularity:** {row['popularity']}  
                    **📱 Online Order:** {'Yes' if row['online_order'] == 'Yes' else 'No'}  
                    **🪑 Table Booking:** {'Yes' if row['book_table'] == 'Yes' else 'No'}
                    **🔍 Segment:** {row['segment']}
                    """)
        else:
            st.warning("No restaurants match your criteria.")

# -------------------------
# INSIGHTS PAGE
# -------------------------
def show_insights_page():
    df = load_data()
    st.header("📊 Insights & Analytics Dashboard")
    
    # Overview Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Restaurants", len(df))
    col2.metric("Average Rating", f"{df['numeric_rate'].mean():.1f}/5")
    col3.metric("Average Cost", f"₹{df['approx_cost'].mean():.0f}")
    col4.metric("Total Votes", f"{df['votes'].sum():,}")
    
    # Tabbed interface for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📉 Distributions", "🍽️ Cuisines", "💰 Budget Analysis", "🤖 ML Insights"])
    
    with tab1:
        st.subheader("Data Insights")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rating Distribution
        sns.histplot(df['numeric_rate'], bins=20, kde=False, ax=axes[0], color='skyblue')
        axes[0].set_title("Rating Distribution")
        axes[0].set_xlabel("Rating (0-5)")
        
        # Cost Distribution
        sns.histplot(df['approx_cost'], bins=20, kde=False, ax=axes[1], color='salmon')
        axes[1].set_title("Cost Distribution")
        axes[1].set_xlabel("Cost (₹)")
        
        st.pyplot(fig)
        
        # Interactive distribution plot
        dist_col = st.selectbox("Select distribution to view:", ['votes', 'approx_cost', 'numeric_rate'])
        plt.figure(figsize=(10, 4))
        sns.histplot(df[dist_col], bins=20, kde=False, color='purple')
        plt.title(f"{dist_col.title()} Distribution")
        st.pyplot(plt)
    
    with tab2:
        st.subheader("Cuisine Analysis")
        
        # Top Cuisines
        top_n = st.slider("Show top:", 5, 20, 10)
        top_cuisines = df['cuisine'].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        top_cuisines.plot(kind='barh', color='teal', ax=ax)
        ax.set_title(f"Top {top_n} Cuisines")
        ax.set_xlabel("Number of Restaurants")
        st.pyplot(fig)
        
        # Cuisine vs Rating
        st.write("### Cuisine vs Average Rating")
        cuisine_stats = df.groupby('cuisine').agg({'numeric_rate': 'mean', 'approx_cost': 'mean'})
        cuisine_stats = cuisine_stats.sort_values('numeric_rate', ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='cuisine', y='numeric_rate', data=cuisine_stats.reset_index(), 
                   hue='cuisine', palette='viridis', legend=False)
        plt.xticks(rotation=45)
        plt.title("Top Cuisines by Average Rating")
        plt.ylabel("Average Rating")
        st.pyplot(plt)
    
    with tab3:
        st.subheader("Budget Analysis")
        
        # Budget Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        df['budget'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=axes[0], 
                                      colors=['lightgreen', 'gold', 'salmon'])
        axes[0].set_title("Budget Level Distribution")
        
        # Budget vs Rating
        sns.boxplot(x='budget', y='numeric_rate', data=df, ax=axes[1], 
                   hue='budget', palette='pastel', legend=False)
        axes[1].set_title("Rating Distribution by Budget Level")
        st.pyplot(fig)
        
        # Interactive budget analysis
        budget_metric = st.selectbox("Analyze by:", ['votes', 'numeric_rate', 'approx_cost'])
        plt.figure(figsize=(10, 5))
        sns.barplot(x='budget', y=budget_metric, data=df, estimator=np.mean, 
                   errorbar=None, hue='budget', palette='coolwarm', legend=False)
        plt.title(f"Average {budget_metric.title()} by Budget Level")
        st.pyplot(plt)
    
    with tab4:
        st.subheader("Machine Learning Insights")
        
        # Clustering Results
        st.write("### Restaurant Segments")
        cluster_stats = df.groupby('segment').agg({
            'numeric_rate': 'mean',
            'approx_cost': 'mean',
            'votes': 'mean',
            'name': 'count'
        }).rename(columns={'name': 'count'})
        
        st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        
        # Visualize clusters
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='approx_cost', y='numeric_rate', hue='segment', 
                        size='votes', sizes=(20, 200), data=df, palette='viridis')
        plt.title("Restaurant Clusters by Cost, Rating & Popularity")
        plt.xlabel("Approximate Cost (₹)")
        plt.ylabel("Rating")
        st.pyplot(plt)
        
        # Classification Demo
        st.write("### Popularity Prediction (Decision Tree)")
        st.write("Predicting if a restaurant will be highly popular (votes > 500)")
        
        # Prepare data
        X = df[['numeric_rate', 'approx_cost']]
        y = (df['votes'] > 500).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X_train, y_train)
        
        # Show results
        st.write("#### Feature Importance:")
        feat_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': clf.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(feat_importance.set_index('Feature'))
        
        st.write("#### Sample Predictions:")
        sample = X_test.sample(5, random_state=42)
        predictions = clf.predict(sample)
        sample['Predicted_Popular'] = ['Yes' if p else 'No' for p in predictions]
        sample['Actual_Popular'] = ['Yes' if a else 'No' for a in y_test.loc[sample.index]]
        st.dataframe(sample)

# -------------------------
# AI DISH PREDICTION PAGE
# -------------------------
def show_dish_prediction():
    df = load_data()
    st.header("🤖 AI Dish Prediction")
    st.write("Let me recommend a dish tailored to your preferences!")

    # Interactive Questions
    col1, col2 = st.columns(2)
    
    with col1:
        preferred_cuisine = st.selectbox("Which cuisine do you prefer?", 
                                       sorted(df['cuisine'].unique()))
        dietary_restrictions = st.multiselect("Any dietary restrictions?", 
                                            ["Vegetarian", "Vegan", "Non-vegetarian", "Gluten-free", "Nut-free"])
        meal_type = st.selectbox("Meal type:", 
                               ["Any", "Breakfast", "Lunch", "Dinner", "Snack", "Dessert"])
        
    with col2:
        spice_level = st.select_slider("Preferred spice level:", 
                                     ["Mild", "Medium", "Hot", "Very Hot"])
        favorite_ingredients = st.text_input("Any specific ingredients you love?")
        disliked_ingredients = st.text_input("Any ingredients you dislike?")

    budget_range = st.slider("What's your budget for the dish? (₹)", 
                           100, 3000, (200, 1000))
    health_conscious = st.checkbox("Looking for healthier options?")
    cooking_time = st.radio("Preferred cooking time:", 
                          ["Quick (under 30 mins)", "Moderate (30-60 mins)", "Slow-cooked (over 60 mins)"])

    # Enhanced dish recommendations with dietary tags
    dish_recommendations = {
        "Thai": [
            {"name": "Pad Thai", "vegetarian": False, "vegan": False, "spice": "Medium"},
            {"name": "Green Curry", "vegetarian": True, "vegan": False, "spice": "Hot"},
            {"name": "Tom Yum Soup", "vegetarian": False, "vegan": False, "spice": "Hot"},
            {"name": "Som Tum (Papaya Salad)", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Massaman Curry", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Thai Basil Fried Rice", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Vegetable Spring Rolls", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Mango Sticky Rice", "vegetarian": True, "vegan": True, "spice": "Mild"}
        ],
        "Italian": [
            {"name": "Margherita Pizza", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Pasta Carbonara", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Risotto", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Lasagna", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Tiramisu", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Caprese Salad", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Minestrone Soup", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Bruschetta", "vegetarian": True, "vegan": True, "spice": "Mild"}
        ],
        "Chinese": [
            {"name": "Dim Sum", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Kung Pao Chicken", "vegetarian": False, "vegan": False, "spice": "Hot"},
            {"name": "Peking Duck", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Hot Pot", "vegetarian": False, "vegan": False, "spice": "Medium"},
            {"name": "Mapo Tofu", "vegetarian": True, "vegan": True, "spice": "Hot"},
            {"name": "Vegetable Fried Rice", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Spring Rolls", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Wonton Soup", "vegetarian": False, "vegan": False, "spice": "Mild"}
        ],
        "Mexican": [
            {"name": "Tacos al Pastor", "vegetarian": False, "vegan": False, "spice": "Medium"},
            {"name": "Vegetable Quesadilla", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Chiles en Nogada", "vegetarian": True, "vegan": False, "spice": "Medium"},
            {"name": "Pozole", "vegetarian": False, "vegan": False, "spice": "Medium"},
            {"name": "Tamales", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Guacamole", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Bean Burrito", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Churros", "vegetarian": True, "vegan": False, "spice": "Mild"}
        ],
        "French": [
            {"name": "Coq au Vin", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Ratatouille", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Croque Monsieur", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Tarte Tatin", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Quiche Lorraine", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "French Onion Soup", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Salade Niçoise", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Crème Brûlée", "vegetarian": True, "vegan": False, "spice": "Mild"}
        ],
        "Indian": [
            {"name": "Paneer Tikka", "vegetarian": True, "vegan": False, "spice": "Medium"},
            {"name": "Dal Tadka", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Masala Dosa", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Palak Paneer", "vegetarian": True, "vegan": False, "spice": "Mild"},
            {"name": "Chana Masala", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Vegetable Biryani", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Idli Sambar", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Gulab Jamun", "vegetarian": True, "vegan": False, "spice": "Mild"}
        ]
    }

    # Generate Recommendation
    if st.button("🍽️ Get My Dish Recommendation"):
        # Get available dishes for the selected cuisine
        available_dishes = dish_recommendations.get(preferred_cuisine, [
            {"name": "Special Fried Rice", "vegetarian": True, "vegan": True, "spice": "Mild"},
            {"name": "Grilled Salmon", "vegetarian": False, "vegan": False, "spice": "Mild"},
            {"name": "Vegetable Stir Fry", "vegetarian": True, "vegan": True, "spice": "Medium"},
            {"name": "Chocolate Lava Cake", "vegetarian": True, "vegan": False, "spice": "Mild"}
        ])
        
        # Filter dishes based on dietary restrictions
        filtered_dishes = []
        for dish in available_dishes:
            include = True
            
            # Check dietary restrictions
            if "Vegetarian" in dietary_restrictions and not dish["vegetarian"]:
                include = False
            if "Vegan" in dietary_restrictions and not dish["vegan"]:
                include = False
            if "Non-vegetarian" in dietary_restrictions and dish["vegetarian"]:
                include = False
                
            # Check spice level
            spice_mapping = {"Mild": 0, "Medium": 1, "Hot": 2, "Very Hot": 3}
            if spice_mapping[dish["spice"]] < spice_mapping[spice_level]:
                include = False
                
            if include:
                filtered_dishes.append(dish)
        
        # If no dishes match strict filters, relax some criteria
        if not filtered_dishes:
            for dish in available_dishes:
                include = True
                # Only enforce vegetarian/vegan restrictions strictly
                if "Vegetarian" in dietary_restrictions and not dish["vegetarian"]:
                    include = False
                if "Vegan" in dietary_restrictions and not dish["vegan"]:
                    include = False
                if "Non-vegetarian" in dietary_restrictions and dish["vegetarian"]:
                    include = False
                    
                if include:
                    filtered_dishes.append(dish)
        
        # If still no dishes, just show all available
        if not filtered_dishes:
            filtered_dishes = available_dishes
        
        # Select a random dish from filtered options
        recommended_dish = random.choice(filtered_dishes)
        
        st.success(f"## Based on your preferences, we recommend: **{recommended_dish['name']}**!")
        
        with st.expander("See details about this recommendation"):
            st.write(f"**Cuisine:** {preferred_cuisine}")
            st.write(f"**Spice Level:** {recommended_dish['spice']}")
            st.write(f"**Budget Range:** ₹{budget_range[0]} - ₹{budget_range[1]}")
            st.write(f"**Meal Type:** {meal_type}")
            st.write(f"**Cooking Time:** {cooking_time}")
            
            # Display dietary information
            diet_info = []
            if recommended_dish['vegetarian']:
                diet_info.append("Vegetarian")
            if recommended_dish['vegan']:
                diet_info.append("Vegan")
            if not recommended_dish['vegetarian']:
                diet_info.append("Non-vegetarian")
            
            st.write(f"**Dietary Type:** {', '.join(diet_info)}")
            
            if dietary_restrictions:
                st.write(f"**Dietary Considerations:** {', '.join(dietary_restrictions)}")
            if favorite_ingredients:
                st.write(f"**Includes your favorite ingredients:** {favorite_ingredients}")
            if disliked_ingredients:
                st.write(f"**Excludes ingredients you dislike:** {disliked_ingredients}")
            if health_conscious:
                st.write("**Healthier option selected** - This recommendation focuses on balanced nutrition")
        
        # Show restaurants serving similar dishes
        st.subheader("🍴 Recommended Restaurants")
        
        # Create a scoring system for restaurants
        def score_restaurant(row):
            score = 0
            # Base score from rating and popularity
            score += row['numeric_rate'] * 20  # Rating is out of 5
            score += np.log(row['votes'] + 1) * 5  # Logarithmic scale for votes
            
            # Matching cuisine
            if row['cuisine'] == preferred_cuisine:
                score += 50
            
            # Budget match
            if budget_range[0] <= row['approx_cost'] <= budget_range[1]:
                score += 30
            else:
                # Penalize for being outside budget but not too much
                score -= 10
            
            # Dietary restrictions
            if "Vegetarian" in dietary_restrictions and row['diet_type'] == 'veg':
                score += 20
            if "Vegan" in dietary_restrictions and row['diet_type'] == 'vegan':
                score += 20
            if "Non-vegetarian" in dietary_restrictions and row['diet_type'] == 'nonveg':
                score += 20
            
            # Add some randomness to vary results
            score += random.uniform(0, 20)
            
            return score
        
        # Score all restaurants
        df['score'] = df.apply(score_restaurant, axis=1)
        
        # Sort by score and get top 3 unique restaurants
        sorted_restaurants = df.sort_values('score', ascending=False)
        recommendations = sorted_restaurants.head(3).to_dict('records')
        
        if recommendations:
            st.write("Here are some restaurants you might like:")
            for row in recommendations:
                with st.expander(f"{row['name']} - ⭐ {row['numeric_rate']:.1f}"):
                    st.write(f"**Location:** {row['location']}")
                    st.write(f"**Cuisine:** {row['cuisine']}")
                    st.write(f"**Diet Type:** {row['diet_type']}")
                    st.write(f"**Approx Cost:** ₹{row['approx_cost']}")
                    st.write(f"**Votes:** {row['votes']}")
                    st.write(f"**Online Order:** {'Yes' if row['online_order'] == 'Yes' else 'No'}")
                    st.write(f"**Table Booking:** {'Yes' if row['book_table'] == 'Yes' else 'No'}")
                    st.write(f"**Why we recommend this:** Matches {preferred_cuisine} cuisine" 
                            f"{' and fits your budget' if budget_range[0] <= row['approx_cost'] <= budget_range[1] else ''}")
        else:
            st.warning("We couldn't find perfect matches, but here are some top-rated options:")
            top_restaurants = df.sort_values(['numeric_rate', 'votes'], ascending=[False, False]).head(3)
            for _, row in top_restaurants.iterrows():
                with st.expander(f"{row['name']} - ⭐ {row['numeric_rate']:.1f}"):
                    st.write(f"**Location:** {row['location']}")
                    st.write(f"**Cuisine:** {row['cuisine']}")
                    st.write(f"**Diet Type:** {row['diet_type']}")
                    st.write(f"**Approx Cost:** ₹{row['approx_cost']}")
                    st.write(f"**Votes:** {row['votes']}")
                    st.write(f"**Online Order:** {'Yes' if row['online_order'] == 'Yes' else 'No'}")
                    st.write(f"**Table Booking:** {'Yes' if row['book_table'] == 'Yes' else 'No'}")
# -------------------------
# AI RESTAURANT PREDICTION PAGE
# -------------------------
def show_ai_restaurant_prediction():
    df = load_data()
    st.header("🏢 AI Restaurant Prediction")
    st.write("Tell us more about your preferences and we'll recommend the perfect restaurant for you!")
    
    with st.form("restaurant_preferences"):
        st.subheader("Your Preferences")
        
        col1, col2 = st.columns(2)
        
        with col1:
            preferred_cuisine = st.selectbox("Preferred Cuisine", sorted(df['cuisine'].unique()))
            budget = st.selectbox("Budget Level", ['Any', 'Low', 'Medium', 'High'])
            diet_type = st.selectbox("Diet Type", ['Any', 'veg', 'nonveg', 'vegan'])
            online_order = st.radio("Online Ordering Needed?", ['Any', 'Yes', 'No'])
            
        with col2:
            min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
            book_table = st.radio("Table Booking Needed?", ['Any', 'Yes', 'No'])
            popularity = st.selectbox("Popularity Level", ['Any', 'Low', 'Medium', 'High'])
        
        submit_button = st.form_submit_button("Find My Restaurant")
    
    if submit_button:
        # Initialize filter tracking
        filter_strictness = 1  # 1=strict, 2=medium, 3=loose
        
        while filter_strictness <= 3:
            # Filter restaurants based on preferences with progressive relaxation
            filtered = df.copy()
            
            # Always apply minimum rating filter
            filtered = filtered[filtered['numeric_rate'] >= min_rating]
            
            # Apply strict filters first
            if filter_strictness == 1:
                if preferred_cuisine:
                    filtered = filtered[filtered['cuisine'] == preferred_cuisine]
                if budget != 'Any':
                    filtered = filtered[filtered['budget'] == budget]
                if diet_type != 'Any':
                    filtered = filtered[filtered['diet_type'] == diet_type]
                if online_order != 'Any':
                    filtered = filtered[filtered['online_order'] == online_order]
                if book_table != 'Any':
                    filtered = filtered[filtered['book_table'] == book_table]
                if popularity != 'Any':
                    filtered = filtered[filtered['popularity'] == popularity]
            
            # Medium strictness - relax some filters
            elif filter_strictness == 2:
                if preferred_cuisine:
                    filtered = filtered[filtered['cuisine'] == preferred_cuisine]
                if budget != 'Any':
                    filtered = filtered[filtered['budget'] == budget]
                # Relax diet_type, online_order, book_table, popularity
            
            # Loose filters - only essential preferences
            elif filter_strictness == 3:
                if preferred_cuisine:
                    filtered = filtered[filtered['cuisine'] == preferred_cuisine]
                # Relax all other filters except rating
            
            # Check if we have results
            if not filtered.empty:
                break
                
            filter_strictness += 1
        
        # If we still have no results after all relaxation levels, show top rated
        if filtered.empty:
            filtered = df[df['numeric_rate'] >= min_rating]
            if filtered.empty:
                filtered = df.copy()
            message = "Showing top-rated restaurants that match some of your preferences:"
        else:
            if filter_strictness == 1:
                message = "Perfect matches for your preferences:"
            elif filter_strictness == 2:
                message = "Close matches for your preferences (relaxed some filters):"
            else:
                message = "Good options that match your main preferences:"
        
        # Sort by rating and popularity
        filtered = filtered.sort_values(['numeric_rate', 'votes'], ascending=[False, False])
        
        # Get unique recommendations (not shown before)
        if 'shown_restaurants' not in st.session_state:
            st.session_state.shown_restaurants = []
            
        recommendations = []
        for _, row in filtered.iterrows():
            if row['name'] not in st.session_state.shown_restaurants:
                recommendations.append(row)
                st.session_state.shown_restaurants.append(row['name'])
                if len(recommendations) >= 3:
                    break
        
        # If we didn't get enough unique ones, use the top results
        if len(recommendations) < 3:
            recommendations = filtered.head(3).to_dict('records')
        
        st.success(message)
        
        for idx, row in enumerate(recommendations):
            with st.expander(f"{row['name']} - ⭐ {row['numeric_rate']:.1f} ({'Perfect' if filter_strictness == 1 else 'Good' if filter_strictness == 2 else 'Alternative'} match)"):
                st.write(f"**Location:** {row['location']}")
                st.write(f"**Cuisine:** {row['cuisine']}")
                st.write(f"**Diet Type:** {row['diet_type']}")
                st.write(f"**Approx Cost:** ₹{row['approx_cost']} ({row['budget']})")
                st.write(f"**Rating:** {row['numeric_rate']:.1f}/5")
                st.write(f"**Votes:** {row['votes']} ({row['popularity']} popularity)")
                st.write(f"**Online Order:** {'✅ Yes' if row['online_order'] == 'Yes' else '❌ No'}")
                st.write(f"**Table Booking:** {'✅ Yes' if row['book_table'] == 'Yes' else '❌ No'}")
                st.write(f"**Segment:** {row['segment']}")
                
                # Explain why this was recommended
                reasons = []
                if preferred_cuisine and row['cuisine'] == preferred_cuisine:
                    reasons.append("matches your preferred cuisine")
                if budget != 'Any' and row['budget'] == budget:
                    reasons.append("fits your budget level")
                if diet_type != 'Any' and row['diet_type'] == diet_type:
                    reasons.append("meets your dietary needs")
                if online_order != 'Any' and row['online_order'] == online_order:
                    reasons.append("matches your online order preference")
                if book_table != 'Any' and row['book_table'] == book_table:
                    reasons.append("matches your table booking preference")
                if popularity != 'Any' and row['popularity'] == popularity:
                    reasons.append("matches your popularity preference")
                
                if reasons:
                    st.write(f"**Why recommended:** {' and '.join(reasons)}")
                else:
                    st.write("**Why recommended:** Highly rated option you might enjoy")
            
# -------------------------
# ROUTING LOGIC
# -------------------------
if st.session_state.page == 'home':
    show_home()
elif st.session_state.page == 'main':
    show_main_page()