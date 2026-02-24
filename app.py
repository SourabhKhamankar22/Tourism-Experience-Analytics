import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Streamlit UI Setup (MUST BE FIRST) ---
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.title("🌍 Tourism Experience Analytics & Recommendation System")
st.markdown("Predict user behavior and get personalized attraction recommendations.")

# --- 2. Load the Models and Data ---
@st.cache_resource
def load_assets():
    models = joblib.load('tourism_models_and_preprocessors.pkl')
    # Make sure this file name/path matches exactly where your CSV is saved
    df = pd.read_csv('Cleaned_Tourism_Dataset.csv') 
    return models, df

models, df = load_assets()

# Extract individual components
lgb_model = models['classification_model']
svd_model = models['recommendation_model']
scaler = models['scaler']
le_dict = models['label_encoders']

# --- 3. Sidebar: User Input ---
st.sidebar.header("User Profile Input")

# Create dropdowns based on the classes the LabelEncoders learned
input_continent = st.sidebar.selectbox("Continent", le_dict['Continent'].classes_)
input_country = st.sidebar.selectbox("Your Home Country", le_dict['Country'].classes_)
input_attraction_type = st.sidebar.selectbox("Preferred Attraction Type", le_dict['AttractionType'].classes_)
input_year = st.sidebar.number_input("Visit Year", min_value=2010, max_value=2030, value=2024)
input_month = st.sidebar.slider("Visit Month", 1, 12, 6)
input_avg_rating = st.sidebar.slider("Your Historical Average Rating", 1.0, 5.0, 4.0)
input_user_id = st.sidebar.selectbox("Select User ID (for Collaborative Filtering)", df['UserId'].unique()[:50])

# --- 4. Prediction Logic ---
if st.sidebar.button("Predict & Recommend"):
    
    # Preprocess the inputs
    try:
        cont_enc = le_dict['Continent'].transform([input_continent])[0]
        count_enc = le_dict['Country'].transform([input_country])[0]
        att_enc = le_dict['AttractionType'].transform([input_attraction_type])[0]
    except ValueError:
        st.error("Error encoding inputs. Please ensure the selected inputs existed in the training data.")
        st.stop()
        
    # Feature array for LightGBM
    user_features = np.array([[input_year, input_month, input_avg_rating, cont_enc, count_enc, att_enc]])
    user_features_scaled = scaler.transform(user_features)
    
    # Predict Visit Mode
    pred_mode_encoded = lgb_model.predict(user_features_scaled)[0]
    pred_mode_text = le_dict['VisitMode'].inverse_transform([pred_mode_encoded])[0]
    
# --- HYBRID RECOMMENDATION LOGIC ---
    # 1. Content-Based Filter: Narrow down attractions strictly by the preferred Attraction Type
    filtered_df = df[df['AttractionType'] == input_attraction_type]
    
    if filtered_df.empty:
        # Fallback if no attractions of that type exist at all
        filtered_df = df
        st.warning(f"No {input_attraction_type} found in the database. Showing general recommendations.")
        
    # 2. Collaborative Filtering: Predict ratings for the filtered list
    candidate_attractions = filtered_df['AttractionId'].unique()
    visited_by_user = df[df['UserId'] == input_user_id]['AttractionId'].unique()
    unvisited_candidates = [att for att in candidate_attractions if att not in visited_by_user]
    
    if not unvisited_candidates:
        st.warning(f"You've already visited all highly-rated {input_attraction_type}s! Try a different category.")
        top_5_names = []
    else:
        # SVD model predicts the rating the user would give to the unvisited candidates
        preds = [svd_model.predict(input_user_id, att_id) for att_id in unvisited_candidates]
        preds.sort(key=lambda x: x.est, reverse=True)
        top_5_ids = [p.iid for p in preds[:5]]
        
        # Map IDs back to names
        top_5_names = []
        for att_id in top_5_ids:
            name = df[df['AttractionId'] == att_id]['Attraction'].iloc[0]
            top_5_names.append(name)

    # --- 5. Display Results ---
    st.header("🎯 Your Personalized Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"### Predicted Visit Mode:\n# ✈️ {pred_mode_text}")
        st.write("Based on demographics and historical rating patterns, the classification model predicts this travel mode.")
        
    with col2:
        # Updated header to reflect the content-based filter
        st.info(f"### Top '{input_attraction_type}' Recommendations:")
        if top_5_names:
            for i, att in enumerate(top_5_names, 1):
                st.write(f"**{i}. {att}**")

# --- 6. Analytics Visualizations ---
st.markdown("---")
st.header("📊 Global Tourism Analytics Dashboard")
st.write("Insights into popular attractions, top regions, and user segments.")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 10 Most Visited Attraction Types")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    pop_attractions = df['AttractionType'].value_counts().head(10)
    # UPDATED: Added hue and legend=False
    sns.barplot(x=pop_attractions.values, y=pop_attractions.index, hue=pop_attractions.index, palette='viridis', legend=False, ax=ax1)
    ax1.set_xlabel("Number of Visits")
    st.pyplot(fig1)

with col4:
    st.subheader("User Distribution by Travel Mode")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    # UPDATED: Added hue and legend=False
    sns.countplot(data=df, y='VisitMode', hue='VisitMode', order=df['VisitMode'].value_counts().index, palette='magma', legend=False, ax=ax2)
    ax2.set_xlabel("Count")
    st.pyplot(fig2)

# --- Additional Chart: Top Regions (Required by Rubric) ---
st.markdown("---")
st.subheader("🌍 Top 10 User Origins (Countries)")
fig3, ax3 = plt.subplots(figsize=(10, 4))
top_countries = df['Country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index, palette='crest', legend=False, ax=ax3)
ax3.set_xlabel("Number of Visitors")
ax3.set_ylabel("Country")
st.pyplot(fig3)