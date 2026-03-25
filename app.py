import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Page Config
st.set_page_config(page_title="Animation Movie Recommender", layout="wide")

st.title("🎬 Animation Movie Recommendation System")
st.write("Content-Based Recommendation using Cosine Similarity")

# Load Data
@st.cache_data
def load_data():
    data_path = os.path.join("data", "data.csv")
    df = pd.read_csv(data_path)
    return df

df = load_data()

# Data Cleaning
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

df['rating'] = df['rating'].fillna(0)
df['episodes'] = df['episodes'].fillna(0)
df['genre'] = df['genre'].fillna("Unknown")

# Feature Scaling
features = df[['rating', 'episodes']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)
similarity_matrix = cosine_similarity(features_scaled)

# Recommendation Function (Safe)
def recommend(anime_name):
    try:
        idx_list = df.index[df['name'] == anime_name].tolist()
        if len(idx_list) == 0:
            return []
        idx = idx_list[0]

        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

        recs = []
        for i in scores:
            recs.append((df.iloc[i[0]]['name'], round(i[1], 3)))
        return recs
    except:
        return []

# Sidebar
st.sidebar.title("📊 Top Rated Animation Movies")
top_anime = df.sort_values(by="rating", ascending=False).head(5)
st.sidebar.dataframe(top_anime[['name', 'rating']])

# Main Layout
anime_list = df['name'].dropna().unique()
selected_anime = st.selectbox("Select Animation Movie", anime_list)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎥 Selected Animation Details")
    anime_details = df[df['name'] == selected_anime][['genre', 'rating', 'episodes']]
    st.dataframe(anime_details)

with col2:
    if st.button("🔍 Recommend Similar Movies"):
        recommendations = recommend(selected_anime)

        if recommendations:
            st.subheader("⭐ Recommended Animation Movies")
            for anime, score in recommendations:
                st.success(f"{anime}  | Similarity Score: {score}")
        else:
            st.warning("No recommendations found or error occurred.")