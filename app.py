import streamlit as st
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Animation Movie Recommender", layout="wide")

st.title("🎬 Animation Movie Recommendation System")
st.write("Content-Based Recommendation using Cosine Similarity")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    data_path = os.path.join("data", "data.csv")
    df = pd.read_csv(data_path)

    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

    df['rating'] = df['rating'].fillna(0)
    df['episodes'] = df['episodes'].fillna(0)
    df['genre'] = df['genre'].fillna("Unknown")

    return df

df = load_data()

# ---------------------------
# Build Similarity Matrix
# ---------------------------
@st.cache_data
def build_similarity(df):
    features = df[['rating', 'episodes']]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    similarity_matrix = cosine_similarity(features_scaled)
    return similarity_matrix

similarity_matrix = build_similarity(df)

# ---------------------------
# Recommendation Function
# ---------------------------
def recommend(anime_name):
    try:
        idx = df.index[df['name'] == anime_name].tolist()[0]

        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]

        anime_indices = [i[0] for i in sim_scores]

        return df['name'].iloc[anime_indices]
    except:
        return []

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📊 Top Rated Animation Movies")
top_anime = df.sort_values(by="rating", ascending=False).head(5)
st.sidebar.dataframe(top_anime[['name', 'rating']])

# ---------------------------
# Main UI
# ---------------------------
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

        if len(recommendations) > 0:
            st.subheader("⭐ Recommended Animation Movies")
            for anime in recommendations:
                st.success(anime)
        else:
            st.warning("No recommendations found.")