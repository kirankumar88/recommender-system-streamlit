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
st.markdown("Content-Based Recommendation using Cosine Similarity")

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

    df = df.reset_index(drop=True)
    return df

df = load_data()

# ---------------------------
# Metrics
# ---------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Movies", len(df))
col2.metric("Average Rating", round(df['rating'].mean(), 2))
col3.metric("Max Episodes", int(df['episodes'].max()))

st.markdown("---")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("📊 App Information")
st.sidebar.write("This app recommends animation movies using a content-based recommendation system.")
st.sidebar.write("Algorithm: Cosine Similarity")
st.sidebar.write("Features Used:")
st.sidebar.write("- Rating")
st.sidebar.write("- Episodes")

st.sidebar.markdown("---")

st.sidebar.subheader("🎯 Filter Movies")

min_rating = st.sidebar.slider("Minimum Rating", 0.0, 10.0, 5.0)
filtered_df = df[df['rating'] >= min_rating]

st.sidebar.markdown("---")

st.sidebar.subheader("⭐ Top Rated Movies")
top_movies = df.sort_values(by="rating", ascending=False).head(5)
st.sidebar.dataframe(top_movies[['name', 'rating']])

# ---------------------------
# Feature Scaling
# ---------------------------
features = df[['rating', 'episodes']]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# ---------------------------
# Movie Selection
# ---------------------------
selected_index = st.selectbox(
    "🎥 Select Animation Movie",
    filtered_df.index,
    format_func=lambda x: df.loc[x, 'name']
)

col1, col2 = st.columns(2)

# Movie Details
with col1:
    st.subheader("🎬 Selected Movie Details")
    st.dataframe(df.loc[[selected_index], ['genre', 'rating', 'episodes']])

# Recommendations
with col2:
    st.subheader("⭐ Recommended Movies")

    if st.button("🔍 Recommend Similar Movies"):

        selected_vector = features_scaled[selected_index].reshape(1, -1)
        similarity_scores = cosine_similarity(selected_vector, features_scaled)
        similarity_scores = similarity_scores.flatten()

        similar_indices = similarity_scores.argsort()[::-1][1:6]

        for i in similar_indices:
            score = similarity_scores[i]
            st.progress(float(score))
            st.write(f"{df.loc[i, 'name']}  (Score: {round(score,3)})")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("Machine Learning Recommendation System | Streamlit App")