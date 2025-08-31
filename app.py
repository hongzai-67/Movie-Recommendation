import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# =====================================================================
# 推荐系统核心类（从 again.txt 复制过来，保留主要功能）
# =====================================================================
class IMDBContentBasedRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.qualified_movies = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        self.average_rating = None
        self.vote_threshold = None

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    def load_imdb_data(self, file_path="imdb_movies.csv"):
        self.movies_df = pd.read_csv(file_path, low_memory=False)
        # 简化处理
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )
        # 加权评分（简单版）
        self.movies_df['vote_count'] = np.random.randint(50, 500, len(self.movies_df))
        self.average_rating = self.movies_df['score'].mean()
        self.movies_df['weighted_rating'] = (
            0.7 * self.movies_df['score'] + 0.3 * self.average_rating
        )
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.qualified_movies['enhanced_content'] = self.qualified_movies['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.qualified_movies['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.qualified_movies.index, index=self.qualified_movies['orig_title']).drop_duplicates()

    def get_content_recommendations(self, title, n=10):
        if title not in self.indices:
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None
            return "matches", possible_matches.head(5)

        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx[0]

        movie_info = self.qualified_movies.loc[idx]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        similarity_values = [i[1] for i in sim_scores]

        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values
        return movie_info, movies

    def get_similarity_level(self, score):
        if score >= 0.87:
            return "🔥 VERY HIGH"
        elif score >= 0.86:
            return "🟢 HIGH"
        elif score >= 0.85:
            return "🟡 MODERATE"
        else:
            return "🔴 LOW"

# =====================================================================
# Streamlit 应用
# =====================================================================
@st.cache_resource
def load_system():
    system = IMDBContentBasedRecommendationSystem()
    system.load_imdb_data("imdb_movies.csv")
    system.build_content_based_system()
    return system

system = load_system()

st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
st.markdown("=" * 65)
st.subheader("✨ NEW FEATURES: Search by Title, Genre, Crew, or Advanced Multi-Search!")

menu = [
    "1️⃣ Search by Movie Title (Content-based recommendations)",
    "2️⃣ Search by Genre (Top-rated movies in genre)",
    "3️⃣ Search by Crew Member (Movies with specific actor/director)",
    "4️⃣ Advanced Search (Combine multiple criteria)"
]
choice = st.selectbox("🎯 SEARCH OPTIONS:", menu)

if "1️⃣" in choice:
    title = st.text_input("🎬 Enter a movie title:")
    n = st.number_input("📊 Number of recommendations", 1, 20, 10)
    if st.button("Search"):
        cleaned_title = system.clean_title_text(title)
        movie_info, result = system.get_content_recommendations(cleaned_title, n=n)
        if movie_info == "matches":
            st.write("🔍 Did you mean one of these?")
            for i, (_, row) in enumerate(result.iterrows()):
                st.write(f"{i+1}. {row['names']}")
        elif movie_info is not None:
            st.write(f"🎯 Found: {movie_info['names']}")
            st.write(f"📅 Year: {movie_info['date_x']}")
            st.write(f"🎭 Genre: {movie_info['genre']}")
            st.write(f"⭐ Score: {movie_info['score']} → Weighted: {movie_info['weighted_rating']:.2f}")
            st.write(f"📝 Overview: {movie_info['overview'][:100]}...")

            st.markdown(f"🔥 TOP {n} RECOMMENDATIONS (SORTED BY HIGHEST SIMILARITY):")
            for i, (_, rec) in enumerate(result.iterrows()):
                similarity_percent = rec['similarity'] * 100
                similarity_level = system.get_similarity_level(rec['similarity'])
                if i == 0:
                    st.markdown(f"🏆 {i+1}. {rec['names']} ⭐ TOP MATCH!")
                else:
                    st.markdown(f"{i+1}. {rec['names']}")
                st.write(f"🎯 Similarity: {rec['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")
                st.write(f"⭐ Rating: {rec['weighted_rating']:.2f}")
                st.write(f"🎭 Genre: {rec['genre']}")
                st.write("")
        else:
            st.error("❌ Movie not found!")

