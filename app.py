import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ====================================================
# Recommendation System Class
# ====================================================
class IMDBContentBasedRecommendationSystem:
    def __init__(self):
        self.movies_df = None
        self.qualified_movies = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    def load_imdb_data(self, file_path):
        self.movies_df = pd.read_csv(file_path, low_memory=False)
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
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    def get_similarity_level(self, score):
        if score >= 0.96:
            return "🔥 VERY HIGH"
        elif score >= 0.92:
            return "🟢 HIGH"
        elif score >= 0.88:
            return "🟡 MODERATE"
        elif score >= 0.84:
            return "🟠 LOW"
        else:
            return "🔴 VERY LOW"

    def get_content_recommendations(self, title, n=10):
        # 模糊匹配
        if title not in self.indices:
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(5), None

        # 精确匹配
        idx = self.indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = [i[1] for i in sim_scores]
        return "ok", self.qualified_movies.loc[idx], movies


# ====================================================
# Streamlit Terminal-style UI
# ====================================================
def main():
    st.set_page_config(page_title="IMDB Recommender", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
    st.markdown("=================================================================")
    st.markdown("✨ NEW FEATURES: Search by Title, Genre, Crew, or Advanced Multi-Search!")

    uploaded_file = st.sidebar.file_uploader("Upload IMDB dataset (CSV)", type="csv")
    if not uploaded_file:
        st.warning("Please upload imdb_movies.csv in the sidebar.")
        return

    # 初始化推荐器
    recommender = IMDBContentBasedRecommendationSystem()
    recommender.load_imdb_data(uploaded_file)
    recommender.build_content_based_system()

    # 菜单选择
    option = st.radio("🎯 SEARCH OPTIONS:", [
        "1️⃣ Search by Movie Title",
        "2️⃣ Search by Genre",
        "3️⃣ Search by Crew",
        "4️⃣ Advanced Search",
        "5️⃣ Browse Genres",
        "6️⃣ Browse Crew"
    ])

    output = []

    # ----------------- Search by Title -----------------
    if option.startswith("1️⃣"):
        title = st.text_input("🎬 Enter a movie title:")
        n_recs = st.slider("📊 Number of recommendations", 1, 20, 10)
        if st.button("Get Recommendations"):
            cleaned_title = recommender.clean_title_text(title)
            status, movie_info, recs = recommender.get_content_recommendations(cleaned_title, n=n_recs)

            if status is None:
                st.error(f"❌ No matches found for **{title}**")

            elif status == "choose":
                st.markdown("🔍 Did you mean one of these?")
                choices = movie_info['names'].tolist()
                choice = st.selectbox("🎯 Select a movie:", choices)
                if st.button("Confirm Selection"):
                    cleaned_choice = recommender.clean_title_text(choice)
                    status, movie_info, recs = recommender.get_content_recommendations(cleaned_choice, n=n_recs)

            if status == "ok":
                output.append(f"🎬 FINDING RECOMMENDATIONS FOR: '{title}'")
                output.append("="*50)
                output.append(f"🎯 Found: {movie_info['names']}")
                output.append(f"📅 Year: {movie_info.get('date_x','Unknown')}")
                output.append(f"🎭 Genre: {movie_info['genre']}")
                output.append(f"⭐ Score: {movie_info['score']}")
                output.append(f"📝 Overview: {str(movie_info['overview'])[:100]}...\n")
                output.append(f"🔥 TOP {n_recs} RECOMMENDATIONS (SORTED BY HIGHEST SIMILARITY):")
                output.append("-"*70)

                for i, (_, rec) in enumerate(recs.iterrows()):
                    similarity_percent = rec['similarity'] * 100
                    level = recommender.get_similarity_level(rec['similarity'])
                    if i == 0:
                        output.append(f"🏆 {i+1:2d}. {rec['names'][:40]} ⭐ TOP MATCH!")
                    else:
                        output.append(f"   {i+1:2d}. {rec['names'][:40]}")
                    output.append(f"    🎯 Similarity: {rec['similarity']:.4f} ({similarity_percent:.1f}%) - {level}")
                    output.append(f"    ⭐ Rating: {rec['score']:.2f}")
                    output.append(f"    🎭 Genre: {rec['genre']}\n")

    # ----------------- Search by Genre -----------------
    elif option.startswith("2️⃣"):
        genre = st.text_input("🎭 Enter a genre:")
        if st.button("Search Genre"):
            matches = recommender.qualified_movies[recommender.qualified_movies['genre'].str.contains(genre, case=False, na=False)]
            output.append(f"✅ Found {len(matches)} movies in genre '{genre}'")
            st.dataframe(matches[['names','genre','score']].head(10))

    # ----------------- Search by Crew -----------------
    elif option.startswith("3️⃣"):
        crew = st.text_input("👥 Enter crew member name:")
        if st.button("Search Crew"):
            matches = recommender.qualified_movies[recommender.qualified_movies['crew'].str.contains(crew, case=False, na=False)]
            output.append(f"✅ Found {len(matches)} movies with '{crew}'")
            st.dataframe(matches[['names','crew','genre','score']].head(10))

    # ----------------- Advanced Search -----------------
    elif option.startswith("4️⃣"):
        genre = st.text_input("🎭 Genre (optional):") or None
        crew = st.text_input("👥 Crew (optional):") or None
        min_rating = st.number_input("⭐ Minimum rating:", 0.0, 10.0, 0.0)
        if st.button("Advanced Search"):
            results = recommender.qualified_movies.copy()
            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]
            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]
            results = results[results['score'] >= min_rating]
            st.dataframe(results[['names','genre','crew','score']].head(10))

    # ----------------- Browse Genres -----------------
    elif option.startswith("5️⃣"):
        st.write("🎭 Available Genres:")
        all_genres = []
        for g in recommender.qualified_movies['genre'].dropna():
            all_genres.extend(str(g).split('|'))
        st.write(pd.Series(all_genres).value_counts().head(20))

    # ----------------- Browse Crew -----------------
    elif option.startswith("6️⃣"):
        st.write("👥 Popular Crew Members:")
        all_crew = []
        for c in recommender.qualified_movies['crew'].dropna():
            all_crew.extend(re.split(r'[,|;]', str(c)))
        st.write(pd.Series(all_crew).value_counts().head(20))

    # 输出终端风格文字
    if output:
        st.code("\n".join(output), language="text")


if __name__ == "__main__":
    main()
