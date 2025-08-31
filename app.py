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
        self.average_rating = 0
        self.vote_threshold = 0

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    def load_imdb_data(self, file_path):
        self.movies_df = pd.read_csv(file_path, low_memory=False)

        # 填补缺失值
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')
        if 'keywords' not in self.movies_df.columns:
            self.movies_df['keywords'] = ""
        if 'tagline' not in self.movies_df.columns:
            self.movies_df['tagline'] = ""

        # 保留原始标题
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # 去重
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)

        # FIXED: Use the same enhanced content as Colab (overview + genre + crew)
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )

        # FIXED: Create synthetic vote_count like in Colab
        if 'vote_count' not in self.movies_df.columns:
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1_000_000) *
                (self.movies_df['score'].fillna(5) / 2) *
                np.random.uniform(50, 500, len(self.movies_df))
            ).astype(int)
            self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        # FIXED: Calculate weighted rating using IMDb formula like in Colab
        self.average_rating = self.movies_df['score'].mean()
        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)
        
        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)
        
        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        
        # Use weighted_rating instead of weighted
        self.movies_df['weighted'] = self.movies_df['weighted_rating']
        
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
        
        # FIXED: Use same TF-IDF parameters as Colab
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    def get_similarity_level(self, score):
        # FIXED: Use same thresholds as Colab
        if score >= 0.87:
            return "🔥 VERY HIGH"
        elif score >= 0.86:
            return "🟢 HIGH"
        elif score >= 0.85:
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
        
        # FIXED: Handle multiple matches properly
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
            
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # FIXED: Apply the same similarity scaling as Colab (0.8-1.0 range)
        sim_values = [score for _, score in sim_scores]
        min_sim = min(sim_values) if sim_values else 0
        max_sim = max(sim_values) if sim_values else 1

        if max_sim == min_sim:
            scaled_sim_scores = [(i, 0.8) for i, _ in sim_scores]
        else:
            scaled_sim_scores = [
                (i, 0.8 + (score - min_sim) * (1.0 - 0.8) / (max_sim - min_sim))
                for i, score in sim_scores
            ]

        # 按相似度排序 (highest first)
        scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        
        movie_indices = [i[0] for i in scaled_sim_scores]
        similarity_values = [i[1] for i in scaled_sim_scores]
        
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values

        # FIXED: Keep similarity-based order (don't re-sort by weighted rating)
        return "ok", self.qualified_movies.loc[idx], movies

    # FIXED: Add missing search functions from Colab
    def search_by_genre(self, genre, n=10, show_details=True):
        """Search movies by genre and return top-rated matches"""
        try:
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]

            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"

            # Sort by weighted rating (highest first)
            genre_matches = genre_matches.nlargest(n, 'weighted_rating')
            return genre_matches[['names', 'weighted_rating', 'genre', 'overview']].head(n)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10, show_details=True):
        """Search movies by crew member"""
        try:
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]

            if crew_matches.empty:
                return f"❌ No movies found with crew member '{crew_name}'"

            crew_matches = crew_matches.nlargest(n, 'weighted_rating')
            return crew_matches[['names', 'weighted_rating', 'genre', 'crew']].head(n)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def advanced_search(self, genre=None, crew=None, min_rating=None, max_results=10):
        """Advanced search combining multiple criteria"""
        try:
            results = self.qualified_movies.copy()

            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]
            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]
            if min_rating:
                results = results[results['weighted_rating'] >= min_rating]

            if results.empty:
                return "❌ No movies match your criteria"

            results = results.nlargest(max_results, 'weighted_rating')
            return results[['names', 'weighted_rating', 'genre', 'crew']].head(max_results)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_genre_list(self, top_n=20):
        """Get list of available genres"""
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))
        genre_counts = pd.Series(all_genres).value_counts().head(top_n)
        return genre_counts.index.tolist()

    def get_popular_crew(self, top_n=20):
        """Get list of popular crew members"""
        all_crew = []
        for crew in self.qualified_movies['crew'].dropna():
            crew_members = re.split(r'[,|;]', str(crew))
            for member in crew_members:
                member = member.strip()
                if len(member) > 2:
                    all_crew.append(member)
        crew_counts = pd.Series(all_crew).value_counts().head(top_n)
        return crew_counts.index.tolist()


# ====================================================
# Streamlit Terminal-style UI
# ====================================================
def main():
    st.set_page_config(page_title="IMDB Recommender", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
    st.markdown("=================================================================")
    st.markdown("✨ NEW FEATURES: Search by Title, Genre, Crew, or Advanced Multi-Search!")

    # Add reset button in sidebar
    if st.sidebar.button("🔄 Reset All Records", type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

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

    # ----------------- Search by Title -----------------
    if option.startswith("1️⃣"):
        title = st.text_input("🎬 Enter a movie title:")
        n_recs = st.slider("📊 Number of recommendations", 1, 20, 10)

        if st.button("Get Recommendations"):
            cleaned_title = recommender.clean_title_text(title)
            status, movie_info, recs = recommender.get_content_recommendations(cleaned_title, n=n_recs)
            st.session_state.search_status = status
            st.session_state.movie_info = movie_info
            st.session_state.recs = recs
            st.session_state.cleaned_title = cleaned_title

        if "search_status" in st.session_state and st.session_state.search_status == "choose":
            st.markdown("🔍 Did you mean one of these?")
            choices = st.session_state.movie_info['names'].tolist()
            choice = st.selectbox("🎯 Select a movie:", choices, key="movie_choice")
            if st.button("Confirm Selection"):
                cleaned_choice = recommender.clean_title_text(choice)
                status, movie_info, recs = recommender.get_content_recommendations(cleaned_choice, n=n_recs)
                st.session_state.search_status = status
                st.session_state.movie_info = movie_info
                st.session_state.recs = recs
                st.session_state.cleaned_title = cleaned_choice

        if "search_status" in st.session_state and st.session_state.search_status == "ok":
            movie_info = st.session_state.movie_info
            recs = st.session_state.recs
            cleaned_title = st.session_state.cleaned_title

            output = []
            output.append(f"🎬 FINDING RECOMMENDATIONS FOR: '{cleaned_title}'")
            output.append("="*50)
            output.append(f"🎯 Found: {movie_info['names']}")
            output.append(f"📅 Year: {movie_info.get('date_x','Unknown')}")
            output.append(f"🎭 Genre: {movie_info['genre']}")
            output.append(f"⭐ Score: {movie_info['score']:.2f} → Weighted: {movie_info['weighted_rating']:.2f}")
            output.append(f"📝 Overview: {str(movie_info['overview'])[:150]}...\n")
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
                output.append(f"    ⭐ Rating: {rec['weighted_rating']:.2f}")
                output.append(f"    🎭 Genre: {rec['genre']}\n")

            st.code("\n".join(output), language="text")

    # ----------------- Search by Genre -----------------
    elif option.startswith("2️⃣"):
        genre = st.text_input("🎭 Enter a genre:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Genre"):
            if genre:
                results = recommender.search_by_genre(genre, n=n_results, show_details=False)
                if isinstance(results, pd.DataFrame):
                    st.markdown(f"### 🎭 Top {len(results)} movies in '{genre}' genre:")
                    for i, (_, movie) in enumerate(results.iterrows()):
                        st.write(f"**{i+1}. {movie['names']}**")
                        st.write(f"⭐ Rating: {movie['weighted_rating']:.2f}")
                        st.write(f"🎭 Genre: {movie['genre']}")
                        st.write("---")
                else:
                    st.error(results)

    # ----------------- Search by Crew -----------------
    elif option.startswith("3️⃣"):
        crew = st.text_input("👥 Enter crew member name:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Crew"):
            if crew:
                results = recommender.search_by_crew(crew, n=n_results, show_details=False)
                if isinstance(results, pd.DataFrame):
                    st.markdown(f"### 👥 Top {len(results)} movies with '{crew}':")
                    for i, (_, movie) in enumerate(results.iterrows()):
                        st.write(f"**{i+1}. {movie['names']}**")
                        st.write(f"⭐ Rating: {movie['weighted_rating']:.2f}")
                        st.write(f"🎭 Genre: {movie['genre']}")
                        st.write("---")
                else:
                    st.error(results)

    # ----------------- Advanced Search -----------------
    elif option.startswith("4️⃣"):
        col1, col2, col3 = st.columns(3)
        with col1:
            genre = st.text_input("🎭 Genre (optional):")
        with col2:
            crew = st.text_input("👥 Crew member (optional):")
        with col3:
            min_rating = st.number_input("⭐ Min rating (optional):", min_value=0.0, max_value=10.0, step=0.1)
        
        n_results = st.slider("📊 Max results", 1, 20, 10)
        
        if st.button("Advanced Search"):
            results = recommender.advanced_search(
                genre=genre if genre else None,
                crew=crew if crew else None,
                min_rating=min_rating if min_rating > 0 else None,
                max_results=n_results
            )
            if isinstance(results, pd.DataFrame):
                st.markdown(f"### 🔍 Search Results ({len(results)} movies):")
                for i, (_, movie) in enumerate(results.iterrows()):
                    st.write(f"**{i+1}. {movie['names']}**")
                    st.write(f"⭐ Rating: {movie['weighted_rating']:.2f}")
                    st.write(f"🎭 Genre: {movie['genre']}")
                    if crew:
                        st.write(f"👥 Crew: {movie['crew'][:100]}...")
                    st.write("---")
            else:
                st.error(results)

    # ----------------- Browse Genres -----------------
    elif option.startswith("5️⃣"):
        st.markdown("### 🎭 Available Genres:")
        genres = recommender.get_genre_list()
        for i, genre in enumerate(genres):
            st.write(f"{i+1}. {genre}")

    # ----------------- Browse Crew -----------------
    elif option.startswith("6️⃣"):
        st.markdown("### 👥 Popular Crew Members:")
        crew_list = recommender.get_popular_crew()
        for i, person in enumerate(crew_list):
            st.write(f"{i+1}. {person}")


if __name__ == "__main__":
    main()
