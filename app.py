# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ===========================================
# Recommendation System Class
# ===========================================
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

    def load_imdb_data(self, file_path):
        self.movies_df = pd.read_csv(file_path, low_memory=False)
        self.preprocess_data()
        self.calculate_weighted_ratings()

    def preprocess_data(self):
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

    def calculate_weighted_ratings(self):
        self.movies_df['vote_count'] = (
            (self.movies_df['revenue'].fillna(0) / 1_000_000) *
            (self.movies_df['score'].fillna(5) / 10) *
            np.random.uniform(3, 6, len(self.movies_df))
        ).astype(int)

        self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)
        self.average_rating = self.movies_df['score'].mean()
        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)

        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
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

    def get_similarity_level(self, similarity_score):
        if similarity_score >= 0.87:
            return "🔥 VERY HIGH"
        elif similarity_score >= 0.86:
            return "🟢 HIGH"
        elif similarity_score >= 0.85:
            return "🟡 MODERATE"
        elif similarity_score >= 0.84:
            return "🟠 LOW"
        else:
            return "🔴 VERY LOW"

    def display_centralized_results(self, results_df, search_type="Search", original_query="", n=10):
        if results_df.empty:
            return "❌ No results found!"

        display_count = min(n, len(results_df))
        display_results = results_df.head(display_count).copy()
        output = []
        output.append("="*80)
        output.append(f"🎬 {search_type.upper()} RESULTS FOR: '{original_query}'")
        output.append("="*80)
        output.append(f"📊 Showing {display_count} results:")
        output.append("-" * 80)

        for i, (idx, movie) in enumerate(display_results.iterrows()):
            if i == 0 and search_type in ["Content Recommendations", "Hybrid Recommendations"]:
                main_line = f"🏆 {i+1}. {movie['names'][:45]} ⭐ TOP MATCH!"
            else:
                main_line = f"🎬 {i+1}. {movie['names'][:45]}"

            info_parts = []
            try:
                if 'date_x' in movie.index:
                    movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                    if pd.notna(movie_year):
                        info_parts.append(f"📅 Year: {movie_year.year}")
                    else:
                        info_parts.append("📅 Year: Unknown")
                else:
                    info_parts.append("📅 Year: N/A")
            except:
                info_parts.append("📅 Year: Unknown")

            if 'weighted_rating' in movie.index:
                info_parts.append(f"⭐ Rating: {movie['weighted_rating']:.2f}")
            elif 'score' in movie.index:
                info_parts.append(f"⭐ Score: {movie['score']:.1f}")

            output.append(main_line)
            output.append(" ".join(info_parts))

            if 'similarity' in movie.index:
                similarity_percent = movie['similarity'] * 100
                similarity_level = self.get_similarity_level(movie['similarity'])
                output.append(f"🎯 Similarity: {movie['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")

            if 'genre' in movie.index:
                output.append(f"🎭 Genre: {movie['genre']}")
            if 'crew' in movie.index:
                output.append(f"👥 Crew: {movie['crew']}")
            if 'orig_lang' in movie.index:
                output.append(f"🗣️ Language: {movie['orig_lang']}")
            if 'country' in movie.index:
                output.append(f"🌍 Country: {movie['country']}")
            if 'budget_x' in movie.index and pd.notna(movie['budget_x']) and movie['budget_x'] > 0:
                output.append(f"💰 Budget: ${movie['budget_x']:,}")
            if 'revenue' in movie.index and pd.notna(movie['revenue']) and movie['revenue'] > 0:
                output.append(f"💵 Revenue: ${movie['revenue']:,}")
            output.append("")

        output.append("="*80)
        return "\n".join(output)

    # ----------------- Search methods -----------------
    def get_content_recommendations(self, title, n=10):
        if title not in self.indices:
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(5), None

        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_values = [score for _, score in sim_scores]
        min_sim, max_sim = min(sim_values), max(sim_values)
        scaled_sim_scores = [
            (i, 0.8 + (score - min_sim) * 0.2 / (max_sim - min_sim))
            for i, score in sim_scores
        ]
        scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in scaled_sim_scores]
        similarity_values = [i[1] for i in scaled_sim_scores]
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values
        return "ok", self.qualified_movies.loc[idx], movies

    def search_by_genre(self, genre, n=10):
        matches = self.qualified_movies[self.qualified_movies['genre'].str.contains(genre, case=False, na=False)]
        if matches.empty: return f"❌ No movies found with genre '{genre}'"
        return self.display_centralized_results(matches.nlargest(n, 'weighted_rating'), "Genre Search", genre, n)

    def search_by_crew(self, crew_name, n=10):
        matches = self.qualified_movies[self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)]
        if matches.empty: return f"❌ No movies found with crew member '{crew_name}'"
        return self.display_centralized_results(matches.nlargest(n, 'weighted_rating'), "Crew Search", crew_name, n)

    def get_top_movies_by_rating(self, n=20):
        return self.display_centralized_results(self.qualified_movies.nlargest(n, 'weighted_rating'),
                                                "Top Rated Movies", f"Top {n} Movies", n)

    def search_by_year(self, year, n=10):
        self.qualified_movies['year'] = pd.to_datetime(self.qualified_movies['date_x'], errors='coerce').dt.year
        matches = self.qualified_movies[self.qualified_movies['year'] == year]
        if matches.empty: return f"❌ No movies found from year {year}"
        return self.display_centralized_results(matches.nlargest(n, 'weighted_rating'), "Year Search", year, n)

    def search_by_country(self, country, n=10):
        matches = self.qualified_movies[self.qualified_movies['country'].astype(str).str.contains(country, case=False, na=False)]
        if matches.empty: return f"❌ No movies found from country '{country}'"
        return self.display_centralized_results(matches.nlargest(n, 'weighted_rating'), "Country Search", country, n)

    def search_by_language(self, language, n=10):
        matches = self.qualified_movies[self.qualified_movies['orig_lang'].str.contains(language, case=False, na=False)]
        if matches.empty: return f"❌ No movies found in language '{language}'"
        return self.display_centralized_results(matches.nlargest(n, 'weighted_rating'), "Language Search", language, n)

    # ----------------- Hybrid Recommendation -----------------
    def get_hybrid_recommendations(self, title, n=10, alpha=0.7):
        if self.cosine_sim is None or self.indices is None:
            self.build_content_based_system()
        if 'popularity_norm' not in self.qualified_movies.columns:
            ratings = self.qualified_movies['weighted_rating'].fillna(self.average_rating)
            self.qualified_movies['popularity_norm'] = (ratings - ratings.min()) / (ratings.max() - ratings.min())
        if title not in self.indices:
            return None, None, None
        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]
        candidates = self.qualified_movies.iloc[[i for i, _ in sim_scores]].copy()
        candidates['similarity'] = [s for _, s in sim_scores]
        candidates['hybrid_score'] = alpha*candidates['similarity'] + (1-alpha)*candidates['popularity_norm']
        recs = candidates.sort_values('hybrid_score', ascending=False).head(n)
        return "ok", self.qualified_movies.loc[idx], recs

    # ----------------- Evaluation Suite -----------------
    def run_all_evaluations(self, k=10):
        results = {}
        titles = self.qualified_movies['orig_title'].tolist()[:50]  # 限制测试数量避免太慢
        precisions, recalls = [], []
        for title in titles:
            src = self.qualified_movies[self.qualified_movies['orig_title'] == title]
            if src.empty: continue
            src_genres = set(str(src.iloc[0]['genre']).split('|'))
            relevant = self.qualified_movies[self.qualified_movies['genre'].apply(
                lambda g: len(src_genres.intersection(set(str(g).split('|')))) > 0
            )]
            relevant_idx = set(relevant.index) - set(src.index)
            _, _, recs = self.get_content_recommendations(title, n=k)
            if recs is None or recs.empty: continue
            rec_idx = set(recs.index)
            tp = len(rec_idx & relevant_idx)
            precisions.append(tp/len(rec_idx))
            recalls.append(tp/max(1,len(relevant_idx)))
        results['precision'] = np.mean(precisions) if precisions else 0
        results['recall'] = np.mean(recalls) if recalls else 0
        results['f1'] = (2*results['precision']*results['recall'] /
                         (results['precision']+results['recall'])) if results['precision']+results['recall']>0 else 0
        return results

# ===========================================
# Streamlit Enhanced UI
# ===========================================
def main():
    st.set_page_config(page_title="Enhanced IMDB Recommender", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")

    uploaded_file = st.sidebar.file_uploader("Upload IMDB dataset (CSV)", type="csv")
    if not uploaded_file:
        st.warning("Please upload imdb_movies.csv in the sidebar.")
        return

    recommender = IMDBContentBasedRecommendationSystem()
    recommender.load_imdb_data(uploaded_file)
    recommender.build_content_based_system()

    option = st.radio("🎯 SEARCH OPTIONS:", [
        "1️⃣ Search by Movie Title (Content-based recommendations)",
        "2️⃣ Search by Genre",
        "3️⃣ Search by Crew",
        "4️⃣ Search by Year",
        "5️⃣ Search by Country", 
        "6️⃣ Search by Language",
        "7️⃣ Top Rated Movies",
        "8️⃣ Hybrid Recommendations",
        "9️⃣ Run Evaluation Suite"
    ])

    if option.startswith("1️⃣"):
        title = st.text_input("🎬 Enter a movie title:")
        n_recs = st.slider("📊 Number of recommendations", 1, 20, 10)
        if st.button("Get Recommendations"):
            status, movie_info, recs = recommender.get_content_recommendations(title, n=n_recs)
            if status == "ok":
                st.code(recommender.display_centralized_results(recs, "Content Recommendations", movie_info['names'], n_recs), language="text")
    elif option.startswith("2️⃣"):
        genre = st.text_input("🎭 Enter a genre:")
        if st.button("Search by Genre"):
            st.code(recommender.search_by_genre(genre, n=10), language="text")
    elif option.startswith("3️⃣"):
        crew = st.text_input("👥 Enter crew name:")
        if st.button("Search by Crew"):
            st.code(recommender.search_by_crew(crew, n=10), language="text")
    elif option.startswith("4️⃣"):
        year = st.number_input("📅 Enter year:", min_value=1900, max_value=2025, value=2020)
        if st.button("Search by Year"):
            st.code(recommender.search_by_year(year, n=10), language="text")
    elif option.startswith("5️⃣"):
        country = st.text_input("🌍 Enter country:")
        if st.button("Search by Country"):
            st.code(recommender.search_by_country(country, n=10), language="text")
    elif option.startswith("6️⃣"):
        lang = st.text_input("🗣️ Enter language:")
        if st.button("Search by Language"):
            st.code(recommender.search_by_language(lang, n=10), language="text")
    elif option.startswith("7️⃣"):
        if st.button("Top Rated Movies"):
            st.code(recommender.get_top_movies_by_rating(n=20), language="text")
    elif option.startswith("8️⃣"):
        title = st.text_input("🎬 Enter a movie for Hybrid Recommendations:")
        n_recs = st.slider("📊 Number of hybrid recs", 1, 20, 10)
        if st.button("Get Hybrid Recommendations"):
            status, movie_info, recs = recommender.get_hybrid_recommendations(title, n=n_recs)
            if status == "ok":
                st.code(recommender.display_centralized_results(recs, "Hybrid Recommendations", movie_info['names'], n_recs), language="text")
    elif option.startswith("9️⃣"):
        if st.button("Run Evaluation Suite"):
            metrics = recommender.run_all_evaluations(k=10)
            st.write("📊 Evaluation Results:", metrics)

if __name__ == "__main__":
    main()
