# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error
from math import sqrt

# ---------------------------
# Recommendation System Class
# ---------------------------
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

    # --- utilities ---
    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    # --- load & preprocess ---
    def load_imdb_data(self, file_obj):
        # file_obj can be file path or uploaded file-like
        self.movies_df = pd.read_csv(file_obj, low_memory=False)
        # safe-fill missing cols if absent
        if 'overview' not in self.movies_df.columns:
            self.movies_df['overview'] = ''
        if 'genre' not in self.movies_df.columns:
            self.movies_df['genre'] = 'Unknown'
        if 'crew' not in self.movies_df.columns:
            self.movies_df['crew'] = 'Unknown'
        if 'orig_title' not in self.movies_df.columns and 'names' in self.movies_df.columns:
            # fallback
            self.movies_df['orig_title'] = self.movies_df['names']
        self.preprocess_data()
        self.calculate_weighted_ratings()

    def preprocess_data(self):
        # fill missing
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')

        # keep original
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # dedupe
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)

        # enhanced content
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )

        # ensure numeric fields exist
        for col in ['score', 'revenue', 'budget_x']:
            if col not in self.movies_df.columns:
                self.movies_df[col] = np.nan

    def calculate_weighted_ratings(self):
        # Create synthetic vote_count if not exist
        if 'vote_count' not in self.movies_df.columns:
            # synthetic approximation
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1_000_000) *
                (self.movies_df['score'].fillna(5) / 10) *
                np.random.uniform(3, 6, len(self.movies_df))
            ).astype(int)
        self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        self.average_rating = float(self.movies_df['score'].mean()) if 'score' in self.movies_df.columns else 5.0
        self.vote_threshold = int(self.movies_df['vote_count'].quantile(0.90)) if not self.movies_df.empty else 1

        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count'] if 'vote_count' in x else 0
            R = x['score'] if 'score' in x else self.average_rating
            return (v/(v+m) * R) + (m/(m+v) * C) if (v + m) > 0 else C

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.qualified_movies = self.movies_df.copy()

    # --- build TF-IDF & similarity ---
    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
        # TF-IDF with same params as newcode
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

    # --- helper for similarity label ---
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

    # --- centralized display (returns text) ---
    def display_centralized_results(self, results_df, search_type="Search", original_query="", n=10):
        if results_df is None:
            return "❌ No results."
        if isinstance(results_df, str):
            return results_df
        if results_df.empty:
            return "❌ No results found!"
        display_count = min(n, len(results_df))
        display_results = results_df.head(display_count).copy()
        out_lines = []
        out_lines.append("="*80)
        out_lines.append(f"🎬 {search_type.upper()} RESULTS FOR: '{original_query}'")
        out_lines.append("="*80)
        out_lines.append(f"📊 Showing {display_count} results:")
        out_lines.append("-" * 80)
        for i, (_, movie) in enumerate(display_results.iterrows()):
            if i == 0 and search_type.endswith("Recommendations"):
                out_lines.append(f"🏆 {i+1}. {movie.get('names', movie.get('original_title',''))[:60]} ⭐ TOP MATCH!")
            else:
                out_lines.append(f"🎬 {i+1}. {movie.get('names', movie.get('original_title',''))[:60]}")
            # info
            info_parts = []
            try:
                if 'date_x' in movie.index:
                    movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                    if pd.notna(movie_year):
                        info_parts.append(f"📅 Year: {movie_year.year}")
            except Exception:
                pass
            if 'weighted_rating' in movie.index:
                info_parts.append(f"⭐ Rating: {movie['weighted_rating']:.2f}")
            elif 'score' in movie.index:
                info_parts.append(f"⭐ Score: {movie['score']:.1f}")
            if info_parts:
                out_lines.append(" ".join(info_parts))
            # similarity
            if 'similarity' in movie.index:
                sim = movie['similarity']
                out_lines.append(f"🎯 Similarity: {sim:.4f} ({sim*100:.1f}%) - {self.get_similarity_level(sim)}")
            # genre/crew/country/lang
            if 'genre' in movie.index:
                out_lines.append(f"🎭 Genre: {str(movie['genre'])[:120]}")
            if 'crew' in movie.index:
                out_lines.append(f"👥 Crew: {str(movie['crew'])[:120]}")
            if 'orig_lang' in movie.index:
                out_lines.append(f"🗣️ Language: {movie['orig_lang']}")
            if 'country' in movie.index:
                out_lines.append(f"🌍 Country: {str(movie['country'])[:60]}")
            if 'budget_x' in movie.index and pd.notna(movie['budget_x']) and movie['budget_x']>0:
                out_lines.append(f"💰 Budget: ${int(movie['budget_x']):,}")
            if 'revenue' in movie.index and pd.notna(movie['revenue']) and movie['revenue']>0:
                out_lines.append(f"💵 Revenue: ${int(movie['revenue']):,}")
            out_lines.append("")
        out_lines.append("="*80)
        return "\n".join(out_lines)

    # -------------------------
    # Search / Recommendation Methods
    # -------------------------
    def get_content_recommendations(self, title, n=10):
        if self.indices is None:
            return None, None, None
        if title not in self.indices:
            # fuzzy find candidates
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(8), None

        # exact
        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_values = [score for _, score in sim_scores]
        if sim_values:
            min_sim = min(sim_values)
            max_sim = max(sim_values)
        else:
            min_sim, max_sim = 0, 1

        if max_sim == min_sim:
            scaled_sim_scores = [(i, 0.8) for i, _ in sim_scores]
        else:
            scaled_sim_scores = [
                (i, 0.8 + (score - min_sim) * (1.0 - 0.8) / (max_sim - min_sim))
                for i, score in sim_scores
            ]

        scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i for i, _ in scaled_sim_scores]
        similarity_values = [s for _, s in scaled_sim_scores]
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values
        # return original movie row as series
        return "ok", self.qualified_movies.loc[idx], movies

    def search_by_genre(self, genre, n=10):
        try:
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]
            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"
            return self.display_centralized_results(genre_matches.nlargest(n, 'weighted_rating'), "Genre Search", genre, n)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10):
        try:
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]
            if crew_matches.empty:
                return f"❌ No movies found with crew member '{crew_name}'"
            return self.display_centralized_results(crew_matches.nlargest(n, 'weighted_rating'), "Crew Search", crew_name, n)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_top_movies_by_rating(self, n=20):
        try:
            top_movies = self.qualified_movies.nlargest(n, 'weighted_rating').copy()
            return self.display_centralized_results(top_movies, "Top Rated Movies", f"Top {n} Movies", n)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_year(self, year, n=10):
        try:
            self.qualified_movies['year'] = pd.to_datetime(self.qualified_movies['date_x'], errors='coerce').dt.year
            year_matches = self.qualified_movies[self.qualified_movies['year'] == year]
            if year_matches.empty:
                return f"❌ No movies found from year {year}"
            return self.display_centralized_results(year_matches.nlargest(n, 'weighted_rating'), "Year Search", year, n)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_country(self, country, n=10):
        try:
            if 'country' not in self.qualified_movies.columns:
                return "❌ Country information not available in dataset"
            country_matches = self.qualified_movies[
                self.qualified_movies['country'].astype(str).str.contains(country, case=False, na=False)
            ]
            if country_matches.empty:
                return f"❌ No movies found from country '{country}'"
            return self.display_centralized_results(country_matches.nlargest(n, 'weighted_rating'), "Country Search", country, n)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_language(self, language, n=10):
        try:
            if 'orig_lang' not in self.qualified_movies.columns:
                return "❌ Language information not available in dataset"
            lang_matches = self.qualified_movies[
                self.qualified_movies['orig_lang'].str.contains(language, case=False, na=False)
            ]
            if lang_matches.empty:
                return f"❌ No movies found in language '{language}'"
            return self.display_centralized_results(lang_matches.nlargest(n, 'weighted_rating'), "Language Search", language, n)
        except Exception as e:
            return f"❌ Error: {str(e)}"

    # -------------------------
    # Hybrid Recommendation
    # -------------------------
    def get_hybrid_recommendations(self, title, n=10, alpha=0.7):
        # ensure TF-IDF built
        if self.cosine_sim is None or self.indices is None:
            self.build_content_based_system()
        # normalize popularity
        if 'popularity_norm' not in self.qualified_movies.columns:
            ratings = self.qualified_movies['weighted_rating'].fillna(self.average_rating)
            min_r, max_r = ratings.min(), ratings.max()
            denom = (max_r - min_r) if max_r != min_r else 1.0
            self.qualified_movies['popularity_norm'] = (ratings - min_r) / denom

        if title not in self.indices:
            # fuzzy fallback
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            # if fuzzy, return choose
            return "choose", possible_matches.head(8), None

        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]
        candidate_idx = [i for i, _ in sim_scores]
        candidates = self.qualified_movies.iloc[candidate_idx].copy()
        candidates['similarity'] = [s for _, s in sim_scores]
        candidates['hybrid_score'] = alpha * candidates['similarity'] + (1 - alpha) * candidates['popularity_norm']
        recommendations = candidates.sort_values('hybrid_score', ascending=False).head(n)
        return "ok", self.qualified_movies.loc[idx], recommendations

    # -------------------------
    # Evaluation Suite (Precision/Recall/F1 and RMSE)
    # -------------------------
    def evaluate_precision_recall_f1(self, k=10, sample_size=200):
        """Macro-averaged precision/recall/f1 using genre-overlap as relevance."""
        if self.indices is None:
            return {'precision': 0, 'recall': 0, 'f1': 0}
        titles = self.qualified_movies['orig_title'].tolist()
        # limit sample for speed
        titles = titles[:min(sample_size, len(titles))]
        precisions = []
        recalls = []
        for title in titles:
            src = self.qualified_movies[self.qualified_movies['orig_title'] == title]
            if src.empty:
                continue
            src_genres = set(str(src.iloc[0]['genre']).split('|'))
            relevant_mask = self.qualified_movies['genre'].apply(
                lambda g: len(src_genres.intersection(set(str(g).split('|')))) > 0
            )
            relevant_indices = set(self.qualified_movies[relevant_mask].index) - set(src.index)
            # get recommendations
            status, _, recs = self.get_content_recommendations(title, n=k)
            if recs is None or recs.empty:
                continue
            rec_indices = set(recs.index)
            tp = len(rec_indices & relevant_indices)
            precisions.append(tp / max(1, len(rec_indices)))
            recalls.append(tp / max(1, len(relevant_indices)))
        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls)) if recalls else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def evaluate_rating_prediction_rmse(self, k=10, sample_size=200):
        """Predict rating by weighted neighbors and compute RMSE."""
        if self.cosine_sim is None:
            return {'mse': None, 'rmse': None}
        n = len(self.qualified_movies)
        sample_indices = list(range(min(sample_size, n)))
        y_true = []
        y_pred = []
        for idx in sample_indices:
            sims = list(enumerate(self.cosine_sim[idx]))
            sims = [(i, s) for i, s in sims if i != idx]
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:k]
            if not sims:
                continue
            neighbor_idx = [i for i, _ in sims]
            weights = np.array([s for _, s in sims], dtype=float)
            neighbor_ratings = self.qualified_movies.iloc[neighbor_idx]['weighted_rating'].fillna(self.average_rating).values
            w_sum = weights.sum()
            pred = float((weights @ neighbor_ratings) / w_sum) if w_sum > 0 else float(np.mean(neighbor_ratings))
            y_true.append(float(self.qualified_movies.iloc[idx]['weighted_rating']))
            y_pred.append(pred)
        if not y_true:
            return {'mse': None, 'rmse': None}
        errors = np.array(y_true) - np.array(y_pred)
        mse = float(np.mean(errors ** 2))
        rmse = float(sqrt(mse))
        return {'mse': mse, 'rmse': rmse}

    def run_all_evaluations(self, k=10, sample_size=200, progress_callback=None):
        """Run both PRF and RMSE, call progress_callback(percentage, message) if provided."""
        results = {}
        # PRF
        if progress_callback: progress_callback(5, "Starting Precision/Recall/F1 evaluation...")
        prf = self.evaluate_precision_recall_f1(k=k, sample_size=sample_size)
        results.update(prf)
        if progress_callback: progress_callback(60, "Running RMSE rating prediction evaluation...")
        rmse = self.evaluate_rating_prediction_rmse(k=k, sample_size=sample_size)
        results.update(rmse)
        if progress_callback: progress_callback(100, "Evaluation completed.")
        return results

# ---------------------------
# Streamlit App UI
# ---------------------------
def main():
    st.set_page_config(page_title="Enhanced IMDB Recommender", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")

    # sidebar - reset + file uploader
    with st.sidebar:
        st.markdown("### Upload dataset")
        uploaded_file = st.file_uploader("Upload IMDB dataset (CSV)", type="csv")
        if st.button("🔄 Reset All (Clear session state)"):
            # preserve uploader content won't persist across rerun, but clear other states
            for k in list(st.session_state.keys()):
                try:
                    del st.session_state[k]
                except Exception:
                    pass
            st.experimental_rerun()

    if not uploaded_file:
        st.warning("Please upload imdb_movies.csv in the sidebar to use the app.")
        st.stop()

    # initialize recommender
    recommender = IMDBContentBasedRecommendationSystem()
    try:
        recommender.load_imdb_data(uploaded_file)
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

    # build TF-IDF & similarity
    try:
        recommender.build_content_based_system()
    except Exception as e:
        st.error(f"Failed to build content system: {e}")
        st.stop()

    # main menu
    option = st.radio("🎯 SEARCH OPTIONS:", [
        "1️⃣ Search by Movie Title (Content-based recommendations)",
        "2️⃣ Search by Genre (Top-rated movies in genre)",
        "3️⃣ Search by Crew Member (Movies with specific actor/director)",
        "4️⃣ Search by Year (Movies from specific year)",
        "5️⃣ Search by Country (Movies from specific country)", 
        "6️⃣ Search by Language (Movies in specific language)",
        "7️⃣ Top Rated Movies (Highest rated films)",
        "8️⃣ Hybrid Recommendations (Content + Popularity)",
        "9️⃣ Run Evaluation Suite (Precision/Recall & RMSE)"
    ])

    # --- 1. Title / content recommendations (with fuzzy choose)
    if option.startswith("1️⃣"):
        st.subheader("🎬 Content-based Recommendations")
        title_input = st.text_input("Enter a movie title (partial or full):", key="title_input")
        n_recs = st.slider("Number of recommendations", 1, 20, 10, key="title_n_recs")
        if st.button("Get Recommendations", key="get_recs_btn"):
            cleaned = recommender.clean_title_text(title_input)
            status, movie_info, recs = recommender.get_content_recommendations(cleaned, n=n_recs)
            if status is None:
                st.error("No matches found. Try another title.")
            elif status == "choose":
                st.info("Multiple possible matches found — please choose the correct one:")
                choices = movie_info['names'].tolist() if 'names' in movie_info else movie_info['original_title'].tolist()
                choice = st.selectbox("Choose movie", choices, key="choose_title_select")
                if st.button("Confirm selection", key="confirm_choice_btn"):
                    cleaned_choice = recommender.clean_title_text(choice)
                    status2, movie_info2, recs2 = recommender.get_content_recommendations(cleaned_choice, n=n_recs)
                    if status2 == "ok":
                        formatted = recommender.display_centralized_results(recs2, "Content Recommendations", movie_info2['names'], n=n_recs)
                        st.code(formatted, language="text")
                    else:
                        st.error("No recommendations available for selected movie.")
            elif status == "ok":
                formatted = recommender.display_centralized_results(recs, "Content Recommendations", movie_info['names'], n=n_recs)
                st.code(formatted, language="text")

    # --- 2. Genre
    elif option.startswith("2️⃣"):
        st.subheader("🎭 Search by Genre")
        genre = st.text_input("Enter genre (e.g. Comedy, Drama):", key="genre_input")
        n_results = st.slider("Number of results", 1, 50, 10, key="genre_n")
        if st.button("Search by Genre", key="genre_btn"):
            out = recommender.search_by_genre(genre, n=n_results)
            if isinstance(out, str):
                if out.startswith("❌"):
                    st.error(out)
                else:
                    st.code(out, language="text")
            else:
                st.code(out, language="text")

    # --- 3. Crew
    elif option.startswith("3️⃣"):
        st.subheader("👥 Search by Crew Member")
        crew = st.text_input("Enter crew member name (actor/director):", key="crew_input")
        n_results = st.slider("Number of results", 1, 50, 10, key="crew_n")
        if st.button("Search by Crew", key="crew_btn"):
            out = recommender.search_by_crew(crew, n=n_results)
            if isinstance(out, str) and out.startswith("❌"):
                st.error(out)
            else:
                st.code(out, language="text")

    # --- 4. Year
    elif option.startswith("4️⃣"):
        st.subheader("📅 Search by Year")
        year = st.number_input("Enter release year:", min_value=1800, max_value=2100, value=2020, key="year_input")
        n_results = st.slider("Number of results", 1, 50, 10, key="year_n")
        if st.button("Search by Year", key="year_btn"):
            out = recommender.search_by_year(year, n=n_results)
            if isinstance(out, str) and out.startswith("❌"):
                st.error(out)
            else:
                st.code(out, language="text")

    # --- 5. Country
    elif option.startswith("5️⃣"):
        st.subheader("🌍 Search by Country")
        country = st.text_input("Enter country name:", key="country_input")
        n_results = st.slider("Number of results", 1, 50, 10, key="country_n")
        if st.button("Search by Country", key="country_btn"):
            out = recommender.search_by_country(country, n=n_results)
            if isinstance(out, str) and out.startswith("❌"):
                st.error(out)
            else:
                st.code(out, language="text")

    # --- 6. Language
    elif option.startswith("6️⃣"):
        st.subheader("🗣️ Search by Language")
        lang = st.text_input("Enter language (e.g. English):", key="lang_input")
        n_results = st.slider("Number of results", 1, 50, 10, key="lang_n")
        if st.button("Search by Language", key="lang_btn"):
            out = recommender.search_by_language(lang, n=n_results)
            if isinstance(out, str) and out.startswith("❌"):
                st.error(out)
            else:
                st.code(out, language="text")

    # --- 7. Top Rated ---
    elif option.startswith("7️⃣"):
        st.subheader("🏅 Top Rated Movies")
        n_results = st.slider("Number of top movies to show", 1, 100, 20, key="top_n")
        if st.button("Show Top Rated Movies", key="top_btn"):
            out = recommender.get_top_movies_by_rating(n=n_results)
            if isinstance(out, str) and out.startswith("❌"):
                st.error(out)
            else:
                st.code(out, language="text")

    # --- 8. Hybrid Recommendations ---
    elif option.startswith("8️⃣"):
        st.subheader("🔀 Hybrid Recommendations (Content + Popularity)")
        title_input_h = st.text_input("Enter a movie title (for hybrid recs):", key="hybrid_title")
        alpha = st.slider("Alpha (weight for content similarity). Higher = more content-focused", 0.0, 1.0, 0.7, step=0.05, key="alpha_slider")
        n_recs_h = st.slider("Number of hybrid recommendations", 1, 30, 10, key="hybrid_n")
        if st.button("Get Hybrid Recommendations", key="hybrid_btn"):
            cleaned = recommender.clean_title_text(title_input_h)
            status, movie_info, recs = recommender.get_hybrid_recommendations(cleaned, n=n_recs_h, alpha=alpha)
            if status is None:
                st.error("No match found for hybrid recommendations. Try different title.")
            elif status == "choose":
                st.info("Multiple matches found — pick one:")
                choices = movie_info['names'].tolist() if 'names' in movie_info else movie_info['original_title'].tolist()
                choice = st.selectbox("Choose movie", choices, key="hybrid_choose")
                if st.button("Confirm hybrid choice", key="hybrid_confirm"):
                    cleaned_choice = recommender.clean_title_text(choice)
                    status2, movie_info2, recs2 = recommender.get_hybrid_recommendations(cleaned_choice, n=n_recs_h, alpha=alpha)
                    if status2 == "ok":
                        formatted = recommender.display_centralized_results(recs2, "Hybrid Recommendations", movie_info2.get('names', ''), n=n_recs_h)
                        st.code(formatted, language="text")
                    else:
                        st.error("No hybrid recommendations available.")
            elif status == "ok":
                formatted = recommender.display_centralized_results(recs, "Hybrid Recommendations", movie_info.get('names',''), n=n_recs_h)
                st.code(formatted, language="text")

    # --- 9. Run Evaluation Suite ---
    elif option.startswith("9️⃣"):
        st.subheader("🧪 Run Evaluation Suite")
        st.markdown("This runs a quick Precision/Recall/F1 (by genre overlap) and a rating-prediction RMSE (item-item).")
        k = st.slider("k (top-k) for recommendations/evaluation", 1, 50, 10, key="eval_k")
        sample_size = st.slider("Sample size for evaluation (keeps runtime reasonable)", 10, 1000, 200, step=10, key="eval_sample")
        if st.button("Run Evaluation Suite", key="eval_btn"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(percent, message):
                progress_bar.progress(int(percent))
                status_text.info(message)

            try:
                status_text.info("Starting evaluations...")
                results = recommender.run_all_evaluations(k=k, sample_size=sample_size, progress_callback=progress_callback)
                progress_bar.empty()
                status_text.success("Evaluations finished.")
                # Display results nicely
                st.markdown("### 📊 Evaluation Results")
                # Precision/Recall/F1
                prf = {k: results.get(k) for k in ('precision', 'recall', 'f1')}
                st.write("Precision/Recall/F1 (macro-averaged):")
                st.json(prf)
                # RMSE
                rmse = {'mse': results.get('mse'), 'rmse': results.get('rmse')}
                st.write("Rating prediction:")
                st.json(rmse)
            except Exception as e:
                progress_bar.empty()
                status_text.error(f"Evaluation failed: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
