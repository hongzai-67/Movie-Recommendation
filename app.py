# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from math import sqrt
import joblib
import os

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

    def clean_title_text(self, text):
        if pd.isna(text):
            return ""
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()
        return cleaned

    def load_imdb_data(self, file_obj):
        # file_obj: uploaded file-like or filepath
        self.movies_df = pd.read_csv(file_obj, low_memory=False)

        # ensure expected columns exist
        if 'overview' not in self.movies_df:
            self.movies_df['overview'] = ''
        if 'genre' not in self.movies_df:
            self.movies_df['genre'] = 'Unknown'
        if 'crew' not in self.movies_df:
            self.movies_df['crew'] = 'Unknown'
        if 'orig_title' not in self.movies_df and 'names' in self.movies_df:
            self.movies_df['orig_title'] = self.movies_df['names']

        self.preprocess_data()
        self.calculate_weighted_ratings()

    def preprocess_data(self):
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')

        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # drop duplicates based on cleaned title
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)

        # enhanced content used by TF-IDF
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str)
        )

        # ensure numeric cols exist
        for c in ['score', 'revenue', 'budget_x']:
            if c not in self.movies_df.columns:
                self.movies_df[c] = np.nan

    def calculate_weighted_ratings(self):
        # synthetic vote_count if missing
        if 'vote_count' not in self.movies_df.columns:
            self.movies_df['vote_count'] = (
                (self.movies_df['revenue'].fillna(0) / 1_000_000) *
                (self.movies_df['score'].fillna(5) / 10) *
                np.random.uniform(3, 6, len(self.movies_df))
            ).astype(int)

        self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)

        self.average_rating = float(self.movies_df['score'].mean()) if 'score' in self.movies_df else 5.0
        self.vote_threshold = int(self.movies_df['vote_count'].quantile(0.90)) if not self.movies_df.empty else 1

        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            v = x['vote_count'] if 'vote_count' in x else 0
            R = x['score'] if 'score' in x else C
            return (v/(v+m) * R) + (m/(m+v) * C) if (v + m) > 0 else C

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)
        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
        # TF-IDF params matched to your newcode.txt
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

    # ---------------------------
    # Centralized formatting (matches again.txt / Colab sample)
    # ---------------------------
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

    def _safe_val(self, ser, key):
        """Helper: returns value or None (not NaN/None-string)."""
        if key in ser.index:
            v = ser[key]
            if pd.isna(v):
                return None
            return v
        return None

    def display_centralized_results(self, results_df, search_type="Search", original_query="", n=10):
        """
        Build a text block that matches the centralized CLI-like output in again.txt / your Colab sample.
        Returns a single string suitable for st.code(..., language='text')
        """
        if results_df is None:
            return "❌ No results found!"
        if isinstance(results_df, str):
            return results_df
        if results_df.empty:
            return "❌ No results found!"

        display_count = min(n, len(results_df))
        display_results = results_df.head(display_count).copy()

        lines = []
        lines.append("="*80)
        lines.append(f"🎬 {search_type.upper()} RESULTS FOR: '{original_query}'")
        lines.append("="*80)
        lines.append(f"📊 Showing {display_count} results:")
        lines.append("-" * 80)

        for i, (_, movie) in enumerate(display_results.iterrows()):
            # safe name (avoid None)
            name = self._safe_val(movie, 'names') or self._safe_val(movie, 'original_title') or ""
            if i == 0 and search_type.endswith("Recommendations"):
                lines.append(f"🏆 {i+1}. {name[:60]} ⭐ TOP MATCH!")
            else:
                lines.append(f"🎬 {i+1}. {name[:60]}")

            # Year + Rating (build only if present)
            info_parts = []
            date_x = self._safe_val(movie, 'date_x')
            if date_x is not None:
                try:
                    movie_year = pd.to_datetime(date_x, errors='coerce')
                    if pd.notna(movie_year):
                        info_parts.append(f"📅 Year: {movie_year.year}")
                except:
                    pass

            rating_val = None
            if 'weighted_rating' in movie.index:
                rating_val = movie.get('weighted_rating')
            elif 'score' in movie.index:
                rating_val = movie.get('score')
            if rating_val is not None and not pd.isna(rating_val):
                try:
                    info_parts.append(f"⭐ Rating: {float(rating_val):.2f}")
                except:
                    pass

            if info_parts:
                lines.append(" ".join(info_parts))

            # Genre
            genre_display = self._safe_val(movie, 'genre')
            if genre_display:
                lines.append(f"🎭 Genre: {genre_display}")

            # Crew (truncate if long)
            crew_display = self._safe_val(movie, 'crew')
            if crew_display:
                crew_str = str(crew_display)
                if len(crew_str) > 120:
                    crew_str = crew_str[:117] + "..."
                lines.append(f"👥 Crew: {crew_str}")

            # Language
            lang = self._safe_val(movie, 'orig_lang')
            if lang:
                lines.append(f"🗣️ Language:  {lang}")

            # Country
            country = self._safe_val(movie, 'country')
            if country:
                lines.append(f"🌍 Country: {country}")

            # Budget & Revenue
            budget = self._safe_val(movie, 'budget_x')
            try:
                if budget is not None and float(budget) > 0:
                    lines.append(f"💰 Budget: ${float(budget):,.1f}")
            except:
                pass
            revenue = self._safe_val(movie, 'revenue')
            try:
                if revenue is not None and float(revenue) > 0:
                    lines.append(f"💵 Revenue: ${float(revenue):,.1f}")
            except:
                pass

            # Similarity (for content/hybrid)
            sim = self._safe_val(movie, 'similarity')
            if sim is not None and not pd.isna(sim):
                try:
                    simf = float(sim)
                    lines.append(f"🎯 Similarity: {simf:.4f} ({simf*100:.1f}%) - {self.get_similarity_level(simf)}")
                except:
                    pass

            # Hybrid score
            hs = self._safe_val(movie, 'hybrid_score')
            if hs is not None and not pd.isna(hs):
                try:
                    lines.append(f"🔀 Hybrid Score: {float(hs):.4f}")
                except:
                    pass

            lines.append("")

        lines.append("="*80)
        return "\n".join(lines)

    # ---------------------------
    # Search & Recommendation Methods (kept like again.txt)
    # ---------------------------
    def get_content_recommendations(self, title, n=10):
        if self.indices is None:
            return None, None, None
        if title not in self.indices:
            possible_matches = self.qualified_movies[
                self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
            ]
            if possible_matches.empty:
                return None, None, None
            return "choose", possible_matches.head(8), None

        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
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

        scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in scaled_sim_scores]
        similarity_values = [i[1] for i in scaled_sim_scores]

        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = similarity_values

        return "ok", self.qualified_movies.loc[idx], movies

    def search_by_genre(self, genre, n=10):
        matches = self.qualified_movies[
            self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
        ]
        return matches.nlargest(n, 'weighted_rating')

    def search_by_crew(self, crew, n=10):
        matches = self.qualified_movies[
            self.qualified_movies['crew'].str.contains(crew, case=False, na=False)
        ]
        return matches.nlargest(n, 'weighted_rating')

    def get_top_movies_by_rating(self, n=20):
        return self.qualified_movies.nlargest(n, 'weighted_rating')

    def search_by_year(self, year, n=10):
        self.qualified_movies['year'] = pd.to_datetime(self.qualified_movies['date_x'], errors='coerce').dt.year
        matches = self.qualified_movies[self.qualified_movies['year'] == year]
        return matches.nlargest(n, 'weighted_rating')

    def search_by_country(self, country, n=10):
        matches = self.qualified_movies[
            self.qualified_movies['country'].astype(str).str.contains(country, case=False, na=False)
        ]
        return matches.nlargest(n, 'weighted_rating')

    def search_by_language(self, language, n=10):
        matches = self.qualified_movies[
            self.qualified_movies['orig_lang'].str.contains(language, case=False, na=False)
        ]
        return matches.nlargest(n, 'weighted_rating')

    # ---------------------------
    # Hybrid & Evaluation (from newcode.txt)
    # ---------------------------
    def get_hybrid_recommendations(self, title, n=10, alpha=0.7):
        # ensure popularity_norm
        if 'popularity_norm' not in self.qualified_movies.columns:
            ratings = self.qualified_movies['weighted_rating'].fillna(self.average_rating)
            min_r, max_r = ratings.min(), ratings.max()
            denom = (max_r - min_r) if max_r != min_r else 1.0
            self.qualified_movies['popularity_norm'] = (ratings - min_r) / denom

        if title not in self.indices:
            possible = self.qualified_movies[self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)]
            if possible.empty:
                return None, None, None
            return "choose", possible.head(8), None

        idx = self.indices[title]
        if hasattr(idx, '__iter__') and not isinstance(idx, str):
            idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = [(i, s) for i, s in sim_scores if i != idx]

        candidate_indices = [i for i, _ in sim_scores]
        candidates_df = self.qualified_movies.iloc[candidate_indices].copy()
        candidates_df['similarity'] = [s for _, s in sim_scores]

        candidates_df['hybrid_score'] = alpha * candidates_df['similarity'] + (1 - alpha) * candidates_df['popularity_norm']

        recommendations = candidates_df.sort_values('hybrid_score', ascending=False).head(n)
        return "ok", self.qualified_movies.loc[idx], recommendations

    def run_all_evaluations(self, k=10, sample_size=50, progress_callback=None):
        results = {}
        titles = self.qualified_movies['orig_title'].tolist()[:sample_size]
        precisions, recalls = [], []

        for i, title in enumerate(titles):
            status, _, recs = self.get_content_recommendations(title, n=k)
            if recs is None or recs.empty:
                continue
            src = self.qualified_movies[self.qualified_movies['orig_title'] == title]
            src_genres = set(str(src.iloc[0]['genre']).split('|'))
            relevant_mask = self.qualified_movies['genre'].apply(
                lambda g: len(src_genres.intersection(set(str(g).split('|')))) > 0
            )
            relevant_indices = set(self.qualified_movies[relevant_mask].index) - set(src.index)
            rec_indices = set(recs.index)
            tp = len(rec_indices & relevant_indices)
            precisions.append(tp / max(1, len(rec_indices)))
            recalls.append(tp / max(1, len(relevant_indices)))

            if progress_callback:
                progress_callback(int((i + 1) / len(titles) * 50), "Evaluating Precision/Recall/F1...")

        precision = float(np.mean(precisions)) if precisions else 0.0
        recall = float(np.mean(recalls)) if recalls else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results.update({'precision': precision, 'recall': recall, 'f1': f1})

        # Rating prediction RMSE
        y_true, y_pred = [], []
        n_samples = min(sample_size, len(self.qualified_movies))
        for idx in range(n_samples):
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

            if progress_callback:
                progress_callback(50 + int((idx + 1) / n_samples * 50), "Evaluating RMSE...")

        if y_true:
            errors = np.array(y_true) - np.array(y_pred)
            mse = float(np.mean(errors ** 2))
            rmse = float(sqrt(mse))
            results.update({'mse': mse, 'rmse': rmse})
        else:
            results.update({'mse': None, 'rmse': None})

        return results

# ---------------------------
# Streamlit App (keeps again.txt UI & output style)
# ---------------------------
def main():
    st.set_page_config(page_title="ENHANCED IMDB RECOMMENDER", layout="wide")
    # Use title() so it appears as page title
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")

    st.markdown("=================================================================")
    st.markdown("✨ CENTRALIZED RESULTS: All searches now show comprehensive movie information!")
    st.markdown("📊 Displays: Name, Year, Rating, Genre, Crew, Language, Country, Similarity")
    st.write("")
    
    # Try to load bundled joblib first (minimal addition)
    local_joblib_path = "recommender.joblib"
    recommender = None
    if os.path.exists(local_joblib_path):
        st.sidebar.success(f"Found bundled model: {local_joblib_path}")
        try:
            recommender = joblib.load(local_joblib_path)
        except Exception as e:
            st.sidebar.error(f"Failed to load bundled model: {e}")
            recommender = None

    # If no bundled recommender, fall back to uploaded CSV
    if recommender is None:
        if not uploaded_file:
            st.sidebar.info("No bundled model found. Upload imdb_movies.csv in the sidebar or place 'recommender.joblib' next to this app.")
            st.warning("Please upload imdb_movies.csv in the sidebar.")
            return

        recommender = IMDBContentBasedRecommendationSystem()
        try:
            recommender.load_imdb_data(uploaded_file)
            recommender.build_content_based_system()
        except Exception as e:
            st.error(f"Failed to load/build dataset: {e}")
            return

    # Menu (keeps numbering and labels same as your sample)
    option = st.radio("🎯 SEARCH OPTIONS:", [
        "1️⃣ Search by Movie Title (Content-based recommendations)",
        "2️⃣ Search by Genre (Top-rated movies in genre)",
        "3️⃣ Search by Crew Member (Movies with specific actor/director)",
        "4️⃣ Search by Year (Movies from specific year)",
        "5️⃣ Search by Country (Movies from specific country)",
        "6️⃣ Search by Language (Movies in specific language)",
        "7️⃣ Top Rated Movies (Highest rated films)",
        "8️⃣ Hybrid Recommendations (Content + Popularity)",
        "9️⃣ Run Evaluation Suite (Precision/Recall & RMSE)",
    ])

    # ----------------- 1. Title (content-based) -----------------
    if option.startswith("1️⃣"):
        st.subheader("Search by Movie Title (Content-based recommendations)")
        title = st.text_input("🎬 Enter a movie title:")
        n_recs = st.slider("📊 Number of recommendations", 1, 20, 10)

        if st.button("Get Recommendations"):
            st.session_state.pop('choices_title', None)
            st.session_state.pop('confirmed_title', None)
            cleaned = recommender.clean_title_text(title)
            status, movie_info, recs = recommender.get_content_recommendations(cleaned, n=n_recs)
            if status is None:
                st.error("❌ No matches found.")
            elif status == "choose":
                # store choices in session for confirm workflow
                choices = movie_info['names'].tolist() if 'names' in movie_info else movie_info['original_title'].tolist()
                st.session_state['choices_title'] = choices
            elif status == "ok":
                formatted = recommender.display_centralized_results(recs, "Content Recommendations", movie_info.get('names', ''), n_recs)
                st.code(formatted, language="text")

        # confirm selection UI preserved using session_state
        if 'choices_title' in st.session_state:
            st.markdown("🔍 Did you mean one of these?")
            choice = st.selectbox("🎯 Select a movie:", st.session_state['choices_title'], key="choose_title_select")
            if st.button("Confirm Selection"):
                st.session_state['confirmed_title'] = choice
                del st.session_state['choices_title']

        if 'confirmed_title' in st.session_state:
            cleaned_choice = recommender.clean_title_text(st.session_state['confirmed_title'])
            status2, movie_info2, recs2 = recommender.get_content_recommendations(cleaned_choice, n=n_recs)
            if status2 == "ok":
                formatted = recommender.display_centralized_results(recs2, "Content Recommendations", movie_info2.get('names', ''), n_recs)
                st.code(formatted, language="text")

    # ----------------- 2. Genre -----------------
    elif option.startswith("2️⃣"):
        st.subheader("Search by Genre")
        genre = st.text_input("🎭 Enter a genre:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Genre"):
            res = recommender.search_by_genre(genre, n=n_results)
            formatted = recommender.display_centralized_results(res, "Genre Search", genre, n=n_results)
            st.code(formatted, language="text")

    # ----------------- 3. Crew -----------------
    elif option.startswith("3️⃣"):
        st.subheader("Search by Crew Member")
        crew = st.text_input("👥 Enter crew member name:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Crew"):
            res = recommender.search_by_crew(crew, n=n_results)
            formatted = recommender.display_centralized_results(res, "Crew Search", crew, n=n_results)
            st.code(formatted, language="text")

    # ----------------- 4. Year -----------------
    elif option.startswith("4️⃣"):
        st.subheader("Search by Year")
        year_input = st.number_input("📅 Enter release year:", min_value=1800, max_value=2100, value=2020)
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Year"):
            res = recommender.search_by_year(year_input, n=n_results)
            formatted = recommender.display_centralized_results(res, "Year Search", str(year_input), n=n_results)
            st.code(formatted, language="text")

    # ----------------- 5. Country -----------------
    elif option.startswith("5️⃣"):
        st.subheader("Search by Country")
        country = st.text_input("🌍 Enter country name:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Country"):
            res = recommender.search_by_country(country, n=n_results)
            formatted = recommender.display_centralized_results(res, "Country Search", country, n=n_results)
            st.code(formatted, language="text")

    # ----------------- 6. Language -----------------
    elif option.startswith("6️⃣"):
        st.subheader("Search by Language")
        language = st.text_input("🗣️ Enter language:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        if st.button("Search by Language"):
            res = recommender.search_by_language(language, n=n_results)
            formatted = recommender.display_centralized_results(res, "Language Search", language, n=n_results)
            st.code(formatted, language="text")

    # ----------------- 7. Top Rated -----------------
    elif option.startswith("7️⃣"):
        st.subheader("Top Rated Movies")
        n_results = st.slider("📊 Number of results", 1, 50, 20)
        if st.button("Show Top Rated Movies"):
            res = recommender.get_top_movies_by_rating(n=n_results)
            formatted = recommender.display_centralized_results(res, "Top Rated Movies", f"Top {n_results} Movies", n=n_results)
            st.code(formatted, language="text")

    # ----------------- 8. Hybrid Recommendations -----------------
    elif option.startswith("8️⃣"):
        st.subheader("Hybrid Recommendations (Content + Popularity)")
        title = st.text_input("🎬 Enter a movie title:")
        alpha = st.slider("⚖️ Alpha for hybrid (0-1)", 0.0, 1.0, 0.7, step=0.05)
        n_recs = st.slider("📊 Number of recommendations", 1, 20, 10)

        if st.button("Get Hybrid Recs"):
            st.session_state.pop('choices_hybrid', None)
            st.session_state.pop('confirmed_hybrid', None)
            cleaned = recommender.clean_title_text(title)
            status, movie_info, recs = recommender.get_hybrid_recommendations(cleaned, n=n_recs, alpha=alpha)
            if status is None:
                st.error("❌ No matches found.")
            elif status == "choose":
                choices = movie_info['names'].tolist() if 'names' in movie_info else movie_info['original_title'].tolist()
                st.session_state['choices_hybrid'] = choices
            elif status == "ok":
                formatted = recommender.display_centralized_results(recs, "Hybrid Recommendations", movie_info.get('names', ''), n_recs)
                st.code(formatted, language="text")

        if 'choices_hybrid' in st.session_state:
            st.markdown("🔍 Did you mean one of these?")
            choice = st.selectbox("🎯 Select a movie:", st.session_state['choices_hybrid'], key="choose_hybrid_select")
            if st.button("Confirm Selection (Hybrid)"):
                st.session_state['confirmed_hybrid'] = choice
                del st.session_state['choices_hybrid']

        if 'confirmed_hybrid' in st.session_state:
            cleaned_choice = recommender.clean_title_text(st.session_state['confirmed_hybrid'])
            status2, movie_info2, recs2 = recommender.get_hybrid_recommendations(cleaned_choice, n=n_recs, alpha=alpha)
            if status2 == "ok":
                formatted = recommender.display_centralized_results(recs2, "Hybrid Recommendations", movie_info2.get('names', ''), n_recs)
                st.code(formatted, language="text")

    # ----------------- 9. Run Evaluation Suite -----------------
    elif option.startswith("9️⃣"):
        st.subheader("Run Evaluation Suite (Precision/Recall & RMSE)")
        k = st.slider("k (top-k)", 1, 30, 10)
        sample_size = st.slider("Sample size (for speed)", 10, 500, 50, step=10)
        if st.button("Run Evaluation Suite"):
            prog = st.progress(0)
            status_txt = st.empty()

            def progress_cb(percent, message):
                prog.progress(percent)
                status_txt.info(message)

            try:
                status_txt.info("Starting evaluations...")
                results = recommender.run_all_evaluations(k=k, sample_size=sample_size, progress_callback=progress_cb)
                prog.empty()
                status_txt.success("Evaluations completed.")
                # Format output similar to your Colab example
                header = []
                header.append("🧪 RUNNING EVALUATIONS")
                header.append("="*30)
                st.code("\n".join(header), language="text")

                # show PRF
                prf_lines = []
                prf_lines.append(f"📏 EVALUATION (CONTENT): ")
                prf_lines.append(f"🎯 Precision (k={k}): {results.get('precision',0):.3f} ")
                prf_lines.append(f"📚 Recall (k={k}): {results.get('recall',0):.3f} ")
                prf_lines.append(f"⚖️ F1: {results.get('f1',0):.3f} ")
                st.code("\n".join(prf_lines), language="text")

                # RMSE (may be None)
                if 'mse' in results:
                    rmse_lines = []
                    rmse_lines.append("\n📉 Rating Prediction: ")
                    rmse_lines.append(f"📊 MSE: {results.get('mse'):.4f}" if results.get('mse') is not None else "📊 MSE: N/A")
                    rmse_lines.append(f"🔮 RMSE: {results.get('rmse'):.4f}" if results.get('rmse') is not None else "🔮 RMSE: N/A")
                    rmse_lines.append(f"⚡ (k={k})")
                    st.code("\n".join(rmse_lines), language="text")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

if __name__ == "__main__":
    main()



