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

    def get_similarity_level(self, similarity_score):
        """Convert similarity score to descriptive level"""
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
        """
        CENTRALIZED RESULTS DISPLAY - Shows comprehensive movie information in compact format
        Returns formatted string for Streamlit display
        """
        if results_df.empty:
            return "❌ No results found!"

        # Ensure we don't exceed available results
        display_count = min(n, len(results_df))
        display_results = results_df.head(display_count).copy()

        output = []
        output.append("="*80)
        output.append(f"🎬 {search_type.upper()} RESULTS FOR: '{original_query}'")
        output.append("="*80)
        output.append(f"📊 Showing {display_count} results:")
        output.append("-" * 80)

        for i, (idx, movie) in enumerate(display_results.iterrows()):
            # Build the main line with rank and movie name
            if i == 0 and search_type == "Content Recommendations":
                main_line = f"🏆 {i+1}. {movie['names'][:45]} ⭐ TOP MATCH!"
            else:
                main_line = f"🎬 {i+1}. {movie['names'][:45]}"

            # Build the info line with year and rating
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

            info_line = " ".join(info_parts)

            output.append(main_line)
            output.append(info_line)

            # Only show similarity if the column exists (in content recommendations)
            if 'similarity' in movie.index and search_type == "Content Recommendations":
                similarity_percent = movie['similarity'] * 100
                similarity_level = self.get_similarity_level(movie['similarity'])
                output.append(f"🎯 Similarity: {movie['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")

            # Genre
            if 'genre' in movie.index:
                genre_display = str(movie['genre'])[:50]
                if len(str(movie['genre'])) > 50:
                    genre_display += "..."
                output.append(f"🎭 Genre: {genre_display}")

            # Crew
            if 'crew' in movie.index:
                crew_display = str(movie['crew'])[:60]
                if len(str(movie['crew'])) > 60:
                    crew_display += "..."
                output.append(f"👥 Crew: {crew_display}")

            if 'orig_lang' in movie.index:
                output.append(f"🗣️ Language: {movie['orig_lang']}")

            if 'country' in movie.index:
                country_display = str(movie['country'])[:30]
                if len(str(movie['country'])) > 30:
                    country_display += "..."
                output.append(f"🌍 Country: {country_display}")

            if 'budget_x' in movie.index and pd.notna(movie['budget_x']) and movie['budget_x'] > 0:
                output.append(f"💰 Budget: ${movie['budget_x']:,}")

            if 'revenue' in movie.index and pd.notna(movie['revenue']) and movie['revenue'] > 0:
                output.append(f"💵 Revenue: ${movie['revenue']:,}")

            output.append("")  # Empty line between movies

        output.append("="*80)
        return "\n".join(output)

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

    def search_by_genre(self, genre, n=10, show_details=True):
        """Search movies by genre with CENTRALIZED DISPLAY"""
        try:
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]

            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"

            # Sort by weighted rating (highest first)
            genre_matches = genre_matches.nlargest(n, 'weighted_rating')
            
            if show_details:
                return self.display_centralized_results(genre_matches, "Genre Search", genre, n)
            else:
                return genre_matches

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10, show_details=True):
        """Search movies by crew member with CENTRALIZED DISPLAY"""
        try:
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]

            if crew_matches.empty:
                return f"❌ No movies found with crew member '{crew_name}'"

            crew_matches = crew_matches.nlargest(n, 'weighted_rating')
            
            if show_details:
                return self.display_centralized_results(crew_matches, "Crew Search", crew_name, n)
            else:
                return crew_matches

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def advanced_search(self, genre=None, crew=None, min_rating=None, max_results=10, show_details=True):
        """Advanced search with CENTRALIZED DISPLAY"""
        try:
            # Build query description
            query_parts = []
            if genre:
                query_parts.append(f"Genre: {genre}")
            if crew:
                query_parts.append(f"Crew: {crew}")
            if min_rating:
                query_parts.append(f"Rating ≥ {min_rating}")

            query_description = " | ".join(query_parts) if query_parts else "All Movies"

            # Start with all movies
            results = self.qualified_movies.copy()

            # Apply filters
            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]

            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]

            if min_rating:
                results = results[results['weighted_rating'] >= min_rating]

            if results.empty:
                return "❌ No movies match your criteria"

            # Sort by weighted rating
            results = results.nlargest(max_results, 'weighted_rating')

            if show_details:
                return self.display_centralized_results(results, "Advanced Search", query_description, max_results)
            else:
                return results

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_top_movies_by_rating(self, n=20, show_details=True):
        """Get top movies by weighted rating with centralized display"""
        try:
            top_movies = self.qualified_movies.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(top_movies, "Top Rated Movies", f"Top {n} Movies", n)
            else:
                return top_movies

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_year(self, year, n=10, show_details=True):
        """Search movies by release year with centralized display"""
        try:
            # Extract year from date
            self.qualified_movies['year'] = pd.to_datetime(
                self.qualified_movies['date_x'], errors='coerce'
            ).dt.year

            year_matches = self.qualified_movies[
                self.qualified_movies['year'] == year
            ]

            if year_matches.empty:
                return f"❌ No movies found from year {year}"

            year_matches = year_matches.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(year_matches, "Year Search", year, n)
            else:
                return year_matches

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_country(self, country, n=10, show_details=True):
        """Search movies by country with centralized display"""
        try:
            if 'country' not in self.qualified_movies.columns:
                return "❌ Country information not available in dataset"

            country_matches = self.qualified_movies[
                self.qualified_movies['country'].str.contains(country, case=False, na=False)
            ]

            if country_matches.empty:
                return f"❌ No movies found from country '{country}'"

            country_matches = country_matches.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(country_matches, "Country Search", country, n)
            else:
                return country_matches

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_language(self, language, n=10, show_details=True):
        """Search movies by original language with centralized display"""
        try:
            if 'orig_lang' not in self.qualified_movies.columns:
                return "❌ Language information not available in dataset"

            lang_matches = self.qualified_movies[
                self.qualified_movies['orig_lang'].str.contains(language, case=False, na=False)
            ]

            if lang_matches.empty:
                return f"❌ No movies found in language '{language}'"

            lang_matches = lang_matches.nlargest(n, 'weighted_rating').copy()

            if show_details:
                return self.display_centralized_results(lang_matches, "Language Search", language, n)
            else:
                return lang_matches

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
# Streamlit Enhanced UI
# ====================================================
def main():
    st.set_page_config(page_title="Enhanced IMDB Recommender", layout="wide")
    st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
    st.markdown("=================================================================")
    st.markdown("✨ CENTRALIZED RESULTS: All searches now show comprehensive movie information!")
    st.markdown("📊 Displays: Name, Year, Rating, Genre, Crew, Language, Country, Similarity")

    # Add reset button in sidebar
    if st.sidebar.button("🔄 Reset All Records", type="secondary"):
        # Preserve uploaded file data
        file_uploader_key = None
        for key in st.session_state.keys():
            if "uploader" in key.lower() or "file" in key.lower():
                file_uploader_key = key
                break
        
        preserved_data = {}
        if file_uploader_key and file_uploader_key in st.session_state:
            preserved_data[file_uploader_key] = st.session_state[file_uploader_key]
        
        # Clear all session state except file uploader
        for key in list(st.session_state.keys()):
            if key not in preserved_data:
                del st.session_state[key]
        
        # Restore preserved data
        for key, value in preserved_data.items():
            st.session_state[key] = value
            
        st.success("✅ All search records cleared! CSV file preserved.")
        st.rerun()

    uploaded_file = st.sidebar.file_uploader("Upload IMDB dataset (CSV)", type="csv")
    if not uploaded_file:
        st.warning("Please upload imdb_movies.csv in the sidebar.")
        return

    # Initialize recommender
    recommender = IMDBContentBasedRecommendationSystem()
    recommender.load_imdb_data(uploaded_file)
    recommender.build_content_based_system()

    # Enhanced menu with more options
    option = st.radio("🎯 SEARCH OPTIONS:", [
        "1️⃣ Search by Movie Title (Content-based recommendations)",
        "2️⃣ Search by Genre (Top-rated movies in genre)",
        "3️⃣ Search by Crew Member (Movies with specific actor/director)",
        "4️⃣ Advanced Search (Combine multiple criteria)",
        "5️⃣ Search by Year (Movies from specific year)",
        "6️⃣ Search by Country (Movies from specific country)", 
        "7️⃣ Search by Language (Movies in specific language)",
        "8️⃣ Top Rated Movies (Highest rated films)"
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

            # Use centralized display for content recommendations
            formatted_output = recommender.display_centralized_results(
                recs, 
                "Content Recommendations", 
                f"{movie_info['names']}", 
                n_recs
            )
            st.code(formatted_output, language="text")

    # ----------------- Search by Genre -----------------
    elif option.startswith("2️⃣"):
        genre = st.text_input("🎭 Enter a genre:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Genre"):
            if genre:
                result = recommender.search_by_genre(genre, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Search by Crew -----------------
    elif option.startswith("3️⃣"):
        crew = st.text_input("👥 Enter crew member name:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Crew"):
            if crew:
                result = recommender.search_by_crew(crew, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Advanced Search -----------------
    elif option.startswith("4️⃣"):
        st.markdown("🔍 Advanced Search - Enter criteria (leave empty to skip):")
        col1, col2, col3 = st.columns(3)
        with col1:
            genre = st.text_input("🎭 Genre (optional):")
        with col2:
            crew = st.text_input("👥 Crew member (optional):")
        with col3:
            min_rating = st.number_input("⭐ Min rating (optional):", min_value=0.0, max_value=10.0, step=0.1)
        
        n_results = st.slider("📊 Max results", 1, 20, 10)
        
        if st.button("Advanced Search"):
            result = recommender.advanced_search(
                genre=genre if genre else None,
                crew=crew if crew else None,
                min_rating=min_rating if min_rating > 0 else None,
                max_results=n_results,
                show_details=True
            )
            if isinstance(result, str):
                if result.startswith("❌"):
                    st.error(result)
                else:
                    st.code(result, language="text")

    # ----------------- Search by Year -----------------
    elif option.startswith("5️⃣"):
        year_input = st.number_input("📅 Enter release year:", min_value=1900, max_value=2024, value=2020)
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Year"):
            result = recommender.search_by_year(int(year_input), n=n_results, show_details=True)
            if isinstance(result, str):
                if result.startswith("❌"):
                    st.error(result)
                else:
                    st.code(result, language="text")

    # ----------------- Search by Country -----------------
    elif option.startswith("6️⃣"):
        country = st.text_input("🌍 Enter country name:")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Country"):
            if country:
                result = recommender.search_by_country(country, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Search by Language -----------------
    elif option.startswith("7️⃣"):
        language = st.text_input("🗣️ Enter language (e.g., 'en', 'english'):")
        n_results = st.slider("📊 Number of results", 1, 20, 10)
        
        if st.button("Search by Language"):
            if language:
                result = recommender.search_by_language(language, n=n_results, show_details=True)
                if isinstance(result, str):
                    if result.startswith("❌"):
                        st.error(result)
                    else:
                        st.code(result, language="text")

    # ----------------- Top Rated Movies -----------------
    elif option.startswith("8️⃣"):
        n_results = st.slider("📊 Number of top movies", 1, 50, 20)
        
        if st.button("Get Top Movies"):
            result = recommender.get_top_movies_by_rating(n=n_results, show_details=True)
            if isinstance(result, str):
                if result.startswith("❌"):
                    st.error(result)
                else:
                    st.code(result, language="text")


if __name__ == "__main__":
    main()

