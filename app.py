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
        # Initialize instance variables to store data and models
        self.movies_df = None  # DataFrame for the IMDb dataset
        self.qualified_movies = None  # DataFrame for processed movies (all movies in this version)
        self.tfidf_vectorizer = None  # TF-IDF vectorizer for text features
        self.tfidf_matrix = None  # Matrix of TF-IDF features
        self.cosine_sim = None  # Cosine similarity matrix
        self.indices = None  # Mapping of movie titles to DataFrame indices
        self.average_rating = None  # Mean rating across all movies
        self.vote_threshold = None  # Vote count threshold (used for reference, not filtering)

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
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',  # Remove common English words
            max_features=10000,  # Limit to top 5000 features
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms in fewer than 2 documents
            max_df=0.8  # Ignore terms in more than 80% of documents
        )
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

    def get_content_recommendations(self, title, n=10, show_details=True):
# Print header for recommendations
            if show_details:
                print(f"\n🎬 FINDING RECOMMENDATIONS FOR: '{title}'")
                print("=" * 50)
            try:
                # Check if movie title exists
                if title not in self.indices:
                    # Try fuzzy matching for partial title
                    possible_matches = self.qualified_movies[
                        self.qualified_movies['orig_title'].str.contains(title, case=False, na=False)
                    ]
                    if possible_matches.empty:
                        return f"❌ Movie '{title}' not found!"
                    else:
                        # Show possible matches and let user choose
                        print("🔍 Did you mean one of these?")
                        match_list = []
                        for i, (idx, match) in enumerate(possible_matches.head(5).iterrows()):
                            print(f"  {i+1}. {match['names']}")
                            match_list.append(match['orig_title'])

                        # Get user selection
                        while True:
                            try:
                                choice = input(f"\n🎯 Select a movie (1-{len(match_list)}): ").strip()
                                if choice.isdigit():
                                    choice_idx = int(choice) - 1
                                    if 0 <= choice_idx < len(match_list):
                                        title = match_list[choice_idx]
                                        print(f"✅ Selected: {possible_matches.iloc[choice_idx]['names']}")
                                        break
                                    else:
                                        print(f"❌ Please enter a number between 1 and {len(match_list)}")
                                else:
                                    print("❌ Please enter a valid number")
                            except KeyboardInterrupt:
                                return "❌ Search cancelled by user"
                            except Exception as e:
                                print("❌ Invalid input. Please try again.")

                # FIXED: Get index of the movie - handle single value properly
                idx = self.indices[title]

                # FIXED: If multiple matches, take the first one
                if hasattr(idx, '__iter__') and not isinstance(idx, str):
                    idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]

                # FIXED: Get single movie info properly
                movie_info = self.qualified_movies.loc[idx]

                # Display movie details
                if show_details:
                    print(f"🎯 Found: {movie_info['names']}")
                    # FIXED: Handle date extraction properly
                    try:
                        movie_year = pd.to_datetime(movie_info['date_x'], errors='coerce')
                        if pd.notna(movie_year):
                            print(f"📅 Year: {movie_year.year}")
                        else:
                            print(f"📅 Year: Unknown")
                    except:
                        print(f"📅 Year: Unknown")

                    print(f"🎭 Genre: {movie_info['genre']}")
                    print(f"⭐ Score: {movie_info['score']} → Weighted: {movie_info['weighted_rating']:.2f}")
                    print(f"📝 Overview: {str(movie_info['overview'])[:100]}...")

                # Get similarity scores for the movie
                sim_scores = list(enumerate(self.cosine_sim[idx]))

                # Scale similarity scores to 0.8-1.0 range
                sim_values = [score for _, score in sim_scores]
                min_sim = min(sim_values) if sim_values else 0
                max_sim = max(sim_values) if sim_values else 1

                if max_sim == min_sim:  # Avoid division by zero
                    scaled_sim_scores = [(i, 0.8) for i, _ in sim_scores]
                else:
                    scaled_sim_scores = [
                        (i, 0.8 + (score - min_sim) * (1.0 - 0.8) / (max_sim - min_sim))
                        for i, score in sim_scores
                    ]

                # Sort by scaled similarity score (highest first)
                scaled_sim_scores = sorted(scaled_sim_scores, key=lambda x: x[1], reverse=True)

                # Get top n similar movies (exclude the movie itself)
                scaled_sim_scores = scaled_sim_scores[1:n+1]

                # Extract movie indices and similarity scores
                movie_indices = [i[0] for i in scaled_sim_scores]
                similarity_values = [i[1] for i in scaled_sim_scores]

                # Create recommendations DataFrame
                movies = self.qualified_movies.iloc[movie_indices].copy()
                movies['similarity'] = similarity_values

                # Compute composite score (70% similarity + 30% normalized rating)
                movies['composite_score'] = (
                    0.7 * movies['similarity'] +  # Weight similarity
                    0.3 * (movies['weighted_rating'] / 10.0)  # Weight normalized rating
                )

                # Keep similarity-based order
                recommendations = movies[['names', 'weighted_rating', 'similarity', 'composite_score', 'genre']].head(n)

                # Display recommendations
                if show_details:
                    print(f"\n🔥 TOP {n} RECOMMENDATIONS (SORTED BY HIGHEST SIMILARITY):")
                    print("-" * 70)
                    for i, (_, rec) in enumerate(recommendations.iterrows()):
                        # Convert similarity to percentage
                        similarity_percent = rec['similarity'] * 100  # converts 0.0–1.0 to 0%–100%
                        similarity_level = self.get_similarity_level(rec['similarity'])

                        # Highlight top recommendation
                        if i == 0:
                            print(f"🏆 {i+1:2d}. {rec['names'][:40]} ⭐ TOP MATCH!")
                        else:
                            print(f"   {i+1:2d}. {rec['names'][:40]}")

                        print(f"    🎯 Similarity: {rec['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")
                        print(f"    ⭐ Rating: {rec['weighted_rating']:.2f}")
                        print(f"    🎭 Genre: {rec['genre'][:50]}")
                        print()

                return recommendations

            except Exception as e:
                return f"❌ Error: {str(e)}"

# NEW ENHANCED SEARCH FUNCTIONS

    def search_by_genre(self, genre, n=10, show_details=True):
        """
        Search movies by genre and return top-rated matches
        """
        if show_details:
            print(f"\n🎭 SEARCHING BY GENRE: '{genre}'")
            print("=" * 50)

        try:
            # Filter movies that contain the specified genre
            genre_matches = self.qualified_movies[
                self.qualified_movies['genre'].str.contains(genre, case=False, na=False)
            ]

            if genre_matches.empty:
                return f"❌ No movies found with genre '{genre}'"

            # Sort by weighted rating (highest first)
            genre_matches = genre_matches.nlargest(n, 'weighted_rating')

            if show_details:
                print(f"✅ Found {len(genre_matches)} movies with genre '{genre}'")
                print(f"\n🏆 TOP {len(genre_matches)} MOVIES IN '{genre.upper()}' GENRE:")
                print("-" * 70)

                for i, (_, movie) in enumerate(genre_matches.iterrows()):
                    print(f"🎬 {i+1:2d}. {movie['names'][:45]}")
                    print(f"    ⭐ Rating: {movie['weighted_rating']:.2f}")
                    print(f"    🎭 Genres: {movie['genre'][:50]}")

                    # Show year if available
                    try:
                        movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                        if pd.notna(movie_year):
                            print(f"    📅 Year: {movie_year.year}")
                    except:
                        pass
                    print()

            return genre_matches[['names', 'weighted_rating', 'genre', 'overview']].head(n)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def search_by_crew(self, crew_name, n=10, show_details=True):
        """
        Search movies by crew member (director, actor, etc.)
        """
        if show_details:
            print(f"\n👥 SEARCHING BY CREW: '{crew_name}'")
            print("=" * 50)

        try:
            # Filter movies that contain the specified crew member
            crew_matches = self.qualified_movies[
                self.qualified_movies['crew'].str.contains(crew_name, case=False, na=False)
            ]

            if crew_matches.empty:
                return f"❌ No movies found with crew member '{crew_name}'"

            # Sort by weighted rating (highest first)
            crew_matches = crew_matches.nlargest(n, 'weighted_rating')

            if show_details:
                print(f"✅ Found {len(crew_matches)} movies with '{crew_name}'")
                print(f"\n🏆 TOP {len(crew_matches)} MOVIES WITH '{crew_name.upper()}':")
                print("-" * 70)

                for i, (_, movie) in enumerate(crew_matches.iterrows()):
                    print(f"🎬 {i+1:2d}. {movie['names'][:45]}")
                    print(f"    ⭐ Rating: {movie['weighted_rating']:.2f}")
                    print(f"    🎭 Genre: {movie['genre'][:50]}")

                    # Show year if available
                    try:
                        movie_year = pd.to_datetime(movie['date_x'], errors='coerce')
                        if pd.notna(movie_year):
                            print(f"    📅 Year: {movie_year.year}")
                    except:
                        pass
                    print()

            return crew_matches[['names', 'weighted_rating', 'genre', 'crew']].head(n)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def advanced_search(self, genre=None, crew=None, min_rating=None, max_results=10, show_details=True):
        """
        Advanced search combining multiple criteria
        """
        if show_details:
            print(f"\n🔍 ADVANCED SEARCH")
            print("=" * 30)
            print(f"Genre: {genre if genre else 'Any'}")
            print(f"Crew: {crew if crew else 'Any'}")
            print(f"Min Rating: {min_rating if min_rating else 'Any'}")

        try:
            # Start with all movies
            results = self.qualified_movies.copy()

            # Apply genre filter
            if genre:
                results = results[results['genre'].str.contains(genre, case=False, na=False)]
                if show_details:
                    print(f"📊 After genre filter: {len(results)} movies")

            # Apply crew filter
            if crew:
                results = results[results['crew'].str.contains(crew, case=False, na=False)]
                if show_details:
                    print(f"📊 After crew filter: {len(results)} movies")

            # Apply rating filter
            if min_rating:
                results = results[results['weighted_rating'] >= min_rating]
                if show_details:
                    print(f"📊 After rating filter: {len(results)} movies")

            if results.empty:
                return "❌ No movies match your criteria"

            # Sort by weighted rating
            results = results.nlargest(max_results, 'weighted_rating')

            if show_details:
                print(f"\n🏆 TOP {len(results)} MATCHING MOVIES:")
                print("-" * 60)

                for i, (_, movie) in enumerate(results.iterrows()):
                    print(f"🎬 {i+1:2d}. {movie['names'][:40]}")
                    print(f"    ⭐ Rating: {movie['weighted_rating']:.2f}")
                    print(f"    🎭 Genre: {movie['genre'][:45]}")
                    if crew:
                        # Highlight the searched crew member
                        crew_info = movie['crew'][:60]
                        if crew.lower() in crew_info.lower():
                            crew_info = crew_info.replace(crew, f"**{crew}**")
                        print(f"    👥 Crew: {crew_info}")
                    print()

            return results[['names', 'weighted_rating', 'genre', 'crew']].head(max_results)

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_genre_list(self, top_n=20):
        """Get list of available genres"""
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))

        genre_counts = pd.Series(all_genres).value_counts().head(top_n)

        print(f"🎭 TOP {top_n} AVAILABLE GENRES:")
        print("-" * 40)
        for i, (genre, count) in enumerate(genre_counts.items()):
            print(f"{i+1:2d}. {genre} ({count} movies)")

        return genre_counts.index.tolist()

    def get_popular_crew(self, top_n=20):
        """Get list of popular crew members"""
        # Extract individual crew members
        all_crew = []
        for crew in self.qualified_movies['crew'].dropna():
            # Split by common delimiters
            crew_members = re.split(r'[,|;]', str(crew))
            for member in crew_members:
                member = member.strip()
                if len(member) > 2:  # Filter out very short names
                    all_crew.append(member)

        crew_counts = pd.Series(all_crew).value_counts().head(top_n)

        print(f"👥 TOP {top_n} POPULAR CREW MEMBERS:")
        print("-" * 50)
        for i, (person, count) in enumerate(crew_counts.items()):
            print(f"{i+1:2d}. {person} ({count} movies)")

        return crew_counts.index.tolist()

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

