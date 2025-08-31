# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Define the recommendation system class
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
        """
        Clean title text by removing special characters and converting to lowercase.

        Args:
            text (str): Raw title text

        Returns:
            str: Cleaned title text with words separated by spaces
        """
        if pd.isna(text):
            return ""

        # Convert to string and remove special characters, keep only alphanumeric and spaces
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))

        # Replace multiple spaces with single space and convert to lowercase
        cleaned = re.sub(r'\s+', ' ', cleaned).lower().strip()

        return cleaned

    def load_imdb_data(self, file_path='imdb_movies.csv'):
        # Print header for user feedback
        print("🎬 LOADING IMDB DATASET FROM KAGGLE")
        print("=" * 50)
        # Load CSV file into a pandas DataFrame
        self.movies_df = pd.read_csv(file_path, low_memory=False)  # low_memory=False handles large datasets
        # Display dataset shape (rows, columns)
        print(f"✅ Original dataset shape: {self.movies_df.shape}")
        # Count of duplicated values
        print(f"🔢 Total Duplicated Values: {self.movies_df.duplicated().sum()}")
        # Count of missing (null/NaN) values
        print(f"❓ Total Missing Values:\n {self.movies_df.isnull().sum()}")
        # List column names
        print(f"📋 Columns: {list(self.movies_df.columns)}")
        # Show first three rows for preview
        print(f"📊 Sample data:")
        print(self.movies_df.head(3))
        print("\n🚀 Loading IMDB DataSet From Kaggle Completed!\n")
        # Call preprocessing method to clean and prepare data
        self.preprocess_data()


    def preprocess_data(self):
        """Clean and preprocess the movie data"""
        # Print preprocessing header
        print(f"\n🔧 PREPROCESSING DATA")
        print("=" * 30)

        # Replace missing 'overview' values with a placeholder
        self.movies_df['overview'] = self.movies_df['overview'].fillna('No description available')
        # Replace missing 'genre' values with 'Unknown'
        self.movies_df['genre'] = self.movies_df['genre'].fillna('Unknown')
        # Replace missing 'crew' values with 'Unknown'
        self.movies_df['crew'] = self.movies_df['crew'].fillna('Unknown')

        # =============================================================================
        # IMPROVED TITLE PROCESSING - SEPARATE INTO SINGLE WORDS, REMOVE SPECIAL CHARS, LOWERCASE
        # =============================================================================
        print("🎬 Processing movie titles: removing special characters, converting to lowercase...")

        # Store original title for reference
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()

        # Clean title: remove special characters, convert to lowercase, separate into words
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # Show examples of title processing
        print("📋 Title processing examples:")
        sample_titles = self.movies_df[['original_title', 'orig_title']].head(3)
        for idx, row in sample_titles.iterrows():
            print(f"    Original: '{row['original_title']}' → Processed: '{row['orig_title']}'")
        print("✅ Title processing completed!\n")

        # =============================================================================
        # REMOVE DUPLICATES
        # =============================================================================
        print("🗑️ Checking For Duplicate Movies")
        before_count = len(self.movies_df)
        self.movies_df = self.movies_df.drop_duplicates(
            subset=['orig_title'],  # remove based on cleaned titles
            keep='first'            # keep the first occurrence
        ).reset_index(drop=True)
        after_count = len(self.movies_df)
        print(f"❌ Removed {before_count - after_count} duplicates. Remaining movies: {after_count}")
        print("🎉 Duplicate removal completed!\n")

        # =============================================================================
        # CREATE ENHANCED CONTENT FEATURES
        # =============================================================================
        print("🔍 Creating enhanced content features...")
        # Create 'enhanced_content' column by combining overview, genre, and crew
        self.movies_df['enhanced_content'] = (
            self.movies_df['overview'].astype(str) + ' ' +  # Convert overview to string
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +  # Replace '|' with spaces in genres
            self.movies_df['crew'].astype(str)  # Add crew names
        )
        print("✨ Enhanced content features created successfully!\n")

        # =============================================================================
        # PREPROCESSING COMPLETION
        # =============================================================================
        print("🚀 Data preprocessing completed!\n")

    def calculate_weighted_ratings(self):
        """
        CORRECTING RATING BIAS / TRUSTWORTHY PROBLEM
        Implement IMDb's Weighted Rating Formula to solve bias issues
        MODIFIED: All movies are now qualified (no filtering by vote threshold)
        """
        # Print header and explain the bias problem
        print(f"\n⚖️  CORRECTING RATING BIAS / TRUSTWORTHY PROBLEM")
        print("=" * 55)

        # =============================================================================
        # CREATE SYNTHETIC VOTE COUNT
        # =============================================================================
        print("📊 Creating synthetic vote count based on movie popularity...")

        self.movies_df['vote_count'] = (
            (self.movies_df['revenue'].fillna(0) / 1_000_000) *   # Normalize revenue
            (self.movies_df['score'].fillna(5) / 2) *             # Scale score (default 5 if missing)
            np.random.uniform(50, 500, len(self.movies_df))       # Add random variation
        ).astype(int)  # Convert to integers

        # Ensure vote counts are at least 1
        self.movies_df['vote_count'] = self.movies_df['vote_count'].clip(lower=1)
        print("✅ Synthetic vote count created!\n")

        # =============================================================================
        # CALCULATE GLOBAL AVERAGES
        # =============================================================================
        self.average_rating = self.movies_df['score'].mean()
        print(f"📈 Global average rating (C): {self.average_rating:.2f}")

        self.vote_threshold = self.movies_df['vote_count'].quantile(0.90)
        print(f"📊 Minimum votes threshold (m): {self.vote_threshold:.0f}")

        # =============================================================================
        # APPLY IMDB WEIGHTED RATING FORMULA
        # =============================================================================
        print("\n🧮 Applying IMDb Weighted Rating Formula:")
        print("🔢   WR = (v/(v+m)) × R + (m/(m+v)) × C")

        def weighted_rating(x, m=self.vote_threshold, C=self.average_rating):
            """
            IMDb Weighted Rating Formula
            v = vote count, R = raw rating, m = min threshold, C = global mean
            """
            v = x['vote_count']
            R = x['score']
            return (v/(v+m) * R) + (m/(m+v) * C)

        self.movies_df['weighted_rating'] = self.movies_df.apply(weighted_rating, axis=1)

        # Mark all movies as qualified (no filtering)
        self.qualified_movies = self.movies_df.copy()
        print(f"✅ MODIFIED: All movies are now qualified: {self.qualified_movies.shape[0]}")
        print(f"📊 Total movies = Qualified movies: {len(self.movies_df)} movies\n")
        # =============================================================================
        # SHOW EXAMPLES OF BIAS CORRECTION
        # =============================================================================
        self.show_bias_correction_examples()

    def show_bias_correction_examples(self):
        """Show examples of how bias correction works"""
        # Print header for bias correction examples
        print(f"📋 BIAS CORRECTION EXAMPLES:")
        print("-" * 40)
        # Select movies with low votes and high ratings to show bias correction
        low_vote_high_rating = self.movies_df[
            (self.movies_df['vote_count'] < self.vote_threshold/2) &
            (self.movies_df['score'] > 8.0)
        ].head(3)
        # Display examples of low-vote, high-rating movies
        print("🔴 HIGH RATING + LOW VOTES (Bias Corrected):")
        for _, movie in low_vote_high_rating.iterrows():
            print(f"  '{movie['names'][:30]}...'")
            print(f"    Raw: {movie['score']:.1f} ({movie['vote_count']:.0f} votes)")
            print(f"    Weighted: {movie['weighted_rating']:.2f} (Pulled toward {self.average_rating:.1f})")
            print()
        # Select movies with high votes to show trustworthy ratings
        high_vote_movies = self.movies_df[
            self.movies_df['vote_count'] >= self.vote_threshold
        ].nlargest(3, 'vote_count')
        # Display examples of high-vote movies
        print("🟢 HIGH VOTES (Maintains Original Rating):")
        for _, movie in high_vote_movies.iterrows():
            print(f"  '{movie['names'][:30]}...'")
            print(f"    Raw: {movie['score']:.1f} ({movie['vote_count']:.0f} votes)")
            print(f"    Weighted: {movie['weighted_rating']:.2f} (Trustworthy)")
            print()

        print("🚀 Correcting Rating Bias / Trustworthy Problem Completed!\n")

    def build_content_based_system(self):
        """
        PART 1: CONTENT BASED FILTERING
        Build TF-IDF based content similarity system
        """
        # Print header for content-based filtering
        print(f"\n🎯 PART 1: CONTENT BASED FILTERING")
        print("=" * 45)
        # Use all movies (qualified_movies = movies_df)
        working_df = self.qualified_movies.copy()
        # Initialize TF-IDF vectorizer for text features
        print(f"📝 Building TF-IDF Matrix...")
        print("Features: stop_words='english', max_features=10000, ngram_range=(1,2)")
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',  # Remove common English words
            max_features=10000,  # Limit to top 5000 features
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms in fewer than 2 documents
            max_df=0.8  # Ignore terms in more than 80% of documents
        )
        # Fill missing 'enhanced_content' values
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        # Create TF-IDF matrix from enhanced content
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        print(f"✅ TF-IDF Matrix Shape: {self.tfidf_matrix.shape}")
        print(f"📚 Vocabulary size: {len(self.tfidf_vectorizer.get_feature_names_out())}")
        # Show sample features from TF-IDF vocabulary
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"📖 Sample features: {feature_names[:20]}")
        # Compute cosine similarity matrix using linear kernel
        print(f"\n🧮 Computing Content Similarity Matrix...")
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        print(f"✅ Similarity Matrix Shape: {self.cosine_sim.shape}")
        # Create mapping of movie titles to DataFrame indices
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        # Store working DataFrame
        self.qualified_movies = working_df
        print(f"\n🚀 Content-based system built successfully!")

    def get_content_recommendations(self, title, n=10, show_details=True):
            """
            Get content-based recommendations with HIGHEST SIMILARITY first
            MODIFIED: Similarity scores scaled to 0.8-1.0 range, top recommendation is highest similarity
            FIXED: Handle multiple movie matches properly with interactive selection
            """
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

    # Keep all existing functions...
    def analyze_similarity_quality(self):
            """
            FIXED: Analyze quality with improved metrics calculation
            """
            print(f"\n📊 ANALYZING IMPROVED SIMILARITY QUALITY")
            print("=" * 40)
            print("🎯 Target Metrics:")
            print("   • Genre Consistency: 0.4-0.6 (Good: > 0.3)")
            print("   • Rating Difference: <2.5 points (Good: < 2.0)")
            print("   • Content Similarity: 0.75–0.90 (Good: > 0.80)")
            print("   • Recommendation Diversity: 0.6-0.85 (Good: > 0.6)")

            # Test with popular movies
            test_movies = ['The Godfather', 'Pulp Fiction', 'The Dark Knight']
            available_test_movies = []

            for movie in test_movies:
                matches = self.qualified_movies[
                    self.qualified_movies['orig_title'].str.contains(movie, case=False, na=False)
                ]
                if not matches.empty:
                    available_test_movies.append(matches.iloc[0]['orig_title'])

            if not available_test_movies:
                # Use top-rated movies instead
                available_test_movies = self.qualified_movies.nlargest(5, 'weighted_rating')['names'].tolist()

            print(f"🎬 Testing with movies: {available_test_movies[:3]}")

            quality_metrics = {
                'genre_consistency': [],
                'rating_similarity': [],
                'diversity_scores': [],
                'content_similarity': []
            }

            for movie in available_test_movies[:3]:
                recs = self.get_content_recommendations(movie, n=10, show_details=False)

                if isinstance(recs, pd.DataFrame) and not recs.empty:
                    # Get original movie info
                    original_movie = self.qualified_movies[
                        self.qualified_movies['orig_title'].str.contains(movie, case=False, na=False)
                    ].iloc[0]

                    original_genres = set(str(original_movie['genre']).split('|'))
                    original_rating = original_movie['weighted_rating']

                    # Compute genre consistency
                    genre_sims = []
                    for _, rec in recs.iterrows():
                        candidate_genres = set(str(rec['genre']).split('|'))
                        if original_genres and candidate_genres:
                            intersection = len(original_genres.intersection(candidate_genres))
                            union = len(original_genres.union(candidate_genres))
                            jacc = intersection / union if union > 0 else 0.0
                            genre_sims.append(jacc)

                    if genre_sims:
                        quality_metrics['genre_consistency'].extend(genre_sims)

                    # Compute rating similarity
                    rating_diffs = [abs(original_rating - rec['weighted_rating']) for _, rec in recs.iterrows()]
                    quality_metrics['rating_similarity'].extend(rating_diffs)

                    # Content similarity
                    quality_metrics['content_similarity'].extend(recs['similarity'].tolist())

                    # Diversity (unique genres)
                    all_rec_genres = []
                    for _, rec in recs.iterrows():
                        all_rec_genres.extend(str(rec['genre']).split('|'))
                    diversity = len(set(all_rec_genres)) / len(all_rec_genres) if all_rec_genres else 0
                    quality_metrics['diversity_scores'].append(diversity)

            # Print improved quality metrics
            print(f"\n📈 IMPROVED QUALITY METRICS:")
            print("-" * 40)

            if quality_metrics['genre_consistency']:
                genre_mean = np.mean(quality_metrics['genre_consistency'])
                genre_std = np.std(quality_metrics['genre_consistency'])
                genre_status = "🟢 GOOD" if genre_mean >= 0.3 else "🔴 NEEDS WORK"
                print(f"Genre Consistency: {genre_mean:.3f} ± {genre_std:.3f} {genre_status}")

            if quality_metrics['rating_similarity']:
                rating_mean = np.mean(quality_metrics['rating_similarity'])
                rating_std = np.std(quality_metrics['rating_similarity'])
                rating_status = "🟢 GOOD" if rating_mean <= 2.5 else "🔴 NEEDS WORK"
                print(f"Rating Difference: {rating_mean:.3f} ± {rating_std:.3f} {rating_status}")

            if quality_metrics['content_similarity']:
                content_mean = np.mean(quality_metrics['content_similarity'])
                content_std = np.std(quality_metrics['content_similarity'])
                content_status = "🟢 GOOD" if content_mean >= 0.3 else "🔴 NEEDS WORK"
                print(f"Content Similarity: {content_mean:.3f} ± {content_std:.3f} {content_status}")

            if quality_metrics['diversity_scores']:
                diversity_mean = np.mean(quality_metrics['diversity_scores'])
                diversity_std = np.std(quality_metrics['diversity_scores'])
                diversity_status = "🟢 GOOD" if 0.6 <= diversity_mean <= 0.85 else "🟡 MODERATE"
                print(f"Recommendation Diversity: {diversity_mean:.3f} ± {diversity_std:.3f} {diversity_status}")

            return quality_metrics

    def get_similarity_level(self, similarity_score):
        """Convert similarity score to descriptive level"""
        # Assign descriptive labels based on similarity score
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

    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations and graphs"""
        print(f"\n📊 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 50)

        # Set seaborn style
        plt.style.use('seaborn-v0_8')

        # 2 rows × 4 columns = 8 subplots
        fig = plt.figure(figsize=(22, 12))

        # Subplot 1: Scatter plot of original vs weighted ratings
        plt.subplot(2, 4, 1)
        plt.scatter(self.movies_df['vote_count'], self.movies_df['score'],
                    alpha=0.6, s=30, label='Original Score', color='blue')
        plt.scatter(self.movies_df['vote_count'], self.movies_df['weighted_rating'],
                    alpha=0.6, s=30, label='Weighted Score', color='red')
        plt.axvline(self.vote_threshold, color='green', linestyle='--',
                    label=f'Threshold: {self.vote_threshold:.0f}')
        plt.xlabel('Vote Count')
        plt.ylabel('Rating')
        plt.title('Original vs Weighted Ratings')
        plt.legend(fontsize=8)
        plt.xscale('log')

        # Subplot 2: Histogram of rating distribution
        plt.subplot(2, 4, 2)
        plt.hist(self.movies_df['score'], bins=30, alpha=0.7, color='skyblue',
                label=f'Original (μ={self.movies_df["score"].mean():.2f})')
        plt.hist(self.qualified_movies['weighted_rating'], bins=30, alpha=0.7, color='orange',
                label=f'Weighted (μ={self.qualified_movies["weighted_rating"].mean():.2f})')
        plt.xlabel('Rating')
        plt.ylabel('Frequency')
        plt.title('Rating Distribution')
        plt.legend(fontsize=8)

        # Subplot 3: Bar plot of top genres
        plt.subplot(2, 4, 3)
        all_genres = []
        for genres in self.qualified_movies['genre'].dropna():
            all_genres.extend(str(genres).split('|'))
        genre_counts = pd.Series(all_genres).value_counts().head(10)
        genre_counts.plot(kind='barh', color='coral')
        plt.title('Top 10 Genres')
        plt.xlabel('Count')

        # Subplot 4: Scatter plot of rating vs release year
        plt.subplot(2, 4, 4)
        self.qualified_movies['year'] = pd.to_datetime(
            self.qualified_movies['date_x'], errors='coerce').dt.year
        valid_years = self.qualified_movies.dropna(subset=['year'])
        if not valid_years.empty:
            plt.scatter(valid_years['year'], valid_years['weighted_rating'],
                        alpha=0.6, color='purple')
            plt.xlabel('Release Year')
            plt.ylabel('Weighted Rating')
            plt.title('Rating vs Release Year')

        # Subplot 5: Scatter plot of budget vs revenue
        plt.subplot(2, 4, 5)
        if 'budget_x' in self.qualified_movies.columns and 'revenue' in self.qualified_movies.columns:
            valid_budget = self.qualified_movies[
                (self.qualified_movies['budget_x'] > 0) &
                (self.qualified_movies['revenue'] > 0)
            ]
            if not valid_budget.empty:
                plt.scatter(valid_budget['budget_x'], valid_budget['revenue'],
                            alpha=0.6, c=valid_budget['weighted_rating'], cmap='viridis')
                plt.colorbar(label='Weighted Rating')
                plt.xlabel('Budget')
                plt.ylabel('Revenue')
                plt.title('Budget vs Revenue')
                plt.xscale('log')
                plt.yscale('log')

        # Subplot 6: Bar plot of top countries
        plt.subplot(2, 4, 6)
        if 'country' in self.qualified_movies.columns:
            country_counts = self.qualified_movies['country'].value_counts().head(10)
            country_counts.plot(kind='bar', color='gold')
            plt.title('Top 10 Countries')
            plt.xticks(rotation=45, fontsize=8)
            plt.xlabel('Countries')
            plt.ylabel('Movie Count')

        # Subplot 7: Bar plot of top movies by weighted rating
        plt.subplot(2, 4, 7)
        top_movies = self.qualified_movies.nlargest(10, 'weighted_rating')
        plt.barh(range(len(top_movies)), top_movies['weighted_rating'], color='red', alpha=0.7)
        plt.yticks(range(len(top_movies)),
                  [title if len(title) <= 25 else title[:25] + '...' for title in top_movies['names']],
                  fontsize=8)
        plt.xlabel('Weighted Rating')
        plt.title('Top 10 Movies')

        # Subplot 8: Pie chart of language distribution with legend box
        plt.subplot(2, 4, 8)
        if 'orig_lang' in self.qualified_movies.columns:
            lang_counts = self.qualified_movies['orig_lang'].value_counts().head(8)

            # Draw pie without labels/percentages
            wedges, texts = plt.pie(
                lang_counts.values,
                labels=None,   # no labels inside
                startangle=90
            )

            # Build legend: Language + percentage
            legend_labels = [
                f"{lang} ({val/sum(lang_counts.values)*100:.1f}%)"
                for lang, val in zip(lang_counts.index, lang_counts.values)
            ]

            # Add legend box at the side
            plt.legend(
                wedges, legend_labels,
                title="Languages",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)  # push outside to the right
            )

            plt.title('Language Distribution')


        # Adjust layout to avoid overlapping
        plt.tight_layout()
        plt.show()

        # Extra quality analysis plots
        self.create_quality_analysis_plots()


    def create_quality_analysis_plots(self):
        """Create detailed quality analysis plots"""
        # Analyze similarity quality and get metrics
        quality_metrics = self.analyze_similarity_quality()

        # Check if metrics are available
        if not any(quality_metrics.values()):
            print("⚠️ No quality metrics available for visualization")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot genre consistency histogram
        if quality_metrics['genre_consistency']:
            axes[0, 0].hist(quality_metrics['genre_consistency'], bins=20,
                          color='lightblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Genre Consistency Distribution')
            axes[0, 0].set_xlabel('Jaccard Similarity')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].axvline(np.mean(quality_metrics['genre_consistency']),
                              color='red', linestyle='--',
                              label=f'Mean: {np.mean(quality_metrics["genre_consistency"]):.3f}')
            axes[0, 0].legend()

        # Plot rating difference histogram
        if quality_metrics['rating_similarity']:
            axes[0, 1].hist(quality_metrics['rating_similarity'], bins=20,
                          color='lightgreen', alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Rating Difference Distribution')
            axes[0, 1].set_xlabel('Rating Difference')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].axvline(np.mean(quality_metrics['rating_similarity']),
                              color='red', linestyle='--',
                              label=f'Mean: {np.mean(quality_metrics["rating_similarity"]):.3f}')
            axes[0, 1].legend()

        # Plot Content Similarity Histogram
        if quality_metrics['content_similarity']:
            axes[1, 0].hist(quality_metrics['content_similarity'], bins=20,
                            color='orange', alpha=0.7, edgecolor='black')
            axes[1, 0].set_title('Content Similarity Distribution')
            axes[1, 0].set_xlabel('Cosine Similarity')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].axvline(np.mean(quality_metrics['content_similarity']),
                              color='red', linestyle='--',
                              label=f'Mean: {np.mean(quality_metrics["content_similarity"]):.3f}')
            axes[1, 0].legend()

        # Plot diversity scores
        if quality_metrics['diversity_scores']:
            axes[1, 1].bar(range(len(quality_metrics['diversity_scores'])),
                          quality_metrics['diversity_scores'],
                          color='coral', alpha=0.7)
            axes[1, 1].set_title('Recommendation Diversity by Test Movie')
            axes[1, 1].set_xlabel('Test Movie Index')
            axes[1, 1].set_ylabel('Diversity Score')

        plt.tight_layout()
        plt.show()

    def interactive_recommendation_system(self):
        """Enhanced interactive recommendation system with multiple search options"""
        # Print header for interactive system
        print(f"\n🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
        print("=" * 65)
        print("✨ NEW FEATURES: Search by Title, Genre, Crew, or Advanced Multi-Search!")

        # Show available options
        print(f"\n🎯 SEARCH OPTIONS:")
        print("1️⃣ Search by Movie Title (Content-based recommendations)")
        print("2️⃣ Search by Genre (Top-rated movies in genre)")
        print("3️⃣ Search by Crew Member (Movies with specific actor/director)")
        print("4️⃣ Advanced Search (Combine multiple criteria)")
        print("5️⃣ Browse Available Genres")
        print("6️⃣ Browse Popular Crew Members")
        print("7️⃣ Exit")

        # Start interactive loop
        while True:
            print(f"\n" + "="*60)
            choice = input("🎬 Choose an option (1-7): ").strip()

            if choice == '7' or choice.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the enhanced recommendation system!")
                break

            elif choice == '1':
                # Search by movie title
                movie_title = input("🎬 Enter a movie title: ").strip()
                if movie_title:
                    cleaned_title = self.clean_title_text(movie_title)
                    n_recs = input("📊 Number of recommendations (default 10): ").strip()
                    n_recs = int(n_recs) if n_recs.isdigit() else 10
                    n_recs = min(max(n_recs, 1), 20)

                    result = self.get_content_recommendations(cleaned_title, n=n_recs)
                    if isinstance(result, str):
                        print(result)

            elif choice == '2':
                # Search by genre
                genre = input("🎭 Enter a genre: ").strip()
                if genre:
                    n_results = input("📊 Number of results (default 10): ").strip()
                    n_results = int(n_results) if n_results.isdigit() else 10
                    n_results = min(max(n_results, 1), 20)

                    result = self.search_by_genre(genre, n=n_results)
                    if isinstance(result, str):
                        print(result)

            elif choice == '3':
                # Search by crew
                crew_name = input("👥 Enter crew member name (actor, director, etc.): ").strip()
                if crew_name:
                    n_results = input("📊 Number of results (default 10): ").strip()
                    n_results = int(n_results) if n_results.isdigit() else 10
                    n_results = min(max(n_results, 1), 20)

                    result = self.search_by_crew(crew_name, n=n_results)
                    if isinstance(result, str):
                        print(result)

            elif choice == '4':
                # Advanced search
                print("\n🔍 Advanced Search - Enter criteria (leave empty to skip):")
                genre = input("🎭 Genre: ").strip() or None
                crew = input("👥 Crew member: ").strip() or None
                min_rating = input("⭐ Minimum rating: ").strip()
                min_rating = float(min_rating) if min_rating.replace('.','').isdigit() else None
                n_results = input("📊 Max results (default 10): ").strip()
                n_results = int(n_results) if n_results.isdigit() else 10
                n_results = min(max(n_results, 1), 20)

                result = self.advanced_search(
                    genre=genre,
                    crew=crew,
                    min_rating=min_rating,
                    max_results=n_results
                )
                if isinstance(result, str):
                    print(result)

            elif choice == '5':
                # Browse genres
                print("\n")
                self.get_genre_list()

            elif choice == '6':
                # Browse crew
                print("\n")
                self.get_popular_crew()

            else:
                print("❌ Invalid choice. Please select 1-7.")
