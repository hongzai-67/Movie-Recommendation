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
        self.C = 0
        self.m = 0

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
        if 'vote_count' not in self.movies_df.columns:
            self.movies_df['vote_count'] = np.random.randint(50, 1000, size=len(self.movies_df))  # 如果缺少vote_count就随机填充

        # 保留原始标题
        self.movies_df['original_title'] = self.movies_df['orig_title'].copy()
        self.movies_df['orig_title'] = self.movies_df['orig_title'].apply(self.clean_title_text)

        # 去重
        self.movies_df = self.movies_df.drop_duplicates(subset=['orig_title']).reset_index(drop=True)

        # 加强特征：title + overview + genre + crew + keywords + tagline
        self.movies_df['enhanced_content'] = (
            self.movies_df['names'].astype(str) + ' ' +
            self.movies_df['overview'].astype(str) + ' ' +
            self.movies_df['genre'].astype(str).str.replace('|', ' ') + ' ' +
            self.movies_df['crew'].astype(str) + ' ' +
            self.movies_df['keywords'].astype(str) + ' ' +
            self.movies_df['tagline'].astype(str)
        )

        # Weighted Rating
        self.C = self.movies_df['score'].mean()
        self.m = self.movies_df['vote_count'].quantile(0.70)  # 取前70%分位
        self.movies_df['weighted'] = self.movies_df.apply(
            lambda x: (x['vote_count']/(x['vote_count']+self.m) * x['score']) +
                      (self.m/(self.m+x['vote_count']) * self.C),
            axis=1
        )

        self.qualified_movies = self.movies_df.copy()

    def build_content_based_system(self):
        working_df = self.qualified_movies.copy()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1,2))
        working_df['enhanced_content'] = working_df['enhanced_content'].fillna('')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(working_df['enhanced_content'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(working_df.index, index=working_df['orig_title']).drop_duplicates()
        self.qualified_movies = working_df

    def get_similarity_level(self, score):
        if score >= 0.95:
            return "🔥 VERY HIGH"
        elif score >= 0.85:
            return "🟢 HIGH"
        elif score >= 0.75:
            return "🟡 MODERATE"
        elif score >= 0.65:
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

        # 按相似度排序
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        movies = self.qualified_movies.iloc[movie_indices].copy()
        movies['similarity'] = [i[1] for i in sim_scores]

        # 按 similarity + weighted 排序
        movies = movies.sort_values(by=["similarity", "weighted"], ascending=False)
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
            output.append(f"⭐ Score: {movie_info['score']:.2f} → Weighted: {movie_info['weighted']:.2f}")
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
                output.append(f"    ⭐ Rating: {rec['score']:.2f} → Weighted: {rec['weighted']:.2f}")
                output.append(f"    🎭 Genre: {rec['genre']}\n")

            st.code("\n".join(output), language="text")


if __name__ == "__main__":
    main()
