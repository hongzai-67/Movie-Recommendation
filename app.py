import streamlit as st
import pandas as pd
from code import IMDBContentBasedRecommendationSystem  # 直接引入你的类

# 初始化推荐系统
@st.cache_resource
def load_system():
    system = IMDBContentBasedRecommendationSystem()
    system.load_imdb_data("imdb_movies.csv")   # 确保csv在同目录
    system.calculate_weighted_ratings()
    system.build_content_based_system()
    return system

system = load_system()

# ---------------- 页面标题 ----------------
st.title("🎮 ENHANCED INTERACTIVE MOVIE RECOMMENDATION SYSTEM")
st.markdown("=" * 65)
st.subheader("✨ NEW FEATURES: Search by Title, Genre, Crew, or Advanced Multi-Search!")

# ---------------- 菜单选项 ----------------
menu = [
    "1️⃣ Search by Movie Title (Content-based recommendations)",
    "2️⃣ Search by Genre (Top-rated movies in genre)",
    "3️⃣ Search by Crew Member (Movies with specific actor/director)",
    "4️⃣ Advanced Search (Combine multiple criteria)",
    "5️⃣ Browse Available Genres",
    "6️⃣ Browse Popular Crew Members"
]

choice = st.selectbox("🎯 SEARCH OPTIONS:", menu)

# ---------------- 功能对应 ----------------
if "1️⃣" in choice:
    title = st.text_input("🎬 Enter a movie title:")
    n = st.number_input("📊 Number of recommendations", 1, 20, 10)
    if st.button("Search"):
        result = system.get_content_recommendations(system.clean_title_text(title), n=n)
        if isinstance(result, pd.DataFrame):
            st.markdown(f"🔥 **TOP {n} RECOMMENDATIONS (SORTED BY HIGHEST SIMILARITY):**")
            for i, row in result.iterrows():
                similarity_percent = row['similarity'] * 100
                similarity_level = system.get_similarity_level(row['similarity'])
                if i == result.index[0]:
                    st.markdown(f"🏆 {i+1}. {row['names']} ⭐ TOP MATCH!")
                else:
                    st.markdown(f"{i+1}. {row['names']}")
                st.write(f"🎯 Similarity: {row['similarity']:.4f} ({similarity_percent:.1f}%) - {similarity_level}")
                st.write(f"⭐ Rating: {row['weighted_rating']:.2f}")
                st.write(f"🎭 Genre: {row['genre']}")
                st.write("")

elif "2️⃣" in choice:
    genre = st.text_input("🎭 Enter a genre:")
    n = st.number_input("📊 Number of results", 1, 20, 10)
    if st.button("Search"):
        result = system.search_by_genre(genre, n=n)
        st.dataframe(result)

elif "3️⃣" in choice:
    crew = st.text_input("👥 Enter crew member (actor/director):")
    n = st.number_input("📊 Number of results", 1, 20, 10)
    if st.button("Search"):
        result = system.search_by_crew(crew, n=n)
        st.dataframe(result)

elif "4️⃣" in choice:
    st.write("🔍 Advanced Search - Enter criteria:")
    genre = st.text_input("🎭 Genre:")
    crew = st.text_input("👥 Crew member:")
    min_rating = st.number_input("⭐ Minimum rating", 0.0, 10.0, 0.0)
    n = st.number_input("📊 Max results", 1, 20, 10)
    if st.button("Search"):
        result = system.advanced_search(
            genre=genre or None,
            crew=crew or None,
            min_rating=min_rating if min_rating > 0 else None,
            max_results=n
        )
        st.dataframe(result)

elif "5️⃣" in choice:
    genres = system.get_genre_list()
    st.write(genres)

elif "6️⃣" in choice:
    crew_list = system.get_popular_crew()
    st.write(crew_list)
