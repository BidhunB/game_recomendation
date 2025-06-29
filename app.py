from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import datetime

app = Flask(__name__)

def fetch_game_data(date_filter=None):
    url = (
        f"https://api.rawg.io/api/games"
        f"?key=711187b0527641ab907077c4c278a39c"
        f"&page_size=40"
    )
    if date_filter:
        url += f"&dates={date_filter}"
    response = requests.get(url)
    if response.status_code != 200:
        print("Error fetching data:", response.status_code)
        return None
    data = response.json()
    if "results" not in data:
        print("Unexpected API format")
        return None
    df = pd.DataFrame(data["results"])

    # Handle missing values safely
    df['genres'] = df['genres'].apply(lambda x: x if isinstance(x, list) else [])
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    df['playtime'] = df['playtime'].fillna(0)
    
    # Feature engineering
    df['genre_text'] = df['genres'].apply(lambda g: ' '.join([item['name'] for item in g]))
    df['tag_text'] = df['tags'].apply(lambda t: ' '.join([item['name'] for item in t]))
    df['features'] = df['genre_text'] + ' ' + df['tag_text']

    def playtime_label(hours):
        try:
            hours = float(hours)
            if hours < 5: return "short"
            elif hours < 20: return "medium"
            else: return "long"
        except:
            return "unknown"

    df['playtime_label'] = df['playtime'].apply(playtime_label)
    df['features'] += ' ' + df['playtime_label']

    # TF-IDF
    vectorizer = TfidfVectorizer()
    df['combined'] = df['features']
    tfidf_matrix = vectorizer.fit_transform(df['combined'])

    return df, vectorizer, tfidf_matrix

# Fetch all-time games (no date filter)
df_all, vectorizer_all, tfidf_matrix_all = fetch_game_data()

# Fetch new & trending (last 6 months)
today = datetime.date.today()
six_months_ago = today - datetime.timedelta(days=180)
date_filter = f"{six_months_ago},{today}"
df_new, vectorizer_new, tfidf_matrix_new = fetch_game_data(date_filter=date_filter)

@app.route('/', methods=['GET', 'POST'])
def home():
    # All Time Trending
    trending_games_df = df_all.copy()
    trending_games_df['released'] = pd.to_datetime(trending_games_df['released'], errors='coerce')
    trending_games_df = trending_games_df.dropna(subset=['released'])
    trending_games_df = trending_games_df.sort_values(by=["rating", "released"], ascending=[False, False]).head(5)
    trending_games = trending_games_df[["name", "genre_text", "tag_text", "rating", "background_image", "released"]].to_dict(orient="records")

    # New & Trending
    new_trending_df = df_new.copy()
    new_trending_df['released'] = pd.to_datetime(new_trending_df['released'], errors='coerce')
    new_trending_df = new_trending_df.dropna(subset=['released'])
    new_trending_df = new_trending_df[new_trending_df['rating'] > 0]
    new_trending_df = new_trending_df.sort_values(by=["released", "rating"], ascending=[False, False]).head(5)
    new_trending_games = new_trending_df[["name", "genre_text", "tag_text", "rating", "background_image", "released"]].to_dict(orient="records")

    # Popular (from all-time)
    popular_games = df_all.sort_values(by="rating", ascending=False).head(5)
    popular_games = popular_games[["name", "genre_text", "tag_text", "rating", "background_image"]].to_dict(orient="records")
    result = None

    if df_all is None:
        return render_template('index.html', result=None, popular_games=[], trending_games=[])

    if request.method == 'POST':
        genres = request.form.get('genres', '')
        tags = request.form.get('tags', '')

        selected_input = " ".join([
            g.strip().title() for g in genres.split(",") if g.strip()
        ] + [
            t.strip().title() for t in tags.split(",") if t.strip()
        ])

        if selected_input:
            user_vector = vectorizer_all.transform([selected_input])
            similarity_scores = cosine_similarity(user_vector, tfidf_matrix_all)
            top_indices = similarity_scores.argsort()[0][::-1][:3]
            recommended_games_df = df_all.iloc[top_indices]
            result = recommended_games_df[["name", "genre_text", "tag_text", "rating", "background_image"]].to_dict(orient="records")
        else:
            result = []

    return render_template(
    'index.html',
    result=result,
    popular_games=popular_games,
    trending_games=trending_games,
    new_trending_games=new_trending_games
)
if __name__ == '__main__':
    app.run(debug=True)
