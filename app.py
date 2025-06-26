from flask import Flask, render_template, request
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def fetch_game_data():
    url = "https://api.rawg.io/api/games?key=711187b0527641ab907077c4c278a39c"
    response = requests.get(url)
    
    if response.status_code != 200:
        print("Error fetching data:", response.status_code)
        return None, None, None

    data = response.json()
    if "results" not in data:
        print("Unexpected API format")
        return None, None, None

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

# Fetch once on server start
df, vectorizer, tfidf_matrix = fetch_game_data()

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if df is None:
        return render_template('index.html', result=None)

    if request.method == 'POST':
        genres = request.form.get('genres', '')
        tags = request.form.get('tags', '')

        print("Genres:", genres)
        print("Tags:", tags)

        # Preprocess user input
        selected_input = " ".join([
            g.strip().title() for g in genres.split(",") if g.strip()
        ] + [
            t.strip().title() for t in tags.split(",") if t.strip()
        ])

        if selected_input:
            user_vector = vectorizer.transform([selected_input])
            similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
            top_indices = similarity_scores.argsort()[0][::-1][:3]
            recommended_games_df = df.iloc[top_indices]
            result = recommended_games_df[["name", "genre_text", "tag_text", "rating"]].to_dict(orient="records")
        else:
            result = []

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
