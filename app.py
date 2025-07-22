# app.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
CORS(app)

# --- Variabel Global ---
tfidf_matrix = None
cosine_sim = None
df_final = None
indices = None
unique_genres = None

def prepare_model():
    """
    Fungsi untuk memuat data dan melatih model.
    Hanya dijalankan sekali saat server pertama kali hidup.
    """
    global tfidf_matrix, cosine_sim, df_final, indices, unique_genres
    try:
        # Pastikan path file CSV benar. Sesuaikan jika perlu.
        # Contoh: 'data/steam.csv' atau 'C:/Users/Anda/Desktop/proyek/data/steam.csv'
        df = pd.read_csv(r'data/steam.csv')
    except FileNotFoundError:
        print("❌ ERROR: File 'data/steam.csv' tidak ditemukan. Pastikan path file sudah benar.")
        exit()

    # Membersihkan data
    df = df.dropna(subset=['name', 'genres', 'steamspy_tags'])
    
    # Menggabungkan fitur teks untuk analisis
    df['content'] = df['genres'].fillna('') + ' ' + df['steamspy_tags'].fillna('') + ' ' + df['categories'].fillna('')
    
    # Mengambil sampel data untuk efisiensi memori dan kecepatan
    df_final = df.sample(15000, random_state=42).reset_index(drop=True)
    print(f"Total games loaded into model: {len(df_final)}")

    # Membuat TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_final['content'])
    
    # Menghitung cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Membuat series untuk pencarian indeks berdasarkan nama game
    indices = pd.Series(df_final.index, index=df_final['name']).drop_duplicates()
    
    # Mengumpulkan semua genre unik
    all_genres = set()
    for s in df_final['genres'].str.split(';'):
        for genre in s:
            all_genres.add(genre.strip())
    unique_genres = sorted(list(all_genres))
    
    print("✅ Model and data loaded successfully!")

def get_recommendations(game_name, page=1, page_size=20, selected_genres=None):
    """
    Fungsi untuk mendapatkan rekomendasi berdasarkan kemiripan (cosine similarity).
    """
    if game_name not in indices:
        return None, 0

    idx = indices[game_name]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:]
    game_indices = [i[0] for i in sim_scores]
    result_df = df_final.iloc[game_indices]

    # Filter tambahan berdasarkan genre jika dipilih
    if selected_genres:
        def check_genres(genres_str):
            game_genres = set(g.strip() for g in genres_str.split(';'))
            return set(selected_genres).issubset(game_genres)
        result_df = result_df[result_df['genres'].apply(check_genres)]

    # Logika Paginasi
    total_results = len(result_df)
    start_index = (page - 1) * page_size
    end_index = start_index + page_size
    paginated_df = result_df.iloc[start_index:end_index]

    return paginated_df.to_dict(orient='records'), total_results

# --- API Endpoints ---

@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    """
    Endpoint utama yang dapat melakukan 2 hal:
    1. Memberikan rekomendasi berdasarkan nama game (jika diisi).
    2. Memberikan hasil filter berdasarkan genre (jika nama game kosong).
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request tidak valid'}), 400

    game_name = data.get('game_name', '').strip()
    selected_genres = data.get('selected_genres', [])
    page = data.get('page', 1)
    page_size = 20

    # --- ALUR 1: Jika NAMA GAME DIISI (rekomendasi similarity) ---
    if game_name:
        matching_games = [name for name in indices.index if game_name.lower() in name.lower()]
        if not matching_games:
            return jsonify({'error': f"Game '{game_name}' tidak ditemukan."}), 404

        actual_game_name = matching_games[0]
        
        recommendations, total_found = get_recommendations(
            actual_game_name,
            page=page,
            page_size=page_size,
            selected_genres=selected_genres
        )

        response_data = {
            'display_title': f"Menampilkan {total_found} rekomendasi untuk: <strong>{actual_game_name}</strong>",
            'recommendations': recommendations,
            'total_found': total_found,
            'page': page,
            'search_type': 'recommendation'
        }
        return jsonify(response_data)

    # --- ALUR 2: Jika NAMA GAME KOSONG (filter genre) ---
    else:
        if not selected_genres:
            return jsonify({'error': 'Masukkan nama game atau pilih setidaknya satu genre.'}), 400

        result_df = df_final.copy()

        # Fungsi untuk memeriksa apakah semua genre yang dipilih ada di game
        def check_genres(genres_str):
            game_genres = set(g.strip() for g in genres_str.split(';'))
            return set(selected_genres).issubset(game_genres)
        
        result_df = result_df[result_df['genres'].apply(check_genres)]

        # Paginasi
        total_results = len(result_df)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_df = result_df.iloc[start_index:end_index]

        genre_text = ', '.join(selected_genres)
        response_data = {
            'display_title': f"Menampilkan {total_results} game dengan genre: <strong>{genre_text}</strong>",
            'recommendations': paginated_df.to_dict(orient='records'),
            'total_found': total_results,
            'page': page,
            'search_type': 'filter'
        }
        return jsonify(response_data)

@app.route('/random', methods=['GET'])
def random_game_endpoint():
    """Endpoint untuk mendapatkan satu nama game acak."""
    game_name = df_final.sample(n=1)['name'].iloc[0]
    return jsonify({'game_name': game_name})

@app.route('/initial-recommendations', methods=['GET'])
def initial_recommendations_endpoint():
    """Endpoint untuk memberikan rekomendasi awal saat halaman pertama kali dimuat."""
    # Memberikan sampel yang konsisten setiap kali dimuat
    random_games = df_final.sample(n=21, random_state=101)
    return jsonify(random_games.to_dict(orient='records'))

@app.route('/genres', methods=['GET'])
def get_genres_endpoint():
    """Endpoint untuk mendapatkan daftar semua genre unik."""
    return jsonify(unique_genres)

# --- Menjalankan Server ---
if __name__ == '__main__':
    prepare_model()  # Memuat data dan model saat server dimulai
    app.run(debug=True) # Jalankan server Flask
