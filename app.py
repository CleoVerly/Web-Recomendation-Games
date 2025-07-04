# app.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from flask_cors import CORS # Untuk komunikasi antara frontend dan backend

# --- Inisialisasi Aplikasi Flask ---
app = Flask(__name__)
CORS(app) # Mengaktifkan CORS

# --- Global variables untuk menyimpan model dan data ---
tfidf_matrix = None
cosine_sim = None
df_final = None
indices = None

def prepare_model():
    """
    Fungsi untuk memuat data dan melatih model.
    Hanya dijalankan sekali saat server pertama kali hidup.
    """
    global tfidf_matrix, cosine_sim, df_final, indices

    # 1. Memuat dan mempersiapkan data
    try:
        df = pd.read_csv(r'BIGDATA\data\steam.csv')
    except FileNotFoundError:
        print("❌ ERROR: File 'data/steam.csv' tidak ditemukan.")
        exit() # Hentikan aplikasi jika file tidak ada

    df = df.dropna(subset=['name', 'genres', 'steamspy_tags'])
    df['content'] = df['genres'].fillna('') + ' ' + df['steamspy_tags'].fillna('') + ' ' + df['categories'].fillna('')
    
    # Menggunakan random sampling dataset untuk mengurangi beban memori
    #df_final = df.reset_index(drop=True) # Menggunakan seluruh dataset
    df_final = df.sample(15000, random_state=42).reset_index(drop=True) # Menggunakan 15.000 baris acak dari dataset
    print(f"Total games loaded into model: {len(df_final)}")


    # 2. Membuat model TF-IDF dan Cosine Similarity
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df_final['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 3. Membuat series untuk pencarian indeks game
    indices = pd.Series(df_final.index, index=df_final['name']).drop_duplicates()
    
    print("✅ Model and data loaded successfully!")


def get_recommendations(game_name, top_n=10):
    if game_name not in indices:
        return None # Mengembalikan None jika game tidak ditemukan

    idx = indices[game_name]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    game_indices = [i[0] for i in sim_scores]

    # Mengambil data yang relevan dan mengubahnya ke format dictionary
    result = df_final[['name', 'genres', 'price', 'developer']].iloc[game_indices]
    return result.to_dict(orient='records')

# --- API Endpoint untuk Rekomendasi ---
@app.route('/recommend', methods=['POST'])
def recommend_endpoint():
    data = request.get_json()
    if not data or 'game_name' not in data:
        return jsonify({'error': 'Nama game tidak ditemukan di request'}), 400

    game_name = data['game_name']
    
    # Cek apakah input pengguna terkandung di dalam nama game (case-insensitive)
    matching_games = [name for name in indices.index if game_name.lower() in name.lower()]
    
    if not matching_games:
         return jsonify({'error': f"Game '{game_name}' tidak ditemukan di database kami."}), 404

    # Jika ada beberapa game yang cocok, kita ambil hasil pertama yang ditemukan.
    actual_game_name = matching_games[0]
    
    recommendations = get_recommendations(actual_game_name)

    response_data = {
        'found_game': actual_game_name,
        'recommendations': recommendations
    }

    return jsonify(response_data)

# --- API Endpoint untuk Game Acak ---
@app.route('/random', methods=['GET'])
def random_game_endpoint():
    # Pilih 1 game secara acak dari dataframe
    random_game = df_final.sample(n=1)
    
    # Ambil namanya dari baris yang terpilih
    game_name = random_game['name'].iloc[0]
    
    # Kirim nama game tersebut sebagai JSON
    return jsonify({'game_name': game_name})


# --- Menjalankan Server ---
if __name__ == '__main__':
    # Mempersiapkan model saat server dimulai
    prepare_model()
    # Menjalankan aplikasi Flask
    app.run(debug=True)