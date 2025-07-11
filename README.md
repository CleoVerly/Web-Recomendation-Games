# Web-Recomendation-Games
Tentu, ini adalah draf lengkap untuk file `README.md` proyek Anda.

File ini menjelaskan semua aspek penting dari proyek, mulai dari deskripsi, fitur, cara instalasi, hingga teknologi yang digunakan. Anda bisa membuat file baru bernama `README.md` di folder proyek Anda dan menyalin seluruh teks di bawah ini ke dalamnya.

-----

# GameRecs - Sistem Rekomendasi Game 🎮

GameRecs adalah sebuah aplikasi web sederhana yang dirancang untuk memberikan rekomendasi game dari platform Steam. Proyek ini menggunakan metode *Content-Based Filtering* untuk menemukan game-game yang mirip berdasarkan genre, tag, dan kategorinya. Pengguna dapat memasukkan nama game favorit mereka (bahkan dengan nama yang tidak lengkap) dan mendapatkan daftar 10 game lain yang mungkin mereka sukai.

 ---

## \#\# Fitur Utama ✨

  * **Rekomendasi Cerdas**: Menggunakan TF-IDF dan Cosine Similarity untuk menganalisis dan merekomendasikan game berdasarkan kemiripan konten.
  * **Pencarian Fleksibel**: Pengguna dapat mencari game meskipun hanya mengetik sebagian dari namanya (contoh: "witcher" akan menemukan "The Witcher 3: Wild Hunt").
  * **Fitur Game Acak**: Terdapat tombol "Acak" untuk pengguna yang ingin mencoba game baru secara random sebagai titik awal rekomendasi.
  * **Antarmuka Modern**: Dibangun dengan Tailwind CSS dan DaisyUI untuk tampilan yang bersih, responsif, dan menarik.
  * **API Sederhana**: Backend Flask menyediakan API yang ringan dan mudah dipahami.

-----

## \#\# Teknologi yang Digunakan 🛠️

  * **Backend**:
      * Python 3
      * Flask (untuk web server dan API)
      * Pandas (untuk manipulasi data)
      * Scikit-learn (untuk model TF-IDF dan Cosine Similarity)
  * **Frontend**:
      * HTML5
      * Tailwind CSS
      * DaisyUI
  * **Dataset**:
      * [Steam Store Games (Kaggle)](https://www.kaggle.com/datasets/nikdavis/steam-store-games)

-----

## \#\# Instalasi dan Cara Menjalankan 🚀

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer Anda.

### \#\#\# 1. Prasyarat

Pastikan Anda sudah menginstal **Python 3.8** atau versi yang lebih baru.

### \#\#\# 2. Siapkan Proyek

```bash
# 1. Clone repositori ini (atau cukup unduh file dalam bentuk ZIP)
# git clone https://github.com/username/nama-proyek.git

# 2. Masuk ke direktori proyek
# cd nama-proyek

# 3. Buat folder 'data' dan letakkan file 'steam.csv' di dalamnya
mkdir data
# Pindahkan steam.csv ke dalam folder data/
```

### \#\#\# 3. Instalasi Dependensi

Disarankan untuk menggunakan virtual environment. Buat file bernama `requirements.txt` dan isi dengan teks di bawah ini, lalu jalankan perintah instalasi.

**File `requirements.txt`:**

```
Flask
pandas
scikit-learn
Flask-Cors
```

**Perintah instalasi:**

```bash
pip install -r requirements.txt
```

### \#\#\# 4. Jalankan Aplikasi

Setelah semua dependensi terinstal, jalankan server backend terlebih dahulu.

```bash
# Jalankan server Flask
python app.py
```

Terminal akan menampilkan pesan bahwa server berjalan di `http://127.0.0.1:5000`. Biarkan terminal ini tetap terbuka.

### \#\#\# 5. Buka Frontend

Buka File Explorer Anda, cari file `index.html` di dalam folder proyek, dan **klik dua kali** untuk membukanya di browser. Sekarang aplikasi siap digunakan\!

-----

## \#\# Struktur Proyek 📂

```
/nama-proyek/
├── data/
│   └── steam.csv
├── app.py
├── index.html
└── README.md
```

-----

## \#\# Lisensi 📄

Proyek ini dilisensikan di bawah Lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.