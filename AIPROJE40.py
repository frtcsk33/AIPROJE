from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from ast import literal_eval

# Filmlerin metadata verilerini yükleme
film_verileri = pd.read_csv('C:/Users/frtcs/Desktop/archive/DataSets/movies_metadata.csv', low_memory=False)

# Credits ve keywords veri setlerini yükleme
ekip_verileri = pd.read_csv('C:/Users/frtcs/Desktop/archive/DataSets/credits.csv')
anahtar_kelime_verileri = pd.read_csv('C:/Users/frtcs/Desktop/archive/DataSets/keywords.csv')

# Hatalı satırları kaldırma ve id türlerini eşleştirme
film_verileri = film_verileri[film_verileri['id'].apply(lambda x: str(x).isdigit())]  # Yalnızca sayısal değerleri tut
film_verileri['id'] = film_verileri['id'].astype('int')
ekip_verileri['id'] = ekip_verileri['id'].astype('int')

# Film verileri ile ekip ve anahtar kelime verilerini birleştirme
film_verileri = film_verileri.merge(ekip_verileri, on='id')
film_verileri = film_verileri.merge(anahtar_kelime_verileri, on='id')

# En az oy sayısını filtreleme
oy_esik_degeri = film_verileri['vote_count'].quantile(0.90)
filtrelenmis_filmler = film_verileri[film_verileri['vote_count'] >= oy_esik_degeri].copy()

# Ekip, tür ve anahtar kelime özelliklerini parse etme
ozellikler = ['cast', 'crew', 'keywords', 'genres']
for ozellik in ozellikler:
    filtrelenmis_filmler[ozellik] = filtrelenmis_filmler[ozellik].apply(literal_eval)

# Yönetmen adını çıkarma fonksiyonu
def yonetmeni_bul(veri):
    for ekip_uyesi in veri:
        if ekip_uyesi['job'] == 'Director':
            return ekip_uyesi['name']
    return np.nan

# İlk üç oyuncuyu, türleri ve anahtar kelimeleri alma fonksiyonu
def liste_olustur(veri):
    if isinstance(veri, list):
        return [eleman['name'] for eleman in veri[:3]] if len(veri) > 3 else [eleman['name'] for eleman in veri]
    return []

# Yönetmen, oyuncu, tür ve anahtar kelime bilgilerini temizleme
filtrelenmis_filmler['yonetmen'] = filtrelenmis_filmler['crew'].apply(yonetmeni_bul)
ozellikler = ['cast', 'keywords', 'genres']
for ozellik in ozellikler:
    filtrelenmis_filmler[ozellik] = filtrelenmis_filmler[ozellik].apply(liste_olustur)

# Tüm stringleri küçük harfe çevirip boşlukları kaldırma
def temizle(veri):
    if isinstance(veri, list):
        return [str.lower(eleman.replace(" ", "")) for eleman in veri]
    elif isinstance(veri, str):
        return str.lower(veri.replace(" ", ""))
    return ''

for ozellik in ['cast', 'keywords', 'yonetmen', 'genres']:
    filtrelenmis_filmler[ozellik] = filtrelenmis_filmler[ozellik].apply(temizle)

# Birleştirilmiş özelliklerin oluşturulması
def birlestir(veri):
    return ' '.join(veri['keywords']) + ' ' + ' '.join(veri['cast']) + ' ' + veri['yonetmen'] + ' ' + ' '.join(veri['genres'])

filtrelenmis_filmler['birlesik_veri'] = filtrelenmis_filmler.apply(birlestir, axis=1)

# TF-IDF işlemi
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrisi = tfidf.fit_transform(filtrelenmis_filmler['birlesik_veri'])

# KMeans modeli eğitimi
kume_sayisi = 10  # Küme sayısı
kmeans = KMeans(n_clusters=kume_sayisi, random_state=42)
kmeans.fit(tfidf_matrisi)  # Model eğitimi

# Filmleri kümelere atama
filtrelenmis_filmler['kume'] = kmeans.labels_

# Film indeksleri ve başlıkları ters haritalama
filtrelenmis_filmler = filtrelenmis_filmler.reset_index()
indeksler = pd.Series(filtrelenmis_filmler.index, index=filtrelenmis_filmler['title']).drop_duplicates()

# Film öneri fonksiyonu
def onerileri_getir(film_adi, veri=filtrelenmis_filmler, onerilen_sayi=10):
    if film_adi not in indeksler:
        return "Film bulunamadı."

    film_indeksi = indeksler[film_adi]
    film_kumesi = veri.loc[film_indeksi, 'kume']  # Filmin bulunduğu küme
    ayni_kumedeki_filmler = veri[veri['kume'] == film_kumesi]  # Aynı kümedeki diğer filmler

    # Kümedeki filmleri TF-IDF benzerliğine göre sıralama
    kume_indeksleri = ayni_kumedeki_filmler.index
    kume_tfidf_matrisi = tfidf_matrisi[kume_indeksleri]
    kosin_benzerligi = cosine_similarity(kume_tfidf_matrisi, kume_tfidf_matrisi)
    benzerlik_skorlari = list(enumerate(kosin_benzerligi[kume_indeksleri.get_loc(film_indeksi)]))
    benzerlik_skorlari = sorted(benzerlik_skorlari, key=lambda x: x[1], reverse=True)
    benzerlik_skorlari = benzerlik_skorlari[1:onerilen_sayi + 1]

    # En benzer filmleri döndürme
    benzer_film_indeksleri = [ayni_kumedeki_filmler.iloc[skor[0]].name for skor in benzerlik_skorlari]
    return filtrelenmis_filmler['title'].iloc[benzer_film_indeksleri]

# Öneri sistemini test etme
while True:
    film = input("Filmin Adı Nedir? (Programdan çıkmak için 'Bitir' yazınız): ")
    if film.lower() == 'bitir':
        break
    oneriler = onerileri_getir(film)
    if isinstance(oneriler, str):
        print(oneriler)
    else:
        print(f"\nBenzer Filmler: {', '.join(oneriler)}\n")
