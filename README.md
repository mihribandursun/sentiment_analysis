# 🍽️ Restaurant Reviews Sentiment Analysis System

Bu proje, restoran kullanıcı yorumları üzerinde **duygu analizi (sentiment analysis)** yaparak yorumların **pozitif veya negatif** olduğunu tahmin eden, **uçtan uca çalışan (end-to-end)** bir web tabanlı sistemdir.

Projede klasik makine öğrenmesi modelleri, derin öğrenme yaklaşımları ve **BERT tabanlı transformer modeli** karşılaştırmalı olarak değerlendirilmiş; en iyi performansı veren model web arayüzüne entegre edilmiştir.

---

## 📌 Proje Kapsamı

- Restoran kullanıcı yorumlarının duygu analizi
- Birden fazla modelin karşılaştırılması (SVM, Random Forest, LSTM, BERT)
- BERT modeli ile yüksek doğruluklu metin sınıflandırması
- Flask backend + HTML tabanlı web arayüzü
- PostgreSQL veritabanı entegrasyonu
- Kullanıcı kayıt, giriş ve yorum yönetimi

---

## 🧠 Kullanılan Modeller

| Model | Açıklama |
|------|---------|
| SVM | TF-IDF tabanlı klasik ML yaklaşımı |
| Random Forest | Kelime frekanslarına dayalı ensemble yöntem |
| LSTM | Sıralı metin yapısını öğrenebilen derin öğrenme modeli |
| **BERT (Seçilen Model)** | Bağlamsal dil temsili sağlayan transformer modeli |

> 📌 Nihai model olarak **BERT** seçilmiştir.

---

## 🗂️ Proje Klasör Yapısı

project/
│
├── app/                                              → Flask backend ve web uygulaması
│ ├── app.py                                          → Ana Flask uygulaması
│ ├── templates/                                      → HTML arayüz dosyaları
│ │ ├── register.html                                 → Kullanıcı kayıt sayfası
│ │ ├── login.html                                    → Kullanıcı giriş sayfası
│ │ ├── index.html                                    → Restoran listeleme sayfası
│ │ ├── restaurant_detail.html                        → Restoran detay ve yorum sayfası
│ │ └── my_reviews.html                               → Kullanıcıya ait yorumlar
│ └── static/
│   └── style.css                                     → Arayüz stil dosyası
│
├── data/
│ ├── raw/                                            → Ham toplanan veriler
│ │ ├── links                                         → Restoran bağlantıları
│ │ │ ├── besiktas_links
│ │ │ ├── beyoglu_links
│ │ │ ├── maltepe_links
│ │ │ └── restaurant_links.csv
│ │ └── reviews                                       → Ham kullanıcı yorumları
│ │   ├── comments_from_low_to_high_ratings.csv       
│ │   ├── new_to_old_comments.csv
│ │   ├── restaurants_reviews_raw.csv
│ │   └── control.ipynb
│ ├── processed/                                      → Ön işleme uygulanmış veri
│ │   └── restaurant_reviews_processed.csv    
│ └── final/                                          → Model ve veritabanı için son veri setleri
│     ├── restaurant_reviews_lstm.csv
│     ├── restaurant_reviews_ml.csv                      
│     ├── restaurants.csv
│     ├── reviews.csv
│     └── dataset_for_db.ipynb
│
├── db/
│ └── load_csv_to_db.py                               → CSV verilerini veritabanına aktarma scripti
│
├── eda_graphs/                                       → EDA sırasında üretilen grafikler
│ ├── raw/
│ └── processed/
│
├── models/                                           → Eğitilmiş BERT modeli ve tokenizer dosyaları
│ └── best_bert_model/
│    ├── config.json
│    ├── model.safetensors
│    ├── special_tokens_map.json
│    ├── tokenizer_config.json
│    └── vocab.txt
│
├── notebooks/                                        → Veri toplama, EDA ve model deneme notebookları
│ ├── 01_data_collection.ipynb
│ ├── 02_eda.ipynb
│ ├── 03_model_tests.ipynb
│ ├── 04_bert_experiments.ipynb
│ └── category_numbers.txt
│
├── outputs/                                          → Model çıktıları ve değerlendirme sonuçları
│
├── requirements.txt                                  → Proje bağımlılıkları
└── README.md                                         → Proje açıklama ve kullanım dokümantasyonu

## 🗄️ Veri tabanı Tasarımı

Projede **PostgreSQL** kullanılmıştır. Veriler, işlevlerine göre ayrıştırılarak normalize edilmiştir.

### Kullanılan Tablolar

- **users**  
  Kullanıcı kayıt ve giriş bilgileri

- **restaurants**  
  Restoranlara ait statik bilgiler (isim, ilçe, mutfak türü vb.)

- **reviews**  
  Ham restoran yorumları ve skor bilgileri

- **user_reviews**  
  Kullanıcı tarafından eklenen yorumlar ve model tahminleri

#### `user_reviews` Tablosu (Özet)

- review_text
- sentiment_label (Pozitif / Negatif)
- sentiment (0 / 1)
- confidence (model güven skoru)
- created_at

---

## ⚙️ Backend (Flask)

Backend tarafında **Flask** framework’ü kullanılmıştır.

### Backend Sorumlulukları

- Kullanıcı kayıt & giriş işlemleri
- Restoran ve yorum verilerinin yönetimi
- Kullanıcı yorumlarının BERT modeline gönderilmesi
- Model tahminlerinin veritabanına kaydedilmesi
- HTML arayüzlere veri aktarımı

---

## 🖥️ Arayüz (HTML)

Web arayüzü sade ve kullanıcı dostu olacak şekilde tasarlanmıştır.

### Sayfalar

| Sayfa | Açıklama |
|-----|--------|
| register.html | Kullanıcı kayıt |
| login.html | Kullanıcı giriş |
| index.html | Restoran listesi |
| restaurant_detail.html | Restoran detay & yorum ekleme |
| my_reviews.html | Kullanıcının kendi yorumları |

---

## 📊 Model Değerlendirme

Modeller aşağıdaki metriklerle değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

> BERT modeli, özellikle **F1-score ve genellenebilirlik** açısından diğer modellere göre üstün performans göstermiştir.

---

## 🎯 Neden BERT?

- Kelimeleri **bağlamsal** olarak temsil edebilmesi
- Türkçe için önceden eğitilmiş güçlü modellerin bulunması
- Uzun ve kısa metinlerde anlamsal ilişkileri yakalayabilmesi
- Gerçek dünya kullanıcı yorumlarında yüksek başarı

Yüksek hesaplama maliyetine rağmen sunduğu performans artışı nedeniyle **nihai model olarak seçilmiştir**.

---

## 🚀 Kurulum

```bash
pip install -r requirements.txt