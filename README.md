<div align="center">

# 🎓 Öğrenci Terk Riski Erken Uyarı Sistemi
*(Student Dropout Risk Early Warning System)*

**Samsung Innovation Campus & UNDP (Birleşmiş Milletler Kalkınma Programı)** destekli program kapsamında geliştirilmiş, eğitimde fırsat eşitliğini ve öğrenci tutundurma (retention) oranlarını artırmayı hedefleyen veri odaklı proaktif bir asistan.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Akademik danışmanlar ve öğrenci başarı merkezleri için geliştirilmiş, mikroservis mimarisine sahip makine öğrenmesi uygulaması.*

</div>

---

## 📖 Proje Hakkında

Üniversite yönetimleri ve akademik danışmanlar, binlerce öğrencinin süreçlerini takip etmekte zorlanmaktadır. Günümüzde bir öğrencinin okulu bırakma (dropout) riski taşıdığı, genellikle öğrenci kaydını sildirdiğinde veya devamsızlıktan kaldığında fark edilir. 

Bu proje, öğrencilerin akademik geçmişlerini, demografik özelliklerini ve **finansal/barınma durumlarını** harmanlayarak, öğrenci henüz okulu bırakmadan önce **erken uyarı sinyalleri** üretmeyi amaçlamaktadır. Yalnızca bir yüzde tahmini sunmakla kalmaz; danışmanlara aksiyon alabilecekleri somut içgörüler sağlar.

> **Örnek Danışman Uyarısı:**  
> *"Kirada barınan ve bütçe açığı olan öğrenci: barınma maliyeti terk riskini artıran bir stres faktörüdür; KYK yurdu/öğrenci evi alternatifleri görüşülmeli."*

---

## ✨ Öne Çıkan Özellikler

- 🧠 **Proaktif Tahminleme:** UCI'nin *Predict Students' Dropout and Academic Success* veri seti ile eğitilmiş Random Forest modeli, öğrencileri `Terk (Dropout)`, `Kayıtlı (Enrolled)` veya `Mezun (Graduate)` olma olasılıklarına göre sınıflandırır.
- 💡 **Gerçek Dünya Bağlamı:** Modelin ötesinde, Türkiye şartlarına özel (KYK Yurdu, Bütçe Açığı, Kısmi Zamanlı Çalışma, Vakıf Üniversitesi Bursları) kural tabanlı algoritmalarla **Danışman Uyarıları** üretir.
- ⚙️ **Mikroservis Mimarisi:** Gelecekteki kurumsal OBS (Öğrenci Bilgi Sistemi) entegrasyonları düşünülerek Backend (FastAPI) ve Frontend (Streamlit) tamamen birbirinden izole edilmiştir.
- 🔒 **KVKK Uyumluluğu:** Sistem verileri diske yazmaz veya payload loglamaz. Tahminler TC/Öğrenci No yerine takma adlar (pseudonymous) kullanılarak işlenir.

---

## 📸 Ekran Görüntüleri

| Öğrenci Analiz Formu | Danışman Uyarıları Çıktısı |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/aa48ed13-4b0c-47e5-a1d5-9c82a5f5d2bb" width="400"> | <img src="https://github.com/user-attachments/assets/2c95bd53-d29f-4831-97aa-f3b923f8215d" width="400"> |

---

## 🛠️ Teknoloji Yığını (Tech Stack)

| Katman | Teknoloji | Açıklama |
| :--- | :--- | :--- |
| **Backend (API)** | `FastAPI`, `Uvicorn`, `Pydantic` | Saniyeler içinde yanıt veren, asenkron ve dökümante edilmiş REST API. **(Render üzerinde canlıda)** |
| **Machine Learning** | `Scikit-learn`, `Pandas`, `Numpy` | Random Forest Sınıflandırıcısı ve Veri Ön İşleme (Scaler). |
| **Frontend (UI)** | `Streamlit`, `Requests` | Akademik danışmanlar için tasarlanmış hızlı ve kullanıcı dostu arayüz. **(Streamlit Cloud üzerinde canlıda)** |

---

## 🚀 Canlı Demo

Uygulamayı hemen test etmek için aşağıdaki bağlantıları kullanabilirsiniz:

- **Web Arayüzü (Danışman Paneli):** [👉 Canlı Uygulama](https://student-dropout-project-bmuzwbubuyjxrchaow8ive.streamlit.app/)
- **API Dokümantasyonu (Swagger UI):** [👉 API Linki](https://student-api-2upr.onrender.com/docs)

*(Not: API ücretsiz sunucularda barındırıldığı için ilk uyandırma işlemi 30-50 saniye sürebilir.)*

---

## 💻 Kurulum ve Lokal Çalıştırma

Projeyi kendi bilgisayarınızda çalıştırmak isterseniz aşağıdaki adımları izleyebilirsiniz.

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/KULLANICI_ADINIZ/siyarkafurogullari/Student-Dropout-Project](https://github.com/siyarkafurogullari/Student-Dropout-Project)
cd Student-Dropout-Project
