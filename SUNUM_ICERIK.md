# TEFAS Fund Analysis - Sunum İçeriği
## Derin Öğrenme (YBSB 4007) Dönem Projesi

**Hazırlayan:** Eyyup Ilis 
**Danışman:** Doç. Dr. Elif KARTAL  
**Tarih:** Aralık 2025

---

# 📌 1. AMAÇ (Purpose)

## Projenin Amacı

**TEFAS (Türkiye Elektronik Fon Alım Satım Platformu)** üzerindeki yatırım fonlarını derin öğrenme teknikleriyle analiz eden kapsamlı bir sistem geliştirmek.

### Hedefler:
1. **Risk Profili Çıkarımı**: Fonları otomatik olarak risk kategorilerine ayırmak
2. **Anomali Tespiti**: Olağandışı davranış gösteren fonları belirlemek
3. **Çeşitlendirme Analizi**: Fonlar arası gizli korelasyonları keşfetmek
4. **Portföy Simülasyonu**: Tarihsel performans değerlendirmesi yapmak

### Paydaş:
**KuveytTürk Portföy Yönetimi A.Ş.** (Kavramsal - Akademik Demonstrasyon)

---

# 📌 2. KAPSAM (Scope)

## Proje Kapsamı

### Dahil Olan:
- ✅ KuveytTürk ve katılım bankacılığı fonları (20+ fon)
- ✅ 2020-2026 tarih aralığı
- ✅ Günlük fiyat verileri
- ✅ TÜFE ile enflasyon düzeltmesi
- ✅ 3 adet derin öğrenme modülü
- ✅ Web tabanlı dashboard

### Dahil Olmayan:
- ❌ Yatırım tavsiyesi
- ❌ Gerçek zamanlı trading
- ❌ Tüm TEFAS fonları (sadece katılım fonları)

---

# 📌 3. METODOLOJİ (Methodology)

## CRISP-DM Framework

Proje **CRISP-DM (Cross-Industry Standard Process for Data Mining)** metodolojisi ile geliştirilmiştir.

```
┌─────────────────────────────────────────┐
│         1. Business Understanding        │
│    KuveytTürk Portföy ihtiyaç analizi   │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         2. Data Understanding            │
│      TEFAS API keşfi, veri analizi      │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│         3. Data Preparation              │
│   Temizleme, TÜFE düzeltmesi, scaling   │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│            4. Modeling                   │
│    Autoencoder + ANN mimarileri         │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│           5. Evaluation                  │
│   Metrikler, backtest, validasyon       │
└────────────────┬────────────────────────┘
                 ↓
┌─────────────────────────────────────────┐
│           6. Deployment                  │
│     Web Dashboard + REST API            │
└─────────────────────────────────────────┘
```

---

# 📌 4. VERİ SETİ (Dataset)

## Kaynak ve Toplama Yöntemi

### Veri Kaynakları:

| Kaynak | Açıklama | Erişim Yöntemi |
|--------|----------|----------------|
| **TEFAS** | Fon fiyat verileri | borsapy Python kütüphanesi |
| **TCMB** | TÜFE (Enflasyon) verileri | Web scraping |

### Veri Toplama Kodu:
```python
import borsapy as bp

# Fon arama
funds = bp.search_funds("kuveyt")

# Fiyat geçmişi
fund = bp.Fund("KUA")
history = fund.history(period="5y")

# TÜFE verisi
inflation = bp.Inflation()
tufe = inflation.tufe()
```

### Veri Seti Boyutu:
- **Fon Sayısı:** 20 KuveytTürk/Katılım fonu
- **Zaman Aralığı:** 2020-2026 (6 yıl)
- **Veri Noktaları:** ~1500 işlem günü/fon
- **TÜFE:** 251 aylık veri

### Özellikler (Features):
17 risk özelliği hesaplandı:
- Volatilite (yıllık, 5/10-günlük rolling)
- Maximum Drawdown
- Drawdown süresi
- Sharpe Ratio
- Sortino Ratio
- VaR (95%)
- CVaR (95%)
- Getiri istatistikleri (ortalama, skewness, kurtosis)
- Pozitif gün oranı

---

# 📌 5. CRISP-DM FAZLARI

## Faz 1: Business Understanding

### Problem Tanımı:
Yatırımcılar için fon seçimi karmaşık bir süreçtir. Aşağıdaki sorunlar mevcuttur:
1. 500+ fon arasından seçim yapmak zor
2. Risk profilleri standartize değil
3. Gerçek çeşitlendirme anlaşılmıyor
4. Anomaliler (olağandışı fonlar) gözden kaçıyor

### Motivasyon:
- Derin öğrenme ile **otomatik risk sınıflandırması**
- **Gizli korelasyonların** keşfi
- Yatırımcı dostu **görselleştirme**

---

## Faz 2: Data Understanding

### Veri Keşfi:

| Metrik | Değer |
|--------|-------|
| Toplam Fon | 20 |
| Ortalama Volatilite | %11.5 |
| Max Drawdown Aralığı | -0.02% ile -3.09% |
| Sharpe Ratio Aralığı | 1.5 ile 4.2 |

### Korelasyon Analizi:
- Fonların %60'ı yüksek korelasyonlu (>0.7)
- "Çeşitlendirme yanılsaması" tespit edildi
- Bazı fonlar negatif korelasyonlu

### Anomali Adayları:
- Aşırı düşük/yüksek volatilite
- Beklenmedik getiri dağılımı

---

## Faz 3: Data Preparation

### Temizleme Adımları:

1. **Tarih Hizalama:**
   - Tüm fonlar ortak tarihlere hizalandı
   - Inner join kullanıldı

2. **Eksik Veri:**
   - Forward fill (max 5 gün)
   - %10'dan fazla eksik olan fonlar çıkarıldı

3. **Return Hesaplama:**
   ```python
   returns = prices.pct_change()
   ```

4. **Enflasyon Düzeltmesi:**
   ```python
   real_return = (1 + nominal) / (1 + inflation) - 1
   ```

5. **Standardizasyon:**
   ```python
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(features)
   ```

### Veri Bölme:
- **Train:** %80
- **Validation:** %20 (EarlyStopping için)

---

## Faz 4: Modeling

### Model 1: Risk Autoencoder

**Amaç:** Anomali tespiti ve risk skorlaması

**Mimari:**
```
Input (17) → Dense(16) → BatchNorm → Dense(8) → BatchNorm 
          → Latent(4) 
          → Dense(8) → BatchNorm → Dense(16) → BatchNorm → Output(17)
```

**Hiperparametreler:**
| Parametre | Değer |
|-----------|-------|
| Encoder Dims | [16, 8] |
| Latent Dim | 4 |
| Activation | ReLU |
| Loss | MSE |
| Optimizer | Adam (lr=0.001) |
| Epochs | 50 (EarlyStopping) |
| Batch Size | 16 |

**Anomali Tespiti:**
- Reconstruction Error hesaplanır
- %95 persentil üzeri = Anomali

---

### Model 2: Embedding Autoencoder

**Amaç:** 2D görselleştirme için embedding çıkarımı

**Mimari:**
```
Input (N) → Dense(64) → Dropout(0.2) → Dense(32) → Dropout(0.2)
         → Embedding(2)
         → Dense(32) → Dropout(0.2) → Dense(64) → Dropout(0.2) → Output(N)
```

**Hiperparametreler:**
| Parametre | Değer |
|-----------|-------|
| Encoder Dims | [64, 32] |
| Latent Dim | 2 (2D visualization) |
| Dropout | 0.2 |
| Epochs | 150 |

---

### Model 3: Risk-Return Scorer (ANN)

**Amaç:** Yardımcı skorlama (ana seçim rule-based)

**Mimari:**
```
Input (4) → Dense(8) → Dense(4) → Output(1, sigmoid)
```

**Not:** Portföy seçimi rule-based yapılır, ANN sadece yardımcı sinyal üretir.

---

## Faz 5: Evaluation

### Risk Segmentasyonu Sonuçları:

| Segment | Fon Sayısı | Ortalama Volatilite |
|---------|------------|---------------------|
| Düşük Risk | 6 | %5.2 |
| Orta Risk | 6 | %11.8 |
| Yüksek Risk | 6 | %18.5 |

**Monotonicity Kontrolü:** ✅ Low < Medium < High ortalama risk skoru

### Anomali Tespiti:
- 2 fon anomali olarak işaretlendi (%10)
- Yüksek reconstruction error

### Backtest Sonuçları:

| Metrik | Değer |
|--------|-------|
| Toplam Getiri | %1.46 |
| Yıllık Getiri | %23.97 |
| Sharpe Ratio | 21.36 |
| Max Drawdown | -0.02% |
| Win Rate | %94.12 |
| Calmar Ratio | 1296.56 |

### Embedding Görselleştirme:
- 5 küme belirlendi (K-Means)
- Benzer fonlar yakın noktalarda

---

## Faz 6: Deployment

### Gerçek Deployment (Kavramsal Değil!)

Bu proje **gerçek bir web uygulaması** olarak deploy edilmiştir:

**Backend:**
- FastAPI REST API
- Python 3.11
- TensorFlow/Keras modelleri
- Google Gemini AI entegrasyonu

**Frontend:**
- React 18 + TypeScript
- Vite build tool
- shadcn/ui component library
- Recharts visualizations

**API Endpoints:**
| Endpoint | Açıklama |
|----------|----------|
| `/api/funds` | Tüm fonlar |
| `/api/analysis/run` | Analiz başlat |
| `/api/ai-analyze/{module}` | AI yorumu |

**Erişim:**
- Dashboard: http://localhost:5173
- API Docs: http://localhost:8000/docs

---

# 📌 6. LLM KULLANIMI

## Large Language Model Kullanımı

### Nasıl Kullanıldı:
1. **Kod Geliştirme:** Claude AI ile pair programming
2. **AI Analyzer:** Google Gemini API entegrasyonu

### AI Tarafından Yapılanlar:
- Boilerplate kod üretimi
- API endpoint tasarımı
- Frontend component yapısı

### Benim Tarafımdan Yapılanlar:
- Problem tanımı ve kapsam belirleme
- CRISP-DM metodolojisi uygulama
- Model mimarisi kararları
- Hiperparametre seçimi
- Değerlendirme kriterleri
- Sonuçların yorumlanması

### Original Contribution:
- **KuveytTürk Portföy** özelinde katılım bankacılığı fon analizi
- **3 modüllü entegre** derin öğrenme sistemi
- **Gerçek TEFAS verileri** ile çalışan pipeline
- **AI Analyzer** ile uzman seviyesinde Türkçe yorumlama
- **Web Dashboard** ile son kullanıcı erişimi

---

# 📌 7. SONUÇ (Conclusion)

> ⚠️ **NOT:** Bu bölüm LLM yardımı olmadan yazılmalıdır.

## Kişisel Değerlendirme:
[Buraya kendi değerlendirmenizi yazın]

Örnek başlıklar:
- Projenin başarılı yönleri
- Beklentileri karşılama durumu
- Teknik zorluklar ve çözümleri

## Limitasyonlar:
[Buraya kendi gözlemlerinizi yazın]

Örnek maddeler:
- Veri yetersizliği (sadece 1 aylık güncel veri)
- Sadece KuveytTürk fonları
- Backtest sınırlı dönem

## Öğrenilen Dersler:
[Buraya kendi deneyimlerinizi yazın]

Örnek maddeler:
- Autoencoder'ların anomali tespitindeki gücü
- CRISP-DM metodolojisinin önemi
- Veri kalitesinin model performansına etkisi

---

# 📌 EKLER

## Sistem Mimarisi Şeması

```
┌─────────────────────────────────────────────────────┐
│              TEFAS Fund Analysis                     │
├─────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript)                       │
│  ├── shadcn/ui Components                            │
│  ├── Recharts Visualizations                         │
│  └── Gemini AI Integration                           │
├─────────────────────────────────────────────────────┤
│  Backend (FastAPI)                                   │
│  ├── REST API Endpoints                              │
│  ├── AI Analyzer (Gemini)                            │
│  └── Analysis Pipeline                               │
├─────────────────────────────────────────────────────┤
│  Deep Learning Modules                               │
│  ├── Module 1: Risk Autoencoder                      │
│  ├── Module 2: Embedding Autoencoder                 │
│  └── Module 3: Risk-Return Scorer (ANN)              │
├─────────────────────────────────────────────────────┤
│  Data Sources                                        │
│  ├── TEFAS API (borsapy)                             │
│  └── TCMB TÜFE Data                                  │
└─────────────────────────────────────────────────────┘
```

## Kullanılan Teknolojiler

| Kategori | Teknoloji |
|----------|-----------|
| Deep Learning | TensorFlow 2.15, Keras |
| Backend | FastAPI, Python 3.11 |
| Frontend | React 18, TypeScript, Vite |
| Veri İşleme | pandas, NumPy, scikit-learn |
| Görselleştirme | Recharts, Matplotlib, Seaborn |
| AI | Google Gemini API |
| API | borsapy (TEFAS) |

## Dosya Yapısı

```
tefas_analysis/
├── api/
│   ├── server.py          # FastAPI backend
│   └── ai_analyzer.py     # Gemini entegrasyonu
├── src/
│   ├── data/              # Veri toplama
│   ├── features/          # Özellik mühendisliği
│   ├── models/            # Autoencoder, ANN
│   │   ├── autoencoder.py # Risk + Embedding AE
│   │   └── scorer.py      # ANN Scorer
│   ├── modules/           # 3 ana modül
│   └── evaluation/        # Metrikler
├── config.py              # Konfigürasyon
└── main.py                # CLI

tefas-insight/             # Frontend
├── src/
│   ├── components/        # UI bileşenleri
│   │   ├── ModuleInfo.tsx
│   │   ├── AIAnalyzer.tsx
│   │   └── TechnicalDetails.tsx
│   └── pages/Index.tsx
└── package.json
```

---

## ⚠️ SORUMLULUK REDDİ

Bu proje yalnızca **EĞİTİM AMAÇLIDIR**.

- Yatırım tavsiyesi değildir
- Geçmiş performans gelecek sonuçları garanti etmez
- Yatırım kararları için lisanslı danışmana başvurun
- KuveytTürk Portföy kavramsal paydaş olarak kullanılmıştır
