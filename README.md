# TEFAS Fund Analysis 📊

**KuveytTürk Portföy için Deep Learning Tabanlı Fon Analiz Sistemi**

YBSB 4007 - Derin Öğrenme Dönem Projesi

---

## 🎯 Proje Özeti

Bu proje, TEFAS (Türkiye Elektronik Fon Alım Satım Platformu) fonlarını derin öğrenme teknikleriyle analiz eden kapsamlı bir sistemdir. Üç ana modül içerir:

1. **Risk Profili Çıkarıcı** - Autoencoder ile anomali tespiti ve risk segmentasyonu
2. **Korelasyon Haritası** - Gizli ilişkilerin keşfi ve çeşitlendirme analizi  
3. **Portföy Simülasyonu** - Tarihsel backtest ve performans değerlendirmesi

⚠️ **DİKKAT**: Bu proje eğitim amaçlıdır. Yatırım tavsiyesi değildir.

---

## 🏗️ Sistem Mimarisi

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
│  ├── Risk Autoencoder                                │
│  ├── Embedding Autoencoder                           │
│  └── Risk-Return Scorer (ANN)                        │
├─────────────────────────────────────────────────────┤
│  Data Sources                                        │
│  ├── TEFAS API (borsapy)                             │
│  └── TCMB TÜFE Data                                  │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.11+
- Node.js 18+
- npm veya bun

### Kurulum

```bash
# Backend kurulumu
cd tefas_analysis
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend kurulumu
cd ../tefas-insight
npm install
```

### Çalıştırma

**Terminal 1 - Backend:**
```bash
cd tefas_analysis
source venv/bin/activate
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd tefas-insight
npm run dev -- --host
```

**Erişim:**
- Dashboard: http://localhost:5173
- API Docs: http://localhost:8000/docs

---

## 📊 Modüller

### Modül 1: Risk Profili Çıkarıcı

| Özellik | Açıklama |
|---------|----------|
| Model | Autoencoder (17 → 4 → 17) |
| Özellikler | Volatilite, Max Drawdown, Sharpe, VaR, CVaR vb. |
| Çıktı | Risk segmentasyonu (Düşük/Orta/Yüksek), Anomali tespiti |

### Modül 2: Korelasyon Haritası

| Özellik | Açıklama |
|---------|----------|
| Model | Embedding Autoencoder |
| Görselleştirme | 2D scatter plot, Korelasyon ısı haritası |
| Çıktı | Kümeleme, Çeşitlendirme fırsatları |

### Modül 3: Portföy Simülasyonu

| Özellik | Açıklama |
|---------|----------|
| Yöntem | Rule-based seçim + ANN skorlama |
| Metrikler | Sharpe, Sortino, Max Drawdown, Win Rate |
| Çıktı | Tarihsel backtest, Performans grafikleri |

---

## 🤖 AI Analizör

Dashboard'da her modül için **"AI Analizi"** butonu bulunur. Bu buton:

- Google Gemini API kullanır
- Uzman seviyesinde Türkçe yorumlar sunar
- Teknik olmayan kullanıcılar için sade açıklamalar yapar

---

## 📁 Proje Yapısı

```
tefas_analysis/
├── api/
│   ├── server.py          # FastAPI backend
│   └── ai_analyzer.py     # Gemini AI entegrasyonu
├── src/
│   ├── data/              # Veri toplama ve işleme
│   ├── features/          # Özellik mühendisliği
│   ├── models/            # Autoencoder, ANN modelleri
│   ├── modules/           # Ana analiz modülleri
│   └── evaluation/        # Metrik ve görselleştirme
├── config.py              # Konfigürasyon
├── main.py                # CLI interface
└── requirements.txt

tefas-insight/             # Frontend (React)
├── src/
│   ├── components/        # UI bileşenleri
│   ├── hooks/             # React hooks
│   ├── lib/               # API client
│   └── pages/             # Sayfa bileşenleri
└── package.json
```

---

## 🔧 API Endpoints

| Endpoint | Method | Açıklama |
|----------|--------|----------|
| `/api/funds` | GET | Tüm fonlar ve risk profilleri |
| `/api/analysis/run` | POST | Analiz pipeline'ı başlat |
| `/api/portfolio` | GET | Portföy simülasyon sonuçları |
| `/api/correlations` | GET | Korelasyon matrisi |
| `/api/ai-analyze/{module}` | POST | AI analizi al |
| `/api/module-info/{module}` | GET | Modül bilgileri |

---

## 📈 Örnek Çıktılar

### Risk Dağılımı
- Düşük Risk: 6 fon
- Orta Risk: 6 fon  
- Yüksek Risk: 6 fon

### Portföy Metrikleri
- Sharpe Ratio: 21.36
- Max Drawdown: -0.02%
- Win Rate: 94.12%

---

## 🛠️ Teknolojiler

**Backend:**
- Python 3.11
- FastAPI
- TensorFlow/Keras
- pandas, scikit-learn
- borsapy (TEFAS API)

**Frontend:**
- React 18 + TypeScript
- Vite
- shadcn/ui + Tailwind CSS
- Recharts
- TanStack Query

**AI:**
- Google Gemini API

---

## 📚 Metodoloji

Proje **CRISP-DM** metodolojisini takip eder:

1. **Business Understanding** - KuveytTürk ihtiyaç analizi
2. **Data Understanding** - TEFAS veri keşfi
3. **Data Preparation** - Temizleme, TÜFE düzeltmesi
4. **Modeling** - Autoencoder + ANN
5. **Evaluation** - Metrikler, backtest
6. **Deployment** - Web dashboard

---

## ⚠️ Sorumluluk Reddi

Bu yazılım yalnızca **eğitim amaçlıdır**. 

- Yatırım tavsiyesi değildir
- Geçmiş performans gelecek sonuçları garanti etmez
- Yatırım kararları için lisanslı danışmana başvurun
- KuveytTürk Portföy kavramsal paydaş olarak kullanılmıştır

---

## 👨‍💻 Geliştirici

YBSB 4007 - Derin Öğrenme Dönem Projesi

---

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
