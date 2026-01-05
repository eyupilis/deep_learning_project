"""
TEFAS Fund Analysis - AI Analyzer Module
==========================================
Uses Google Gemini API to provide expert-level analysis of each module.
"""

import os
import json
import logging
from typing import Dict, Optional
import google.generativeai as genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC5pSWww3xejVx0nB99xwImm5RoJEP5IrQ")
genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
model = genai.GenerativeModel('gemini-3-flash-preview')

# System prompts for each module
SYSTEM_PROMPTS = {
    "risk_profile": """Sen bir fon risk analizi uzmanısın. KuveytTürk Portföy için çalışan bir veri bilimcisisin.

Görevin: Fon risk profili verilerini analiz edip, hem teknik hem de sade bir dille açıklamak.

Yanıtlarında:
1. Genel değerlendirme (2-3 cümle)
2. Öne çıkan bulgular (madde işaretli)
3. Dikkat edilmesi gerekenler
4. Pratik öneriler (yatırım tavsiyesi değil, genel bilgi)

UYARI: Bu yatırım tavsiyesi değildir. Eğitim amaçlı analizdir.""",

    "correlation": """Sen bir portföy çeşitlendirme uzmanısın. Fonlar arası korelasyon ve gizli ilişkileri analiz ediyorsun.

Görevin: Korelasyon verileri ve embedding analizini yorumlayıp, çeşitlendirme fırsatlarını ve riskleri açıklamak.

Yanıtlarında:
1. Korelasyon yapısı hakkında genel yorum
2. Yüksek korelasyonlu (benzer hareket eden) fon grupları
3. Düşük korelasyonlu (çeşitlendirme için uygun) fon çiftleri
4. Clustering'in ne anlama geldiği

Teknik terimleri sade Türkçe ile açıkla.""",

    "portfolio": """Sen bir portföy yöneticisisin. Simülasyon sonuçlarını yorumluyorsun.

Görevin: Portföy backtest sonuçlarını analiz edip, performans metriklerini açıklamak.

Yanıtlarında:
1. Genel performans değerlendirmesi
2. Sharpe Ratio, Max Drawdown gibi metriklerin ne anlama geldiği
3. Risk-getiri dengesi
4. Simülasyonun sınırlamaları

⚠️ ÖNEMLİ: Bu kesinlikle YATIRIM TAVSİYESİ DEĞİLDİR. Geçmiş performans gelecek sonuçları garanti etmez."""
}


def analyze_risk_profile(funds_data: list, risk_distribution: dict) -> str:
    """Analyze risk profile results with AI."""
    prompt = f"""
Risk Profili Verileri:

Toplam Fon Sayısı: {len(funds_data)}

Risk Dağılımı:
- Düşük Risk: {risk_distribution.get('Low', 0)} fon
- Orta Risk: {risk_distribution.get('Medium', 0)} fon
- Yüksek Risk: {risk_distribution.get('High', 0)} fon

Örnek Fonlar:
{json.dumps(funds_data[:5], indent=2, ensure_ascii=False)}

Anomali Sayısı: {sum(1 for f in funds_data if f.get('hasAnomaly', False))}

Bu verileri analiz et ve yorumla.
"""
    
    try:
        response = model.generate_content(
            [SYSTEM_PROMPTS["risk_profile"], prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"AI analizi şu anda kullanılamıyor: {str(e)}"


def analyze_correlation(correlations: list, embeddings: list) -> str:
    """Analyze correlation and embedding results with AI."""
    # Find high and low correlations
    high_corrs = [c for c in correlations if c['fundA'] != c['fundB'] and c['correlation'] > 0.7]
    low_corrs = [c for c in correlations if c['fundA'] != c['fundB'] and c['correlation'] < 0.3]
    
    # Cluster distribution
    clusters = {}
    for e in embeddings:
        cluster = e.get('cluster', 0)
        clusters[cluster] = clusters.get(cluster, 0) + 1
    
    prompt = f"""
Korelasyon ve Embedding Analizi:

Yüksek Korelasyonlu Çiftler (>0.7):
{json.dumps(high_corrs[:5], indent=2, ensure_ascii=False)}

Düşük Korelasyonlu Çiftler (<0.3):
{json.dumps(low_corrs[:5], indent=2, ensure_ascii=False)}

Küme Dağılımı:
{json.dumps(clusters, indent=2)}

Toplam Embedding: {len(embeddings)} fon

Bu verileri analiz et. Çeşitlendirme fırsatlarını ve gizli risk gruplarını açıkla.
"""
    
    try:
        response = model.generate_content(
            [SYSTEM_PROMPTS["correlation"], prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"AI analizi şu anda kullanılamıyor: {str(e)}"


def analyze_portfolio(metrics: dict, history: list, selected_funds: list) -> str:
    """Analyze portfolio simulation results with AI."""
    prompt = f"""
Portföy Simülasyon Sonuçları:

Seçilen Fonlar: {', '.join(selected_funds) if selected_funds else 'Belirtilmedi'}

Performans Metrikleri:
- Toplam Getiri: {metrics.get('totalReturn', 0):.2f}%
- Yıllık Getiri: {metrics.get('annualizedReturn', 0):.2f}%
- Sharpe Ratio: {metrics.get('sharpeRatio', 0):.2f}
- Maksimum Düşüş: {metrics.get('maxDrawdown', 0):.2f}%
- Kazanma Oranı: {metrics.get('winRate', 0):.2f}%

Simülasyon Süresi: {len(history)} gün

Bu simülasyon sonuçlarını analiz et. Metrikleri sade bir dille açıkla.
"""
    
    try:
        response = model.generate_content(
            [SYSTEM_PROMPTS["portfolio"], prompt],
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1000,
            )
        )
        return response.text
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"AI analizi şu anda kullanılamıyor: {str(e)}"


# Module descriptions in Turkish
MODULE_DESCRIPTIONS = {
    "risk_profile": {
        "title": "Risk Profili Analizi",
        "description": "Her fonu 17+ risk özelliğiyle analiz eder ve otomatik olarak Düşük/Orta/Yüksek risk kategorilerine ayırır.",
        "why_important": "Yatırımcı profilinize uygun fonları hızlıca belirlemenize yardımcı olur. Risk toleransınıza göre filtreleme yapabilirsiniz.",
        "how_to_read": "Yeşil = Düşük Risk (daha az dalgalanma), Sarı = Orta Risk, Kırmızı = Yüksek Risk (daha fazla potansiyel getiri/kayıp). Anomali işareti olan fonlar dikkatle incelenmelidir.",
        "technical": "Autoencoder modeli ile risk özellikleri öğrenilir. Reconstruction error yüksek olan fonlar anomali olarak işaretlenir."
    },
    "correlation": {
        "title": "Korelasyon ve Çeşitlendirme Haritası",
        "description": "Fonlar arasındaki gizli ilişkileri ortaya çıkarır ve gerçek çeşitlendirme fırsatlarını gösterir.",
        "why_important": "Portföyünüzün gerçekten çeşitlendirilmiş olup olmadığını anlamanızı sağlar. Yüksek korelasyonlu fonlar aynı yönde hareket eder.",
        "how_to_read": "Korelasyon 1'e yakın = Benzer hareket. 0'a yakın = Bağımsız hareket. Scatter plot'ta yakın noktalar benzer fonları gösterir.",
        "technical": "Embedding Autoencoder ile fon getirileri 2D uzaya indirgenir. K-Means clustering ile benzer fonlar gruplandırılır."
    },
    "portfolio": {
        "title": "Portföy Simülasyonu",
        "description": "Seçilen risk segmentindeki fonlarla örnek bir portföy oluşturur ve geçmiş performansını simüle eder.",
        "why_important": "Yatırım kararlarınız için tarihsel referans sağlar. Farklı risk seviyelerinin geçmiş performansını karşılaştırabilirsiniz.",
        "how_to_read": "Sharpe Ratio > 1 iyi, Max Drawdown düşük olmalı. Grafik portföy değerinin zaman içinde değişimini gösterir.",
        "technical": "Rule-based fon seçimi + ANN skorlama. Eşit ağırlık veya ters volatilite ağırlıklandırma. Historical backtest.",
        "disclaimer": "⚠️ Bu simülasyon YATIRIM TAVSİYESİ DEĞİLDİR. Geçmiş performans gelecek sonuçları garanti etmez. Yatırım kararlarınız için lisanslı bir danışmana başvurun."
    }
}


TECHNICAL_DETAILS = {
    "architecture": """
## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    TEFAS Fund Analysis                       │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React + TypeScript)                               │
│  ├── Dashboard Components                                    │
│  ├── AI Analyzer Dialogs                                     │
│  └── Recharts Visualizations                                 │
├─────────────────────────────────────────────────────────────┤
│  Backend (FastAPI + Python)                                  │
│  ├── REST API Endpoints                                      │
│  ├── Gemini AI Integration                                   │
│  └── Analysis Pipeline                                       │
├─────────────────────────────────────────────────────────────┤
│  Deep Learning Modules                                       │
│  ├── Module 1: Risk Autoencoder                              │
│  ├── Module 2: Embedding Autoencoder                         │
│  └── Module 3: Risk-Return Scorer (ANN)                      │
├─────────────────────────────────────────────────────────────┤
│  Data Sources                                                │
│  ├── TEFAS API (borsapy)                                     │
│  └── TCMB TÜFE Data                                          │
└─────────────────────────────────────────────────────────────┘
```
""",
    "autoencoder": """
## Autoencoder Nedir?

Autoencoder, veriyi sıkıştırıp yeniden oluşturan bir sinir ağı modelidir.

**Nasıl Çalışır:**
1. Encoder: Yüksek boyutlu veriyi → Düşük boyutlu temsile sıkıştırır
2. Latent Space: Verinin özet temsili (embedding)
3. Decoder: Düşük boyutlu temsili → Orijinal veriye geri çevirir

**Neden Kullanıyoruz:**
- **Anomali Tespiti**: Normal fonları iyi öğrenen model, anormal fonları kötü yeniden oluşturur
- **Embedding**: Fonları 2D uzayda görselleştirmek için
- **Feature Learning**: Gizli örüntüleri otomatik keşfetmek için
""",
    "why_deep_learning": """
## Neden Deep Learning?

**Geleneksel Yöntemlerin Sınırlamaları:**
- Lineer korelasyon sadece doğrusal ilişkileri yakalar
- Önceden tanımlanmış formüller (Sharpe, VaR) tek boyutlu
- İnsan tanımlı özellikler sınırlı

**Deep Learning Avantajları:**
- Non-lineer ilişkileri keşfeder
- Otomatik özellik çıkarımı (feature learning)
- Yüksek boyutlu veriyi anlamlandırır
- Görselleştirilebilir embedding'ler üretir
""",
    "crisp_dm": """
## CRISP-DM Metodolojisi

Bu proje CRISP-DM (Cross-Industry Standard Process for Data Mining) metodolojisi ile geliştirildi:

1. **Business Understanding**: KuveytTürk Portföy ihtiyaç analizi
2. **Data Understanding**: TEFAS fon verilerinin keşfi
3. **Data Preparation**: Veri temizleme, dönüştürme, TÜFE düzeltmesi
4. **Modeling**: Autoencoder + ANN modelleri
5. **Evaluation**: Risk segmentasyonu, backtest metrikleri
6. **Deployment**: Web dashboard + API
"""
}


if __name__ == "__main__":
    # Test
    print("AI Analyzer module loaded")
    print("Available functions: analyze_risk_profile, analyze_correlation, analyze_portfolio")
