# Yönerge Uyumu – Modül 2: Aydınlatma Düzeltme (Elif Yeşilyurt)

Bu dosya, *Image Processing Project Guidelines* (14 Ekim 2026) belgesindeki gerekliliklerin ve grup raporunda Elif'e ait kısmın Modül 2'de nasıl karşılandığını gösterir. Güncel sonuçlar `results_v2/` klasöründedir.

| Gereklilik | Durum | Kanıt |
|---|---|---|
| Veri kaynağı ve lisans | Karşılandı (MIT lisansı, Xu et al. 2018 atfı) | `README.md` → *Data source, licence, citation* |
| Problem, girdi, beklenen çıktı ve değerlendirme | Karşılandı | `README.md` → *Problem, input, output, evaluation*; notebook giriş bölümü |
| Aynı test görüntülerinde yöntem karşılaştırması | Karşılandı (5 yöntem, 50 kırpım) | `results_v2/crops/`; `results_v2/metrics_summary_ocr.csv` |
| Bir yöntemin daha iyi olduğunun nasıl belirlendiği | Karşılandı (CER + eşleştirilmiş bootstrap güven aralığı) | `README.md` → *Results* |
| Raporun global ↔ adaptif karşılaştırması | Karşılandı (Global HE ↔ CLAHE) | `results_v2/figures/representative_global_vs_adaptive.png` |
| Modül tanımı: "flaş parlaması veya gölgeden kaynaklanan dengesiz ışık" | Karşılandı (keşifsel alt gruplar) | `results_v2/illumination_strata.txt`; `figures/cer_by_illumination_exploratory.png` |
| Başarılı ve başarısız örnekler, hata nedenleri | Karşılandı | `results_v2/figures/representative_*.png`; `review/visual_review.csv` |
| Net bireysel görev ve kanıt | Karşılandı | Katkı satırı: `presentation/elif_presentation_EN.md`; MF-Sync IMA-001 … IMA-013; git geçmişi |
| Notebook ve README | Karşılandı | `elif_module2_illumination.ipynb` (çalıştırılmış); `README.md` |
| Sunum (en fazla 10 dk, video) | Slaytlar ve metin hazır; **video kaydı Elif tarafından yapılacak** | 7 İngilizce slayt (yaklaşık 2 dk), konuşma metni |
| Projenin ana adımlarını açıklayabilme | Hazırlık notu hazır | `presentation/elif_presentation_EN.md` → *the whole project* |
| Raporda değişiklik varsa açıklama | Karşılandı (geçici kırpımlar, global HE eklendi, Top/Black-Hat önerilmiyor) | `README.md`; sunumun sonuç slaytı |
| Yöntem kaynakları | Karşılandı (8 kaynak) | `README.md` → *References* |

## Notlar
- Raporda Elif'in modülüne yazılmış Top-Hat/Black-Hat adımı uygulandı; ancak sonucu kötüleştirdi (CER 0,620 → 0,663).
- Karakter segmentasyon doğruluğu Hakan'ın modülüne (Modül 4) ait, burada ölçülmedi. Derin öğrenme kullanılmadı.

## Grupla birlikte tamamlanacaklar
- Modül 1'in (Gül Berra) gerçek kırpımlarıyla yeniden çalıştırma (`INTEGRATION.md`).
- Modül 3'ün (Beyza) CLAHE/Homomorphic çıktısını girdi olarak kullanması.
- Grup `.ipynb` dosyasına birleştirme ve tam katkı tablosu.

## Veri gizliliği notu
GitHub deposu (https://github.com/elifyesilyurt/image) herkese açık. İçinde CCPD'den türetilmiş 50 plaka kırpımı ve plaka metinleri var. CCPD zaten herkese açık ve MIT lisanslı; kaynak veri klasörünün tamamı depoya kopyalanmadı. Deponun gizli olması istenirse görünürlük şu komutla değiştirilebilir:

```bash
gh repo edit elifyesilyurt/image --visibility private
```
