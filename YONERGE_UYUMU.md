# Yönerge Uyumu – Modül 2: Aydınlatma Düzeltme (Elif Yeşilyurt)

Bu dosya, *Image Processing Project Guidelines* (14 Ekim 2026) belgesindeki gerekliliklerin Elif Yeşilyurt'un modülünde (Modül 2 – Illumination Correction) nasıl karşılandığını gösterir.

| Gereklilik | Durum | Kanıt |
|---|---|---|
| Veri kaynağı ve lisans | Karşılandı (MIT lisansı, Xu et al. 2018 atfı) | `README.md` → *Data source, licence, citation* |
| Problem, girdi, beklenen çıktı ve değerlendirme | Karşılandı | `README.md` → *Problem, input, output, evaluation*; notebook giriş bölümü |
| Aynı test görüntülerinde yöntem karşılaştırması | Karşılandı | 50 kırpım × 4 yöntem: `results/crops/`; `results/metrics_summary_ocr.csv` |
| Başarılı ve başarısız örnekler, hata nedenleri | Karşılandı | `results/figures/representative_{success,unchanged,failure}.png`; `review/visual_review.csv`; `README.md` → *Failure causes* |
| Net bireysel görev ve kanıt | Karşılandı | `presentation/elif_sunum.md` (katkı tablosu satırı); MF-Sync IMA-001 … IMA-008 |
| Notebook ve README | Karşılandı | `elif_module2_illumination.ipynb` (çalıştırılmış); `README.md` |
| Raporda değişiklik varsa açıklama | Karşılandı (geçici kırpımlar, 30'dan 50'ye genişletme) | `README.md` → *Image selection*; `results/selection_rule.txt` |

## Notlar
- Raporda Elif'in modülüne yazılmış Top-Hat/Black-Hat adımı da uygulandı; ancak sonucu kötüleştirdi (CER 0,620 → 0,663).
- Karakter segmentasyon doğruluğu Hakan'ın modülüne (Modül 4) ait, burada ölçülmedi. Derin öğrenme kullanılmadı.

## Veri gizliliği notu
GitHub deposu (https://github.com/elifyesilyurt/image) herkese açık. İçinde CCPD'den türetilmiş 50 plaka kırpımı ve plaka metinleri var. CCPD zaten herkese açık ve MIT lisanslı; kaynak veri klasörünün tamamı depoya kopyalanmadı. Deponun gizli olması istenirse görünürlük şu komutla değiştirilebilir:

```bash
gh repo edit elifyesilyurt/image --visibility private
```
