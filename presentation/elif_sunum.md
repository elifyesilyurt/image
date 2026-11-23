# Elif Yeşilyurt – Modül 2: Aydınlatma Düzeltme (bireysel sunum materyali)

> **Not:** Bu Türkçe taslak ilk çalıştırmaya (`results/`, 4 yöntem) dayanır. Güncel ve İngilizce sunum materyali: [`elif_presentation_EN.md`](elif_presentation_EN.md) (`results_v2/`, 5 yöntem).

## Katkı tablosu satırı (grup sunumu için)
| Öğrenci | Görev | Çıktılar |
|---|---|---|
| Elif Yeşilyurt (240610008) | Modül 2 – Illumination Correction | Aynı 50 plaka kırpımında CLAHE, CLAHE+Top/Black-Hat ve Homomorphic Filtering; öncesi/sonrası panelleri; Tesseract CER karşılaştırması; başarılı/başarısız örnek analizi; notebook + README |

## Sunum metni (~80 saniye, ~160 kelime)
**[Slayt: problem ve öncesi/sonrası örneği]**
"Benim modülüm aydınlatma düzeltme. Plaka gece karanlık kalabiliyor ya da flaş ve güneşle parlayabiliyor. Bu durumda karakterle zemin arasındaki kontrast düşüyor ve sonraki ikilileştirme adımı zorlaşıyor.

**[Slayt: yöntemler]**
İki klasik yöntemi karşılaştırdım. CLAHE görüntüyü küçük bölgelere ayırıp her bölgenin histogramını, aşırı güçlendirmeyi sınırlayarak eşitliyor. Bunu yalnızca LAB uzayındaki parlaklık kanalına uyguladım, böylece renk bozulmadı. Homomorphic filtrelemede görüntüyü aydınlatma çarpı yansıma olarak düşünüyoruz. Logaritma alınca bu çarpım toplama dönüşüyor. Fourier alanında düşük frekanslı aydınlatmayı bastırıp yüksek frekanslı kenarları güçlendiriyoruz. Raporumuzdaki Top-Hat/Black-Hat adımını da ayrıca denedim.

**[Slayt: deney ve sonuç grafiği]**
CCPD-DB'den 50 plakayı aynı kırpımlarla, aynı Tesseract ayarlarıyla test ettim. Ölçüt karakter hata oranı. Genel ortalamada iyileşme küçük: orijinal 0,62, CLAHE 0,61, Homomorphic 0,61. Bu farklar 50 görüntüde istatistiksel olarak anlamlı değil. Karanlık plakalarda ise Homomorphic hatayı 0,56'dan 0,50'ye indirdi.

**[Slayt: başarısız örnekler]**
Parlamış plakalarda hiçbir yöntem işe yaramadı, çünkü doygun piksellerde bilgi kalmamış. Çok karanlık görüntülerde JPEG blokları büyüdü. Top-Hat ise karakterlerin etrafında hale oluşturdu.

**[Slayt: sınırlılık]**
Sınırlılıklarım: kırpımlar Modül 1'in nihai çıktısı değil, örneklem küçük ve Tesseract plaka için tasarlanmamış bir OCR motoru."

## Muhtemel sorular ve kısa cevaplar
1. **CLAHE ile Homomorphic Filtering arasındaki temel fark ne?**
   CLAHE uzamsal alanda, bölgesel histogram eşitlemesiyle kontrastı yeniden dağıtıyor; clip limit gürültünün aşırı güçlenmesini sınırlıyor. Homomorphic frekans alanında çalışıyor. f = i·r modelinde logaritma alıp düşük frekanslı aydınlatma bileşenini (γL = 0,5) bastırıyor, yüksek frekanslı yansıma ve kenarları (γH = 1,5) güçlendiriyor. Yani CLAHE kontrastı yerel olarak yeniden dağıtıyor; Homomorphic ise aydınlatmayı modelleyip ayırıyor.

2. **Neden yalnızca L kanalını işlediniz?**
   RGB kanallarını ayrı ayrı eşitlemek renk tonunu bozar. LAB'da L parlaklığı, a ve b ise rengi taşır. Yalnızca L'yi değiştirince plaka rengi korunuyor; ölçtüğüm renk kayması da düşük kaldı.

3. **Parametreleri nasıl seçtiniz? Sonuca göre ayar yapmadınız mı?**
   Görüntü parametrelerini OCR'ı hiç çalıştırmadan önce, 3 örnek üzerinde parametre taramasına bakarak sabitledim. OCR ayarlarını ise test görüntülerinden ayrı, 20 görüntülük bir kalibrasyon kümesinde ve yalnızca işlenmemiş kırpımlarla seçtim. Böylece hiçbir yöntem lehine ayar yapılmadı.

4. **Sonuç neden bu kadar küçük ya da anlamsız çıktı?**
   Üç neden var:
   - Parlamış plakalarda doygun piksellerde kaybolan bilgiyi hiçbir aydınlatma yöntemi geri getiremez.
   - Tesseract genel amaçlı bir doküman OCR'ı; Çince il karakterini neredeyse hiç okumuyor ve 50 plakanın hiçbirinde tam eşleşme yok.
   - 50 görüntü küçük bir örneklem. Bootstrap güven aralıkları sıfırı içeriyor, bu yüzden "iyileştirdi" diyemiyorum. Karanlık plakalarda Homomorphic'te görülen düşüş bir eğilim.

5. **Başarısız örneklerin nedeni ne ve nasıl düzeltilebilir?**
   - Doygunluk: çekim aşamasında çözülmesi gereken bir sorun (pozlama, HDR).
   - JPEG blokları: karanlık görüntülerde kontrast artırma blok kenarlarını da büyütüyor. Öncesinde hafif bir blok giderme ya da yumuşatma denenebilir.
   - Top-Hat haleleri: yapı elemanını ve ağırlığı küçültmek denenebilir.

   Bu iyileştirmeleri denemedim; sınırlılık olarak raporladım.

## Sunumda kullanılacak görseller
- `results/figures/representative_success.png`, `representative_failure.png`
- `results/figures/cer_by_group.png`
- `results/figures/sweep_homomorphic.png` (yöntemin parametre etkisini anlatmak için)
- `results/figures/luminance_histograms.png`
