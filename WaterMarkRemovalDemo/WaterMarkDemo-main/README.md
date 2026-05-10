# WaterMark Removal System

Görüntülerden watermark temizleyen, **kendi OCR motorunu** kullanan Python sistemi.

## Özellikler

| Özellik | Orijinal | Bu Versiyon |
|---|---|---|
| Watermark tespiti | Sadece model | OCR + renk + frekans + morfoloji |
| OCR motoru | Yok | Kendi motorumuz (CV tabanlı) |
| U-Net derinliği | 2 katman | 4 katman + BatchNorm + Dropout |
| Loss fonksiyonu | BCE | BCE + SSIM hibrit |
| Inpainting | Sabit TELEA | Adaptif NS / TELEA |
| Kalite ölçümü | Yok | PSNR + SSIM |
| PyTorch | Zorunlu | Opsiyonel |
| CLI | Basit menü | Menü + argüman desteği |

## Kurulum

```bash
git clone https://github.com/kullaniciadi/WaterMarkDemo.git
cd WaterMarkDemo/vize

# Zorunlu
pip install opencv-python numpy Pillow scikit-image

# Opsiyonel (U-Net eğitimi için)
pip install torch torchvision tqdm
```

## Kullanım

### İnteraktif Menü
```bash
python main.py
```

### Komut Satırı
```bash
# Watermark temizle (klasik mod — torch gerekmez)
python main.py --remove foto.png

# Watermark temizle (U-Net ile)
python main.py --remove foto.png --model model.pth --output temiz.png

# OCR tespiti görselleştir
python main.py --detect foto.png --viz

# Dataset oluştur
python main.py --dataset --clean-dir clean_images

# Model eğit
python main.py --train
```

## Proje Yapısı

```
vize/
├── main.py              ← Ana sistem (bu dosya)
├── requirements.txt
├── clean_images/        ← Temiz görüntüler (dataset üretimi için)
├── dataset/
│   ├── images/          ← Watermarklı görüntüler
│   └── masks/           ← Watermark maskeleri
└── model.pth            ← Eğitilmiş model (opsiyonel)
```

## Sistem Mimarisi

```
Giriş Görüntüsü
       │
       ├──► OCR Motoru (SimpleOCR)
       │       CLAHE → Otsu → CC Analizi → Projeksiyon → IoU Birleştirme
       │
       ├──► Renk Tespiti     (HSV: yüksek parlaklık + düşük doygunluk)
       ├──► Frekans Tespiti  (FFT büyüklük spektrumu)
       └──► Morfoloji         (Üst-hat dönüşümü)
               │
               ▼  Ağırlıklı oylama (eşik: 0.45)
           Birleşik Mask
               │
               ▼
       AdaptiveInpainter
         <1%  → TELEA r=5
         1-5% → NS    r=7
         >5%  → Kademeli NS→TELEA
               │
               ▼  Gaussian alpha-blend
           Temiz Görüntü  +  PSNR / SSIM metrikleri
```

## Kalite Metrikleri

| Metrik | Kötü | İyi | Mükemmel |
|---|---|---|---|
| PSNR | < 20 dB | > 30 dB | > 40 dB |
| SSIM | < 0.7 | > 0.9 | 1.0 |
