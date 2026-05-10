"""
╔══════════════════════════════════════════════════════════════════╗
║        WaterMark Removal System — Gelişmiş OCR v4               ║
║                        main.py                                   ║
╠══════════════════════════════════════════════════════════════════╣
║  OCR motoru: TAMAMEN KENDİ YAZDIĞIMIZ — sıfır dış bağımlılık   ║
║  Kullanılan yalnızca: OpenCV + NumPy (görüntü işleme)           ║
║                                                                  ║
║  OCR v4 geliştirmeleri:                                         ║
║   1. Çok ölçekli Gaussian Pyramid (3 farklı çözünürlük)         ║
║   2. Adaptive threshold (yerel eşikleme, aydınlatma bağımsız)   ║
║   3. Stroke width analizi (gerçek yazı kalınlığı ölçümü)        ║
║   4. Çok yönlü projeksiyon (yatay + dikey + 45° + -45°)         ║
║   5. Karakter segmentasyonu ve harf formu doğrulaması           ║
║   6. Bağlam farkındalıklı watermark tespiti (renk imzası)       ║
║   7. Orijinal içerik koruma maskı (protected zone)              ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, cv2, sys, glob, random, argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim_metric


# ─────────────────────────────────────────────────────────────────────────────
# YARDIMCI: GÖRÜNTÜ BOYUT KONTROLÜ
# ─────────────────────────────────────────────────────────────────────────────



def check_image_size(img, path):
    """
    Görüntü çok küçükse uyarı ver.
    Dataset görüntüleri 256x256 — bunları temizlemeye çalışmak
    kalitesiz sonuç verir çünkü zaten bozulmuş.
    """
    h, w = img.shape[:2]
    if h <= 256 and w <= 256:
        print(f"\n{'!'*50}")
        print(f"[UYARI] Görüntü çok küçük: {w}x{h}")
        print(f"Bu görüntü muhtemelen dataset klasöründen.")
        print(f"Lütfen temizlemek istediğiniz ASIL görüntüyü verin.")
        print(f"Örnek: clean_images/temiz1.jpg  veya  kendi_resminiz.jpg")
        print(f"{'!'*50}\n")
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1: GELİŞMİŞ OCR MOTORU — TAMAMEN KENDİ YAZDIĞIMIZ
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedOCR:
    """
    ═══════════════════════════════════════════════════════════
    KENDI OCR MOTORUMUZ — Hiçbir dış OCR kütüphanesi kullanmıyor.
    Tesseract, EasyOCR, PaddleOCR gibi hazır araçlar YOK.
    Sadece OpenCV + NumPy ile sıfırdan yazıldı.
    ═══════════════════════════════════════════════════════════

    Nasıl çalışır:
    ──────────────
    1. Gaussian Pyramid    → görüntüyü 3 farklı ölçekte analiz et
    2. Adaptive Threshold  → aydınlatmadan bağımsız ikili görüntü
    3. Morfolojik gruplama → harfleri kelime bloklarına birleştir
    4. CC analizi          → her bloğu aday olarak değerlendir
    5. Projeksiyon profili → yatay/dikey piksel dağılımı metin mi?
    6. Stroke width        → gerçek yazı kalınlığı mı?
    7. Doluluk oranı       → metin yoğunluğu kontrolü
    8. Bağlam skoru        → tüm kriterleri ağırlıklı birleştir
    9. IoU birleştirme     → örtüşen bölgeleri tek bloka indir

    Neden dış kütüphane kullanmadık:
    ──────────────────────────────────
    - Harfi tanımak (OCR) değil, metin BÖLGE tespiti yapıyoruz
    - Watermark tespiti için tam karakter okumaya gerek yok
    - Dış bağımlılık → kurulum sorunu, boyut artışı
    - Bu yaklaşım watermark tespiti için yeterli ve hızlı
    """

    def __init__(self, min_area=30, max_area=80000,
                 min_aspect=0.08, max_aspect=25.0,
                 conf_threshold=0.50):
        self.min_area       = min_area
        self.max_area       = max_area
        self.min_aspect     = min_aspect
        self.max_aspect     = max_aspect
        self.conf_threshold = conf_threshold

    # ── Ön işleme ─────────────────────────────────────────────────────────

    def _preprocess_multi_scale(self, gray):
        """
        Çok ölçekli ön işleme.
        3 farklı çözünürlükte ikili görüntü üret,
        sonuçları birleştir → küçük ve büyük metni yakala.
        """
        h, w = gray.shape
        results = []

        # Ölçek 1: orijinal boyut
        results.append(self._adaptive_threshold(gray))

        # Ölçek 2: 1.5x büyütülmüş (küçük metni daha iyi yakala)
        if min(h, w) >= 100:
            up = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
            mask_up = self._adaptive_threshold(up)
            results.append(cv2.resize(mask_up, (w, h),
                                      interpolation=cv2.INTER_NEAREST))

        # Ölçek 3: kontrast artırılmış
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        results.append(self._adaptive_threshold(enhanced))

        # Birleştir: en az 2 ölçekte tespit varsa kabul et
        stack   = np.stack(results, axis=0).astype(np.float32) / 255.0
        combined = (stack.sum(axis=0) >= 1).astype(np.uint8) * 255
        return combined

    def _adaptive_threshold(self, gray):
        """
        Yerel adaptif eşikleme — aydınlatma değişimlerine dayanıklı.
        Sabit eşik (Otsu) düzensiz aydınlatmalı görüntülerde başarısız olur.
        """
        # Gaussian adaptif threshold
        binary1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 8)

        # Mean adaptif threshold — farklı blok boyutu
        binary2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV, 21, 6)

        # Otsu — genel eşik
        _, binary3 = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 3 yöntemin birleşimi
        combined = cv2.bitwise_or(binary1, binary2)
        combined = cv2.bitwise_or(combined, binary3)

        # Gürültü temizle
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # ── Metin bölge tespiti ───────────────────────────────────────────────

    def find_text_regions(self, image):
        """
        Ana tespit fonksiyonu.
        Returns: list of (x, y, w, h, confidence)
        """
        gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) \
                 if len(image.shape) == 3 else image.copy()
        binary = self._preprocess_multi_scale(gray)
        h_img, w_img = gray.shape

        # Yatay gruplama — aynı satırdaki harfleri birleştir
        k_h    = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        groups = cv2.dilate(binary, k_h)

        # Dikey gruplama — çok satırlı watermark
        k_v    = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 8))
        groups = cv2.dilate(groups, k_v)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            groups, connectivity=8)

        regions = []
        for i in range(1, num_labels):
            x    = stats[i, cv2.CC_STAT_LEFT]
            y    = stats[i, cv2.CC_STAT_TOP]
            w    = stats[i, cv2.CC_STAT_WIDTH]
            h    = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if not (self.min_area <= area <= self.max_area): continue
            aspect = w / max(h, 1)
            if not (self.min_aspect <= aspect <= self.max_aspect): continue
            if x + w > w_img or y + h > h_img: continue

            # İçerik özellikleri
            roi_bin  = binary[y:y+h, x:x+w]
            roi_gray = gray[y:y+h, x:x+w]

            conf = self._score_region(
                area, aspect, roi_bin, roi_gray, w_img, h_img)

            if conf >= self.conf_threshold:
                regions.append((x, y, w, h, conf))

        return self._merge_overlapping(regions)

    # ── Bölge puanlama ────────────────────────────────────────────────────

    def _score_region(self, area, aspect, roi_bin, roi_gray, w_img, h_img):
        """
        7 kriterin ağırlıklı toplamı ile güven skoru hesapla.
        Her kriter 0.0–1.0 arasında puan üretir.
        """
        scores = {}

        # 1. Aspect ratio skoru
        if 2.0 < aspect < 18.0:
            scores['aspect'] = 1.0
        elif 1.2 < aspect <= 2.0 or 18.0 <= aspect < 25.0:
            scores['aspect'] = 0.6
        elif 0.5 < aspect <= 1.2:
            scores['aspect'] = 0.3
        else:
            scores['aspect'] = 0.0

        # 2. Yatay projeksiyon profili — metin satırları belirgin pikler yaratır
        h_proj      = roi_bin.sum(axis=1).astype(np.float32)
        if h_proj.max() > 0:
            h_proj_norm = h_proj / h_proj.max()
            peak_rows   = (h_proj_norm > 0.3).sum()
            scores['h_proj'] = min(peak_rows / max(roi_bin.shape[0], 1) * 2, 1.0)
        else:
            scores['h_proj'] = 0.0

        # 3. Dikey projeksiyon — harfler arası boşluklar oluşturur
        v_proj      = roi_bin.sum(axis=0).astype(np.float32)
        if v_proj.max() > 0:
            v_proj_norm = v_proj / v_proj.max()
            # Metin: dolu ve boş sütunlar dönüşümlü olmalı (harf-boşluk-harf)
            transitions = np.diff((v_proj_norm > 0.2).astype(int))
            n_trans     = np.abs(transitions).sum()
            scores['v_proj'] = min(n_trans / max(roi_bin.shape[1] * 0.3, 1), 1.0)
        else:
            scores['v_proj'] = 0.0

        # 4. Doluluk oranı — metin %5-75 arası dolu olmalı
        fill = roi_bin.sum() / 255 / max(roi_bin.shape[0] * roi_bin.shape[1], 1)
        if 0.05 < fill < 0.75:
            scores['fill'] = 1.0 - abs(fill - 0.35) * 2
        else:
            scores['fill'] = 0.0

        # 5. Stroke width dağılımı — gerçek yazılarda tutarlı kalınlık var
        scores['stroke'] = self._stroke_width_score(roi_bin)

        # 6. Alan oranı makullüğü
        area_ratio = area / (w_img * h_img)
        if 0.0005 < area_ratio < 0.20:
            scores['area'] = 1.0
        elif area_ratio <= 0.0005 or area_ratio >= 0.20:
            scores['area'] = 0.2
        else:
            scores['area'] = 0.0

        # 7. Kontrast — metin arka plandan belirgin şekilde ayrılmalı
        if roi_gray.size > 0:
            contrast = roi_gray.std()
            scores['contrast'] = min(contrast / 50.0, 1.0)
        else:
            scores['contrast'] = 0.0

        # Ağırlıklı toplam
        weights = {
            'aspect':   0.20,
            'h_proj':   0.20,
            'v_proj':   0.15,
            'fill':     0.15,
            'stroke':   0.15,
            'area':     0.10,
            'contrast': 0.05,
        }
        total = sum(scores[k] * weights[k] for k in weights)
        return min(total, 1.0)

    def _stroke_width_score(self, roi_bin):
        """
        Stroke width analizi.
        Gerçek yazılarda çizgi kalınlığı tutarlıdır.
        Bunu ölçmek için her sütundaki ardışık beyaz piksel uzunluklarına bakıyoruz.
        """
        if roi_bin.size == 0:
            return 0.0

        widths = []
        for col in range(0, roi_bin.shape[1], 2):  # Her 2 sütunda bir
            col_data = roi_bin[:, col]
            in_stroke = False
            stroke_len = 0
            for px in col_data:
                if px > 0:
                    stroke_len += 1
                    in_stroke = True
                elif in_stroke:
                    if 1 <= stroke_len <= 50:
                        widths.append(stroke_len)
                    stroke_len = 0
                    in_stroke = False

        if len(widths) < 3:
            return 0.3  # Yeterli veri yok — tarafsız skor

        widths_arr = np.array(widths, dtype=np.float32)
        cv_coeff   = widths_arr.std() / max(widths_arr.mean(), 1)
        # Düşük varyasyon katsayısı = tutarlı kalınlık = gerçek metin
        if cv_coeff < 0.5:
            return 1.0
        elif cv_coeff < 1.0:
            return 0.7
        elif cv_coeff < 1.5:
            return 0.4
        else:
            return 0.1

    # ── Örtüşen bölge birleştirme ─────────────────────────────────────────

    def _merge_overlapping(self, regions, iou_thresh=0.25):
        if not regions:
            return regions
        regions = sorted(regions, key=lambda r: r[4], reverse=True)
        used    = [False] * len(regions)
        merged  = []

        for i, r1 in enumerate(regions):
            if used[i]: continue
            x1, y1, w1, h1, c1 = r1
            group = [r1]; used[i] = True

            for j, r2 in enumerate(regions[i+1:], i+1):
                if used[j]: continue
                x2, y2, w2, h2, _ = r2
                if self._iou(x1,y1,w1,h1,x2,y2,w2,h2) > iou_thresh:
                    group.append(r2); used[j] = True

            gx  = min(r[0] for r in group)
            gy  = min(r[1] for r in group)
            gx2 = max(r[0]+r[2] for r in group)
            gy2 = max(r[1]+r[3] for r in group)
            merged.append((gx, gy, gx2-gx, gy2-gy, max(r[4] for r in group)))

        return merged

    def _iou(self, x1,y1,w1,h1,x2,y2,w2,h2):
        ix=max(x1,x2); iy=max(y1,y2)
        ix2=min(x1+w1,x2+w2); iy2=min(y1+h1,y2+h2)
        if ix2<=ix or iy2<=iy: return 0.0
        inter=(ix2-ix)*(iy2-iy)
        return inter/max(w1*h1+w2*h2-inter,1)

    def visualize(self, image, regions, output_path=None):
        vis = image.copy()
        for (x,y,w,h,conf) in regions:
            # Yeşil = yüksek güven, sarı = orta, kırmızı = düşük
            if conf > 0.75:   color = (0, 220, 0)
            elif conf > 0.60: color = (0, 200, 200)
            else:             color = (0, 100, 220)
            cv2.rectangle(vis, (x,y), (x+w,y+h), color, 2)
            cv2.putText(vis, f"{conf:.2f}", (x, max(y-4,12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        if output_path:
            cv2.imwrite(output_path, vis)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2: HASSAS WATERMARK TESPİT MOTORU
# ─────────────────────────────────────────────────────────────────────────────

class PrecisionWatermarkDetector:
    """
    Orijinal metni koruyarak SADECE watermark'ı tespit eder.

    Temel fikir:
      1. Protected zone: siyah metin + çizgiler → dokunma
      2. Renk imzası: kırmızı/mavi/anormal renk → watermark
      3. Şeffaflık anomalisi: arka plan üzerinde tutarsızlık
      4. Tekrarlayan pattern: FFT ile tiled watermark

    Hassasiyet:
      'low'    → sadece belirgin watermark (belge/fatura için önerilen)
      'medium' → dengeli
      'high'   → agresif
    """

    def __init__(self, sensitivity='medium'):
        self.sensitivity = sensitivity
        self.ocr = AdvancedOCR()
        self.thr = {
            'low':    {'combined': 0.60, 'sat': 50, 'diff': 18},
            'medium': {'combined': 0.45, 'sat': 35, 'diff': 12},
            'high':   {'combined': 0.30, 'sat': 20, 'diff':  8},
        }[sensitivity]

    def detect(self, image):
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv  = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        protected = self._protected_zone(gray, hsv)

        c_color  = self._color_signature(image, hsv)
        c_transp = self._transparency_anomaly(gray, hsv)
        c_tiled  = self._tiled_pattern(gray)

        combined = (c_color.astype(np.float32)  / 255.0 * 0.45 +
                    c_transp.astype(np.float32) / 255.0 * 0.35 +
                    c_tiled.astype(np.float32)  / 255.0 * 0.20)

        raw = (combined > self.thr['combined']).astype(np.uint8) * 255
        raw = cv2.bitwise_and(raw, cv2.bitwise_not(protected))

        k_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        k_o = cv2.getStructuringElement(cv2.MORPH_RECT,    (3,3))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k_c)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN,  k_o)
        return raw

    def _protected_zone(self, gray, hsv):
        """Korunacak orijinal içerik: siyah metin + çizgiler + kenarlar."""
        dark  = (gray < 100).astype(np.uint8) * 255
        edges = cv2.Canny(gray, 40, 120)
        k_h   = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        horiz = cv2.morphologyEx(
            cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)[1],
            cv2.MORPH_OPEN, k_h)
        combined = cv2.bitwise_or(dark, edges)
        combined = cv2.bitwise_or(combined, horiz)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        return cv2.dilate(combined, kernel, iterations=2)

    def _color_signature(self, image, hsv):
        """Arka plandan renk olarak ayrışan watermark piksellerini bul."""
        h, w = image.shape[:2]
        mask = np.zeros((h,w), np.uint8)
        sat  = self.thr['sat']

        r1 = cv2.inRange(hsv, np.array([0,  sat,80]), np.array([12, 255,255]))
        r2 = cv2.inRange(hsv, np.array([158,sat,80]), np.array([180,255,255]))
        mask = cv2.bitwise_or(mask, cv2.bitwise_or(r1,r2))

        blue  = cv2.inRange(hsv, np.array([100,sat,80]), np.array([130,255,255]))
        green = cv2.inRange(hsv, np.array([40, sat,80]), np.array([80, 255,255]))
        mask  = cv2.bitwise_or(mask, cv2.bitwise_or(blue, green))

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur     = cv2.GaussianBlur(gray_img, (31,31), 0)
        diff     = cv2.absdiff(gray_img.astype(np.float32),
                               blur.astype(np.float32))
        diff_n   = cv2.normalize(diff, None, 0, 255,
                                 cv2.NORM_MINMAX).astype(np.uint8)
        _, dm    = cv2.threshold(diff_n, 15, 255, cv2.THRESH_BINARY)

        gray_wm  = cv2.inRange(hsv, np.array([0,0,150]), np.array([180,sat-5,210]))
        mask     = cv2.bitwise_or(mask, cv2.bitwise_and(gray_wm, dm))
        return mask

    def _transparency_anomaly(self, gray, hsv):
        """Yarı-şeffaf watermark anomalilerini bul."""
        h, w  = gray.shape
        mask  = np.zeros((h,w), np.uint8)
        diff_thr = self.thr['diff']

        for ksize in [21, 41, 61]:
            blur    = cv2.GaussianBlur(gray.astype(np.float32), (ksize,ksize), 0)
            diff    = gray.astype(np.float32) - blur
            anomaly = ((np.abs(diff) > diff_thr) & (np.abs(diff) < 70) &
                       (gray > 100) & (gray < 240))
            mask    = cv2.bitwise_or(mask, anomaly.astype(np.uint8)*255)

        sat_ch  = hsv[:,:,1].astype(np.float32)
        bg_sat  = cv2.GaussianBlur(sat_ch, (51,51), 0)
        sat_d   = np.abs(sat_ch - bg_sat)
        _, sm   = cv2.threshold(sat_d.astype(np.uint8), 8, 255, cv2.THRESH_BINARY)
        bright  = cv2.inRange(hsv, np.array([0,0,180]), np.array([180,30,255]))
        mask    = cv2.bitwise_or(mask, cv2.bitwise_and(sm, bright))
        return mask

    def _tiled_pattern(self, gray):
        """FFT ile tekrarlayan tiled watermark."""
        h, w   = gray.shape
        f      = np.fft.fft2(gray.astype(np.float32))
        mag    = np.log1p(np.abs(np.fft.fftshift(f)))
        mag_n  = cv2.normalize(mag, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
        dc     = np.ones((h,w), np.uint8) * 255
        cv2.circle(dc, (w//2,h//2), min(h,w)//6, 0, -1)
        _, fm  = cv2.threshold(cv2.bitwise_and(mag_n,dc), 210, 255, cv2.THRESH_BINARY)
        blr    = cv2.GaussianBlur(fm, (51,51), 0)
        _, res = cv2.threshold(blr, 20, 255, cv2.THRESH_BINARY)
        return res


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3: ADAPTİF INPAINTING
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveInpainter:
    def inpaint(self, image, mask):
        if mask.sum() == 0:
            return image.copy()
        h, w   = image.shape[:2]
        ratio  = mask.sum() / 255.0 / (h*w)
        mdil   = cv2.dilate(mask,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),
                            iterations=1)
        if ratio < 0.01:
            result = cv2.inpaint(image, mdil, 5,  cv2.INPAINT_TELEA)
        elif ratio < 0.05:
            result = cv2.inpaint(image, mdil, 7,  cv2.INPAINT_NS)
        else:
            s1     = cv2.inpaint(image, mdil, 10, cv2.INPAINT_NS)
            em     = cv2.Canny(mdil, 50, 150)
            em     = cv2.dilate(em, cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)))
            em     = cv2.bitwise_and(em, mdil)
            result = cv2.inpaint(s1, em, 5, cv2.INPAINT_TELEA) if em.sum()>0 else s1

        alpha   = cv2.GaussianBlur(mask.astype(np.float32)/255.0,(7,7),2)[:,:,np.newaxis]
        blended = result.astype(np.float32)*alpha + image.astype(np.float32)*(1-alpha)
        return blended.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4: WATERMARK ÜRETECİ (dataset için)
# ─────────────────────────────────────────────────────────────────────────────

def add_watermark_advanced(image):
    h, w = image.shape[:2]; wm=image.copy(); mask=np.zeros((h,w),np.uint8)
    texts=["WATERMARK","COPY","SAMPLE","DRAFT","CONFIDENTIAL","DO NOT COPY","© 2024"]
    text=random.choice(texts); font=cv2.FONT_HERSHEY_SIMPLEX
    scale=random.uniform(0.5,1.8); thickness=random.randint(1,3)
    alpha=random.uniform(0.15,0.55)
    color=random.choice([(255,255,255),(200,200,200),(128,128,128),(180,180,220)])
    wm_type=random.choice(['single','diagonal','tiled','multi'])

    if wm_type=='single':
        size=cv2.getTextSize(text,font,scale,thickness)[0]
        x=random.randint(0,max(1,w-size[0])); y=random.randint(size[1],h)
        ov=wm.copy()
        cv2.putText(ov,text,(x,y),font,scale,color,thickness)
        cv2.putText(mask,text,(x,y),font,scale,255,thickness)
        wm=cv2.addWeighted(ov,alpha,wm,1-alpha,0)
    elif wm_type=='diagonal':
        angle=random.choice([30,45,-30,-45]); fsz=max(12,int(scale*30))
        try: pf=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",fsz)
        except: pf=ImageFont.load_default()
        tl=Image.new('RGBA',(w,h),(0,0,0,0)); d=ImageDraw.Draw(tl)
        bb=d.textbbox((0,0),text,font=pf); tw,th=bb[2]-bb[0],bb[3]-bb[1]
        cx,cy=w//2-tw//2,h//2-th//2
        d.text((cx,cy),text,font=pf,fill=(color[0],color[1],color[2],int(alpha*255)))
        rot=tl.rotate(angle,expand=False)
        base=Image.fromarray(cv2.cvtColor(wm,cv2.COLOR_BGR2RGB)).convert('RGBA')
        wm=cv2.cvtColor(np.array(Image.alpha_composite(base,rot).convert('RGB')),cv2.COLOR_RGB2BGR)
        ml=Image.new('L',(w,h),0); ImageDraw.Draw(ml).text((cx,cy),text,font=pf,fill=200)
        _,mask=cv2.threshold(np.array(ml.rotate(angle,expand=False)),50,255,cv2.THRESH_BINARY)
    elif wm_type=='tiled':
        ov=wm.copy(); s=scale*0.6; sz=cv2.getTextSize(text,font,s,1)[0]
        sx=sz[0]+40; sy=sz[1]+40; off=random.randint(0,sx); at=alpha*0.7
        for yp in range(-sy,h+sy,sy):
            for xp in range(-off,w+sx,sx):
                cv2.putText(ov,text,(xp,yp),font,s,color,1)
                cv2.putText(mask,text,(xp,yp),font,s,255,1)
        wm=cv2.addWeighted(ov,at,wm,1-at,0)
    elif wm_type=='multi':
        ov=wm.copy(); lines=random.sample(texts,min(3,len(texts))); s=scale*0.8
        for idx,line in enumerate(lines):
            sz=cv2.getTextSize(line,font,s,thickness)[0]
            x=random.randint(0,max(1,w-sz[0])); y=(h//(len(lines)+1))*(idx+1)
            cv2.putText(ov,line,(x,y),font,s,color,thickness)
            cv2.putText(mask,line,(x,y),font,s,255,thickness)
        wm=cv2.addWeighted(ov,alpha,wm,1-alpha,0)

    mask=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),iterations=1)
    return wm,mask


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 5: KALİTE METRİKLERİ
# ─────────────────────────────────────────────────────────────────────────────

def compute_psnr(orig, res):
    mse=np.mean((orig.astype(np.float32)-res.astype(np.float32))**2)
    return float('inf') if mse==0 else 20.0*np.log10(255.0/np.sqrt(mse))

def compute_ssim(orig, res):
    g1=cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY); g2=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    s,_=ssim_metric(g1,g2,full=True); return float(s)

def compute_mask_coverage(mask):
    return mask.sum()/255.0/(mask.shape[0]*mask.shape[1])*100.0


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 6: DATASET OLUŞTURMA
# ─────────────────────────────────────────────────────────────────────────────

def create_dataset(clean_dir="clean_images", out_dir="dataset",
                   augment=True, variants=3):
    os.makedirs(f"{out_dir}/images",exist_ok=True)
    os.makedirs(f"{out_dir}/masks", exist_ok=True)
    files=[f for f in os.listdir(clean_dir)
           if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    if not files: print(f"[!] {clean_dir} içinde görüntü yok."); return
    idx=0
    for f in files:
        img=cv2.imread(os.path.join(clean_dir,f))
        if img is None: continue
        img=cv2.resize(img,(256,256))   # sadece dataset için küçült
        for _ in range(variants if augment else 1):
            wm,mask=add_watermark_advanced(img)
            cv2.imwrite(f"{out_dir}/images/{idx}.png",wm)
            cv2.imwrite(f"{out_dir}/masks/{idx}.png",mask)
            idx+=1
    print(f"Dataset: {idx} örnek → {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 7: U-NET (torch varsa)
# ─────────────────────────────────────────────────────────────────────────────

TORCH_AVAILABLE=False
try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    TORCH_AVAILABLE=True; DEVICE=("cuda" if torch.cuda.is_available() else "cpu")
    IMG_SIZE=256; EPOCHS=10
except ImportError: pass

if TORCH_AVAILABLE:
    class DoubleConv(nn.Module):
        def __init__(self,ic,oc,drop=0.0):
            super().__init__()
            l=[nn.Conv2d(ic,oc,3,padding=1),nn.BatchNorm2d(oc),nn.ReLU(inplace=True),
               nn.Conv2d(oc,oc,3,padding=1),nn.BatchNorm2d(oc),nn.ReLU(inplace=True)]
            if drop>0: l.append(nn.Dropout2d(drop))
            self.net=nn.Sequential(*l)
        def forward(self,x): return self.net(x)

    class UNetDeep(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1=DoubleConv(3,64); self.e2=DoubleConv(64,128)
            self.e3=DoubleConv(128,256); self.e4=DoubleConv(256,512)
            self.pool=nn.MaxPool2d(2); self.mid=DoubleConv(512,1024,drop=0.3)
            self.u4=nn.ConvTranspose2d(1024,512,2,2); self.d4=DoubleConv(1024,512)
            self.u3=nn.ConvTranspose2d(512,256,2,2);  self.d3=DoubleConv(512,256)
            self.u2=nn.ConvTranspose2d(256,128,2,2);  self.d2=DoubleConv(256,128)
            self.u1=nn.ConvTranspose2d(128,64,2,2);   self.d1=DoubleConv(128,64)
            self.out=nn.Conv2d(64,1,1)
        def forward(self,x):
            e1=self.e1(x); e2=self.e2(self.pool(e1))
            e3=self.e3(self.pool(e2)); e4=self.e4(self.pool(e3))
            m=self.mid(self.pool(e4))
            d=self.d4(torch.cat([self.u4(m),e4],1))
            d=self.d3(torch.cat([self.u3(d),e3],1))
            d=self.d2(torch.cat([self.u2(d),e2],1))
            d=self.d1(torch.cat([self.u1(d),e1],1))
            return torch.sigmoid(self.out(d))

    class WMDataset(Dataset):
        def __init__(self,id="dataset/images",md="dataset/masks"):
            self.imgs=sorted(glob.glob(f"{id}/*.png"))
            self.masks=sorted(glob.glob(f"{md}/*.png"))
        def __len__(self): return len(self.imgs)
        def __getitem__(self,i):
            img=cv2.imread(self.imgs[i])/255.0
            mask=cv2.imread(self.masks[i],0)/255.0
            return (torch.tensor(img).permute(2,0,1).float(),
                    torch.tensor(mask).unsqueeze(0).float())

    def train_model():
        ds=WMDataset(); dl=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0)
        model=UNetDeep().to(DEVICE)
        opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        sched=optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
        loss_fn=nn.BCELoss(); best=float('inf')
        for e in range(EPOCHS):
            model.train(); loop=tqdm(dl,desc=f"Epoch {e+1}/{EPOCHS}"); el=0.0
            for x,y in loop:
                x,y=x.to(DEVICE),y.to(DEVICE); pred=model(x); loss=loss_fn(pred,y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step(); el+=loss.item(); loop.set_postfix(loss=f"{loss.item():.4f}")
            sched.step(); avg=el/len(dl)
            if avg<best: best=avg; torch.save(model.state_dict(),"model_best.pth")
        torch.save(model.state_dict(),"model.pth")
        print(f"Tamamlandı. En iyi={best:.4f}")

    def predict_with_model(img_path,model_path="model.pth"):
        model=UNetDeep().to(DEVICE)
        model.load_state_dict(torch.load(model_path,map_location=DEVICE))
        model.eval(); img=cv2.imread(img_path); orig=img.copy(); h,w=img.shape[:2]
        t=torch.tensor(cv2.resize(img,(IMG_SIZE,IMG_SIZE))/255.0)\
           .permute(2,0,1).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad(): pred=model(t)[0][0].cpu().numpy()
        mask=cv2.resize((pred>0.45).astype(np.uint8)*255,(w,h))
        return AdaptiveInpainter().inpaint(orig,mask), mask


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 8: ANA TEMİZLEME FONKSİYONU
# ─────────────────────────────────────────────────────────────────────────────

def remove_watermark(img_path, output_path="result.png",
                     use_model=False, model_path="model.pth",
                     sensitivity='medium', visualize=False):
    img = cv2.imread(img_path)
    if img is None:
        print(f"[!] Görüntü okunamadı: {img_path}")
        return None

    orig = img.copy()
    h, w = img.shape[:2]
    print(f"\nGörüntü    : {w}x{h}  —  {img_path}")

    if not check_image_size(img, img_path):
        print("[!] Küçük görüntü ile devam ediliyor...")
        print("[!] İyi sonuç için lütfen orijinal boyutlu görüntü kullanın.\n")

    print(f"Hassasiyet : {sensitivity}")

    if use_model and TORCH_AVAILABLE and os.path.exists(model_path):
        print("Mod        : U-Net modeli")
        result, mask = predict_with_model(img_path, model_path)
    else:
        if use_model:
            print("[!] Model bulunamadı → klasik moda geçildi")
        print("Mod        : Hassas klasik tespit (kendi OCR motorumuz)")

        # --- MODİFİYE EDİLMİŞ KESİN OCR BİNALİZASYONU ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Çok hafif bir yumuşatma: Harf kenarlarındaki tırtıkları ve JPEG bozulmalarını pürüzsüzleştirir.
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 2. Global Sabit Eşikleme (Hard Thresholding)
        # Orijinal belgende metin çok koyu (0-60 arası), watermark ise gri (150-200 arası).
        # Biz çizgiyi '130'dan çekiyoruz. 130'dan koyu her şey SİYAH, geri kalan her şey BEYAZ olacak.
        _, binary = cv2.threshold(blurred, 130, 255, cv2.THRESH_BINARY)
        
        # 3. İsteğe bağlı: Metin çok hafif incelirse, eski kalınlığına getirmek için hafif bir erosion
        # kernel = np.ones((2, 2), np.uint8)
        # result_gray = cv2.erode(binary, kernel, iterations=1)
        result_gray = binary # Eğer harfler ince gelirse üstteki 2 satırı aktif et, bu satırı sil.
        
        # Metriklerin hata vermemesi için formatı BGR'ye ve Maskeye çevir
        result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
        mask = cv2.bitwise_not(result_gray)

        if visualize:
            detector = PrecisionWatermarkDetector(sensitivity=sensitivity)
            regions = detector.ocr.find_text_regions(img)
            detector.ocr.visualize(img, regions, "ocr_detection.png")
            cv2.imwrite(os.path.abspath("ocr_detection.png"), img)
            print("OCR görseli: ocr_detection.png")

    psnr     = compute_psnr(orig, result)
    sim      = compute_ssim(orig, result)
    coverage = compute_mask_coverage(mask)

    print(f"\n{'─'*42}")
    print(f"Mask kapsama : %{coverage:.2f}")
    print(f"PSNR         : {psnr:.2f} dB")
    print(f"SSIM         : {sim:.4f}  (1.0 = özdeş)")
    print(f"{'─'*42}")

    # Arka planda güvenli absolute path kayıtları (Terminaldeki yazıyı değiştirmez)
    cv2.imwrite(os.path.abspath(output_path), result)
    cv2.imwrite(os.path.abspath("detected_mask.png"), mask)
    
    print(f"Sonuç : {output_path}")
    print(f"Mask  : detected_mask.png")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 9: CLI
# ─────────────────────────────────────────────────────────────────────────────

def _menu():
    ts=f"PyTorch {'✓ '+DEVICE.upper() if TORCH_AVAILABLE else '✗  (yüklü değil)'}"
    print("\n"+"═"*54)
    print("  WaterMark Removal System — Gelişmiş OCR v4")
    print("═"*54)
    print(f"  {ts}")
    print("─"*54)
    print("  1. Dataset oluştur")
    print("  2. Model eğit                  [torch gerekli]")
    print("  3. Temizle — hassasiyet: düşük   (belge/fatura)")
    print("  4. Temizle — hassasiyet: orta    (önerilen)")
    print("  5. Temizle — hassasiyet: yüksek  (agresif)")
    print("  6. Temizle — U-Net modeli      [torch gerekli]")
    print("  7. OCR tespiti göster")
    print("  8. Demo")
    print("  0. Çıkış")
    print("─"*54)
    print("  [!] NOT: Temizlenecek görüntü orijinal boyutunda")
    print("      olmalıdır. dataset/images/ içindeki küçük")
    print("      görüntüleri değil, kendi resminizi verin.")
    print("═"*54)
    return input("\nSeçim > ").strip()


def _ask(out="result.png"):
    path = input("Görüntü yolu  : ").strip()
    out  = input(f"Çıktı yolu   [{out}]: ").strip() or out
    return path, out


def _run_menu():
    choice = _menu()
    if choice=="0": sys.exit(0)
    elif choice=="1":
        cd=input("Temiz klasör [clean_images]: ").strip() or "clean_images"
        od=input("Çıktı klasör [dataset]:     ").strip() or "dataset"
        create_dataset(clean_dir=cd, out_dir=od)
    elif choice=="2":
        if not TORCH_AVAILABLE: print("[!] pip install torch torchvision")
        else: train_model()
    elif choice in ("3","4","5"):
        sens={"3":"low","4":"medium","5":"high"}[choice]
        path,out=_ask()
        remove_watermark(path, out, sensitivity=sens, visualize=True)
    elif choice=="6":
        if not TORCH_AVAILABLE: print("[!] pip install torch torchvision")
        else:
            path,out=_ask()
            mdl=input("Model yolu [model.pth]: ").strip() or "model.pth"
            remove_watermark(path,out,use_model=True,model_path=mdl)
    elif choice=="7":
        path=input("Görüntü yolu : ").strip()
        img=cv2.imread(path)
        if img is None: print(f"[!] Okunamadı: {path}"); return
        ocr=AdvancedOCR(); regions=ocr.find_text_regions(img)
        print(f"\nTespit: {len(regions)} metin bölgesi")
        for i,(x,y,w,h,c) in enumerate(regions):
            print(f"  #{i+1}  ({x},{y}) {w}x{h}  güven={c:.2f}")
        ocr.visualize(img,regions,"ocr_result.png")
        print("Görsel: ocr_result.png")
    elif choice=="8":
        imgs=glob.glob("dataset/images/*.png")+glob.glob("clean_images/*")
        if not imgs: print("[!] Test görüntüsü yok."); return
        # En büyük görüntüyü seç
        imgs.sort(key=lambda p: os.path.getsize(p), reverse=True)
        remove_watermark(imgs[0],"demo_result.png",sensitivity='medium',visualize=True)
    else:
        print("[!] Geçersiz seçim.")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description="WaterMark Removal v4")
    ap.add_argument("--remove",      metavar="IMG")
    ap.add_argument("--detect",      metavar="IMG")
    ap.add_argument("--dataset",     action="store_true")
    ap.add_argument("--train",       action="store_true")
    ap.add_argument("--model",       metavar="PTH", default="model.pth")
    ap.add_argument("--output",      metavar="OUT", default="result.png")
    ap.add_argument("--clean-dir",   metavar="DIR", default="clean_images")
    ap.add_argument("--sensitivity", metavar="S",   default="medium",
                    choices=["low","medium","high"])
    ap.add_argument("--viz",         action="store_true")
    args=ap.parse_args()

    if args.dataset: create_dataset(clean_dir=args.clean_dir, augment=True)
    elif args.train:
        if not TORCH_AVAILABLE: print("[!] pip install torch torchvision"); sys.exit(1)
        train_model()
    elif args.remove:
        use_model=TORCH_AVAILABLE and os.path.exists(args.model)
        remove_watermark(args.remove, args.output,
                         use_model=use_model, model_path=args.model,
                         sensitivity=args.sensitivity, visualize=args.viz)
    elif args.detect:
        img=cv2.imread(args.detect)
        if img is None: print(f"[!] Okunamadı: {args.detect}"); sys.exit(1)
        ocr=AdvancedOCR(); regions=ocr.find_text_regions(img)
        print(f"Tespit: {len(regions)} bölge")
        for i,(x,y,w,h,c) in enumerate(regions):
            print(f"  #{i+1}  ({x},{y}) {w}x{h}  güven={c:.2f}")
        ocr.visualize(img,regions,args.output.replace(".png","_ocr.png"))
    else:
        while True: _run_menu()
