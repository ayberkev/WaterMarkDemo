import os
import cv2
import numpy as np
from database import init_db, save_image, update_result


def check_image_size(img):
    h, w = img.shape[:2]
    return h >= 64 and w >= 64


class HybridWatermarkRemover:
    """
    General-purpose watermark remover using hybrid mask generation.

    Goal:
    - Work reasonably on documents and generic images
    - Detect semi-transparent text/logo-like overlays
    - Produce a real mask and clean result for the frontend
    """

    def __init__(self, sensitivity: str = 'medium'):
        sensitivity = (sensitivity or 'medium').lower()
        if sensitivity not in {'low', 'medium', 'high'}:
            sensitivity = 'medium'
        self.sensitivity = sensitivity
        self.config = {
            'low': {
                'vote_threshold': 3,
                'percentile_local': 92,
                'percentile_hat': 92,
                'percentile_color': 93,
                'dilate_iter': 1,
                'kernel_base': 11,
                'inpaint_radius': 3,
                'min_area': 40,
                'max_coverage': 0.18,
            },
            'medium': {
                'vote_threshold': 2,
                'percentile_local': 88,
                'percentile_hat': 89,
                'percentile_color': 90,
                'dilate_iter': 1,
                'kernel_base': 13,
                'inpaint_radius': 4,
                'min_area': 28,
                'max_coverage': 0.26,
            },
            'high': {
                'vote_threshold': 2,
                'percentile_local': 83,
                'percentile_hat': 85,
                'percentile_color': 86,
                'dilate_iter': 2,
                'kernel_base': 15,
                'inpaint_radius': 5,
                'min_area': 18,
                'max_coverage': 0.34,
            },
        }[self.sensitivity]

    def _odd(self, value: int) -> int:
        return value if value % 2 == 1 else value + 1

    def _is_document_like(self, image: np.ndarray) -> bool:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_sat = float(np.mean(hsv[:, :, 1]))
        bright_ratio = float(np.mean(hsv[:, :, 2] > 180))
        return mean_sat < 55 and bright_ratio > 0.45

    def _local_residual_mask(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = self._odd(max(21, min(gray.shape[:2]) // 8))
        blurred = cv2.GaussianBlur(gray, (k, k), 0)
        residual = cv2.absdiff(gray, blurred)
        threshold = np.percentile(residual, self.config['percentile_local'])
        mask = (residual >= threshold).astype(np.uint8) * 255
        return mask, residual

    def _hat_masks(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = self._odd(self.config['kernel_base'])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        t1 = np.percentile(top_hat, self.config['percentile_hat'])
        t2 = np.percentile(black_hat, self.config['percentile_hat'])
        mask = ((top_hat >= t1) | (black_hat >= t2)).astype(np.uint8) * 255
        combined_strength = cv2.max(top_hat, black_hat)
        return mask, top_hat, combined_strength

    def _color_residual_mask(self, image: np.ndarray, document_like: bool) -> tuple[np.ndarray, np.ndarray]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        blur_s = cv2.GaussianBlur(s, (0, 0), sigmaX=7)
        blur_v = cv2.GaussianBlur(v, (0, 0), sigmaX=7)
        sat_residual = cv2.absdiff(s, blur_s)
        val_residual = cv2.absdiff(v, blur_v)

        b, g, r = cv2.split(image)
        blur_b = cv2.GaussianBlur(b, (0, 0), sigmaX=7)
        blur_g = cv2.GaussianBlur(g, (0, 0), sigmaX=7)
        blur_r = cv2.GaussianBlur(r, (0, 0), sigmaX=7)
        color_residual = cv2.max(cv2.max(cv2.absdiff(b, blur_b), cv2.absdiff(g, blur_g)), cv2.absdiff(r, blur_r))

        combined = cv2.max(color_residual, cv2.max(sat_residual, val_residual))
        threshold = np.percentile(combined, self.config['percentile_color'])
        mask = (combined >= threshold).astype(np.uint8) * 255

        if document_like:
            # On bright documents, colored or translucent overlays often have higher saturation than body text.
            doc_mask = (((s > 30) & (v > 80)) | (val_residual > np.percentile(val_residual, 90))).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, doc_mask)

        return mask, combined

    def _edge_mask(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        edges = cv2.Canny(gray, 60, 160)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grown = cv2.dilate(edges, k, iterations=1)
        return grown, edges

    def _refine_mask(self, raw_mask: np.ndarray, score_map: np.ndarray, document_like: bool = False) -> np.ndarray:
        mask = raw_mask.copy()
        close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k, iterations=1)
        if self.config['dilate_iter'] > 0:
            mask = cv2.dilate(mask, open_k, iterations=self.config['dilate_iter'])

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        refined = np.zeros_like(mask)
        img_area = mask.shape[0] * mask.shape[1]

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.config['min_area']:
                continue
            coverage = area / float(img_area)
            if coverage > 0.18:
                # Almost never a valid watermark component on its own.
                continue
            region_mask = (labels == i)
            region_score = float(np.mean(score_map[region_mask])) if np.any(region_mask) else 0.0
            aspect = max(w, h) / max(1.0, min(w, h))

            if document_like:
                # Suppress slender horizontal/vertical body text lines on documents.
                if (aspect > 6.5 and min(w, h) < 24) or (h < 14 and w > 90):
                    continue

            keep = region_score > 48 or area > 260 or (aspect > 2.5 and area > 90)
            if keep:
                refined[region_mask] = 255

        # Prevent overly aggressive masks.
        coverage = float(np.mean(refined > 0))
        if coverage > self.config['max_coverage']:
            # Keep only stronger areas based on score map.
            score_threshold = np.percentile(score_map[refined > 0], 55) if np.any(refined > 0) else 255
            refined = np.where((refined > 0) & (score_map >= score_threshold), 255, 0).astype(np.uint8)
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_k, iterations=1)

        return refined

    def detect_mask(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        document_like = self._is_document_like(image)

        mask_local, residual_local = self._local_residual_mask(gray)
        mask_hat, _, residual_hat = self._hat_masks(gray)
        mask_color, residual_color = self._color_residual_mask(image, document_like)
        mask_edge, residual_edge = self._edge_mask(gray)

        score_map = cv2.normalize(
            cv2.addWeighted(residual_local.astype(np.float32), 0.35,
                            residual_hat.astype(np.float32), 0.25, 0)
            + 0.25 * residual_color.astype(np.float32)
            + 0.15 * residual_edge.astype(np.float32),
            None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        if document_like:
            # Suppress dark body text and focus on soft / translucent overlay regions.
            non_dark = (gray > 85).astype(np.uint8) * 255
            local_hat_overlap = cv2.bitwise_and(mask_local, mask_hat)
            raw_mask = cv2.bitwise_or(mask_color, local_hat_overlap)
            raw_mask = cv2.bitwise_and(raw_mask, non_dark)

            # If a region is color-like and edge-like, it is a stronger watermark candidate.
            color_edge = cv2.bitwise_and(mask_color, mask_edge)
            raw_mask = cv2.bitwise_or(raw_mask, cv2.bitwise_and(color_edge, non_dark))
        else:
            # Vote map: areas supported by multiple cues are more likely to be watermark.
            votes = np.zeros_like(gray, dtype=np.uint8)
            for m in [mask_local, mask_hat, mask_color, mask_edge]:
                votes = votes + (m > 0).astype(np.uint8)
            raw_mask = np.where(votes >= self.config['vote_threshold'], 255, 0).astype(np.uint8)

        refined = self._refine_mask(raw_mask, score_map, document_like=document_like)

        # Fallback: if mask vanished entirely, try a softer union-based mask.
        if np.count_nonzero(refined) == 0:
            union_mask = cv2.bitwise_or(mask_local, mask_hat)
            union_mask = cv2.bitwise_or(union_mask, mask_color)
            if document_like:
                union_mask = cv2.bitwise_and(union_mask, (gray > 85).astype(np.uint8) * 255)
            refined = self._refine_mask(union_mask, score_map, document_like=document_like)

        return refined

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self._is_document_like(image):
            # For invoices / forms / scanned pages, simple global document binarization often
            # preserves content better than broad inpainting.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            clean = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            return clean

        radius = self.config['inpaint_radius']
        telea = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
        ns = cv2.inpaint(image, mask, max(3, radius - 1), cv2.INPAINT_NS)

        # Blend results slightly for smoother fill.
        blended = cv2.addWeighted(telea, 0.7, ns, 0.3, 0)
        return blended



def compute_psnr(img1, img2):
    try:
        return float(cv2.PSNR(img1, img2))
    except Exception:
        return 0.0



def compute_ssim(img1, img2):
    try:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        mu1 = cv2.GaussianBlur(gray1.astype(np.float32), (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(gray2.astype(np.float32), (11, 11), 1.5)
        sigma1_sq = cv2.GaussianBlur(gray1.astype(np.float32) ** 2, (11, 11), 1.5) - mu1 ** 2
        sigma2_sq = cv2.GaussianBlur(gray2.astype(np.float32) ** 2, (11, 11), 1.5) - mu2 ** 2
        sigma12 = cv2.GaussianBlur((gray1.astype(np.float32) * gray2.astype(np.float32)), (11, 11), 1.5) - mu1 * mu2
        c1 = 6.5025
        c2 = 58.5225
        ssim_map = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-8)
        return float(np.clip(ssim_map.mean(), 0, 1))
    except Exception:
        return 0.0



def compute_mask_coverage(mask):
    return float(np.mean(mask > 0) * 100.0)



def remove_watermark(img_path, output_path='result.png', sensitivity='medium', mask_output_path=None):
    image_id = save_image(img_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f'[!] Görüntü okunamadı: {img_path}')
        return None, None

    if not check_image_size(img):
        print('[!] Görüntü boyutu çok küçük, sonuç kalitesi sınırlı olabilir.')

    remover = HybridWatermarkRemover(sensitivity=sensitivity)
    mask = remover.detect_mask(img)
    result = remover.inpaint(img, mask)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(os.path.abspath(output_path), result)

    if mask_output_path is None:
        root, ext = os.path.splitext(output_path)
        mask_output_path = f'{root}_mask.png'
    os.makedirs(os.path.dirname(os.path.abspath(mask_output_path)), exist_ok=True)
    cv2.imwrite(os.path.abspath(mask_output_path), mask)

    psnr = compute_psnr(img, result)
    ssim = compute_ssim(img, result)
    coverage = compute_mask_coverage(mask)
    update_result(image_id, output_path, ssim)

    print(f'Görüntü      : {img_path}')
    print(f'Hassasiyet   : {sensitivity}')
    print(f'Mask kapsama : %{coverage:.2f}')
    print(f'PSNR         : {psnr:.2f} dB')
    print(f'SSIM         : {ssim:.4f}')
    print(f'Sonuç        : {output_path}')
    print(f'Maske        : {mask_output_path}')

    return result, mask_output_path


init_db()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='General-purpose watermark remover')
    parser.add_argument('image', help='Input image path')
    parser.add_argument('--out', default='result.png', help='Output image path')
    parser.add_argument('--sensitivity', default='medium', choices=['low', 'medium', 'high'])
    args = parser.parse_args()

    remove_watermark(args.image, args.out, sensitivity=args.sensitivity)
