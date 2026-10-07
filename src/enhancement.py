import cv2
import numpy as np
from PIL import Image
import torch
from src.config import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE
)

class LeafEnhancementEngine:
    """
    Domain-Specific Image Enhancement for Agricultural Leaf Imagery.
    
    Pipeline:
      1. RGB -> CIE L*a*b* space (isolating luminance from chrominance).
      2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) on L* channel.
      3. Edge-preserving Bilateral Filter on L* channel to attenuate sensor
         noise without blurring sharp necrotic lesion margins or leaf venation.
      4. Merging enhanced L* with original a*, b* and converting back to RGB.
    """
    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid_size: tuple = CLAHE_TILE_GRID_SIZE,
        bilateral_d: int = BILATERAL_D,
        sigma_color: float = BILATERAL_SIGMA_COLOR,
        sigma_space: float = BILATERAL_SIGMA_SPACE
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.bilateral_d = bilateral_d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size
        )

    def enhance(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Enhances an RGB numpy array (H, W, 3) in uint8 [0, 255].
        Returns enhanced RGB numpy array (H, W, 3).
        """
        if not isinstance(img_rgb, np.ndarray):
            img_rgb = np.array(img_rgb)

        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        elif img_rgb.shape[2] == 4:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)

        # 1. Convert to CIE L*a*b*
        lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # 2. Apply CLAHE on L* channel
        cl = self.clahe.apply(l_channel)

        # 3. Apply edge-preserving Bilateral Filter on L* channel
        cl_denoised = cv2.bilateralFilter(
            cl,
            d=self.bilateral_d,
            sigmaColor=self.sigma_color,
            sigmaSpace=self.sigma_space
        )

        # 4. Merge back and convert to RGB
        enhanced_lab = cv2.merge((cl_denoised, a_channel, b_channel))
        enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        return enhanced_rgb

    def __call__(self, img):
        """
        Callable interface compatible with PIL Image or NumPy array.
        """
        if isinstance(img, Image.Image):
            rgb_arr = np.array(img.convert("RGB"))
            enhanced = self.enhance(rgb_arr)
            return Image.fromarray(enhanced)
        elif isinstance(img, np.ndarray):
            return self.enhance(img)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

# Global singleton engine for convenience
default_engine = LeafEnhancementEngine()

def enhance_leaf_image(img_rgb: np.ndarray) -> np.ndarray:
    return default_engine.enhance(img_rgb)
