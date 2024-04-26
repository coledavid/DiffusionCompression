import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as compare_ssim
import cv2

class ImageQualityMetrics:
    """
    ImageQualityMetrics is a custom-built class that calculates the Mean Squared Error (MSE) and the Peak Signal to Noise Ratio (PSNR)
    between a reference image and a compressed image.

    ----------------------------------------------
    Parameters:

    ref_image_path: str
        The path to the reference image.

    comp_image_path: str
        The path to the compressed image.

    """
    def __init__(self, ref_image_path, comp_image_path, im_up = True):
        self.ref_image = np.array(Image.open(ref_image_path))
        if im_up:
            ImageCompressor.upscale(self, comp_image_path, 'uncompressed_image.jpg', [self.ref_image.shape[1],self.ref_image.shape[0]])
            self.comp_image = np.array(Image.open('uncompressed_image.jpg'))
        else:
            self.comp_image = np.array(Image.open(comp_image_path))

    def calculate_mse(self):
        """
        Calculates the Mean Squared Error (MSE) between the reference image and the compressed image.
        """
        err = np.sum((self.ref_image.astype("float") - self.comp_image.astype("float")) ** 2)
        err /= float(self.ref_image.shape[0] * self.ref_image.shape[1])
        return err

    def calculate_psnr(self):
        """
        Calculates the Peak Signal to Noise Ratio (PSNR) between the reference image and the compressed image.
        """
        mse = self.calculate_mse()
        if mse == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

    def calculate_ssim(self):
        """
        Calculates the Structural Similarity Index (SSIM) between the reference image and the compressed image.
        """
        img1 = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.comp_image, cv2.COLOR_BGR2GRAY)
        return compare_ssim(img1, img2)

    def calculate_msssim(self):
        """
        Calculates the Multi-Scale Structural Similarity Index (MS-SSIM) between the reference image and the compressed image.
        """
        img1 = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(self.comp_image, cv2.COLOR_BGR2GRAY)
        return compare_ssim(img1, img2, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

