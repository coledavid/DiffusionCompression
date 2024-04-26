class ImageCompressor:
    """
    Image Compressor is a custom-built class that compresses an image by resizing it to any new dimensions
    and saving it with a specified compression format.

    ----------------------------------------------
    Parameters:
    width: int
        The new width of the image.

    height: int
        The new height of the image.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.new_size = (self.width, self.height)

    def compress(self, image_path, new_path, compression_type, quality=50):
        """
        Compresses an image by resizing it to the new dimensions and saving it with the specified compression format.

        ----------------------------------------------
        Parameters:
        image_path: str
            The path to the image that needs to be compressed.

        new_path: str
            The path to save the compressed image.

        compression_type: str
            The compression format to save the image in.
            Example: 'JPEG', 'PNG', 'WEBP'
        """
        image = Image.open(image_path)
        resized_image = image.resize(self.new_size)
        resized_image.save(new_path, optimize=True, quality=quality, format=compression_type)
        return

    def get_size(self, image_path):
        """
        Returns the size of the image in bytes.

        ----------------------------------------------
        Parameters:
        image_path: str
            The path to the image.
        """
        return os.path.getsize(image_path)
    def upscale(self, image_path, new_path, og_size):
        """
        Uncompresses an image by resizing it to the original dimensions and saving it in the original format.

        ----------------------------------------------
        Parameters:
        image_path: str
            The path to the image that needs to be uncompressed.

        new_path: str
            The path to save the uncompressed image.

        og_size: tuple
            The original dimensions of the image.
        """
        image = Image.open(image_path)
        resized_image = image.resize(og_size)
        resized_image.save(new_path, optimize=True, quality=100)
        return

    def pipeline(self, ref_image_path, comp_image_path, width, height, compression_type, quality):
        """
        A pipeline that compresses an image, calculates the Mean Squared Error (MSE), Peak Signal to Noise Ratio (PSNR), Structural
        Similarity Index (SSIM), and Multi-Scale Structural Similarity Index (MS-SSIM) between the reference image and the compressed image.

        ----------------------------------------------
        Parameters:
        ref_image_path: str
            The path to the reference image.

        comp_image_path: str
            The path to the compressed image.

        width: int
            The new width of the image.

        height: int
            The new height of the image.

        compression_type: str
            The compression format to save the image in.
            Example: 'JPEG', 'PNG', 'WEBP'
        """
        compressor = ImageCompressor(width, height)
        compressor.compress(ref_image_path, comp_image_path, compression_type, quality)

        metrics = ImageQualityMetrics(ref_image_path, comp_image_path)
        mse = metrics.calculate_mse()
        psnr = metrics.calculate_psnr()
        ssim = metrics.calculate_ssim()
        msssim = metrics.calculate_msssim()
        image_size = compressor.get_size(comp_image_path)

        return [mse, psnr, ssim, msssim, image_size]