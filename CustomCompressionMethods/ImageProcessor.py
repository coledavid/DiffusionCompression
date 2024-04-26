import os
import pandas as pd
import matplotlib.pyplot as plt

class ImageProcessor:
    """
    ImageProcessor is a custom-built class that processes a directory of images by compressing them to a specified size and quality then analyses them utilizing the helper classes ImageCompressor and ImageQualityMetrics.

    ----------------------------------------------
    Parameters:

    directory_path: str
        The path to the directory containing the images to be processed.

    compressed_path: str
        The path to the directory where the compressed images will be saved.
    """
    def __init__(self, directory_path, compressed_path):
        self.directory_path = directory_path
        self.compressed_path = compressed_path
        self.image_files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    def process_images(self, width, height, compression_type, quality):
        """
        Processes the images in the directory by compressing them to the specified size and quality, then calculates the Mean Squared Error (MSE), Peak Signal to Noise Ratio (PSNR), Structural Similarity Index (SSIM), and Multi-Scale Structural Similarity Index (MS-SSIM) between the reference image and the compressed image.

        ----------------------------------------------
        Parameters:

        width: int
            The new width of the image.

        height: int
            The new height of the image.

        compression_type: str
            The compression format to save the image in.
            Example: 'JPEG', 'PNG', 'WEBP'

        quality: int
            The quality of the compressed image.
        """
        results = []
        for image_file in self.image_files:
            ref_image_path = os.path.join(self.directory_path, image_file)
            comp_image_path = os.path.join(self.compressed_path, 'compressed_' + image_file)
            compressor = ImageCompressor(width, height)
            metrics = compressor.pipeline(ref_image_path, comp_image_path, width, height, compression_type, quality)
            results.append([image_file] + metrics + [compression_type, quality])
        self.df = pd.DataFrame(results, columns=['Image', 'MSE', 'PSNR', 'SSIM', 'MS-SSIM', 'Size', 'Compression Type', 'Quality'])

    def save_to_csv(self, csv_file_path):
        """
        Saves the processed image data to a CSV file.

        ----------------------------------------------
        Parameters:

        csv_file_path: str
            The path to save the CSV file.
        """
        self.df.to_csv(csv_file_path, index=False)

    def plot_metrics(self, csv_file_path):
        """
        Plots the metrics for each image in the CSV file.

        ----------------------------------------------

        Parameters:

        csv_file_path: str
            The path to the CSV file containing the image data.
        """
        df = pd.read_csv(csv_file_path)
        compression_types = df['Compression Type'].unique()

        for compression_type in compression_types:
            df_type = df[df['Compression Type'] == compression_type]

            plt.figure(figsize=(10, 6))
            plt.plot(df_type['MSE'], label='MSE')
            plt.plot(df_type['PSNR'], label='PSNR')
            plt.plot(df_type['SSIM'], label='SSIM')
            plt.plot(df_type['MS-SSIM'], label='MS-SSIM')
            plt.plot(df_type['Size'], label='Size')
            plt.title(f'Metrics for {compression_type} Compression')
            plt.xlabel('Image')
            plt.ylabel('Value')
            plt.legend()
            plt.show()

    def clear_data(self, csv_file_path):
        self.df = pd.DataFrame()
        self.df.to_csv(csv_file_path, index=False)