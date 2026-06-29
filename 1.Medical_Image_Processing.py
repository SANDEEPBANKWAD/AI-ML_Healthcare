# 1. Install Required Libraries
# pip install opencv-python scikit-image pydicom matplotlib

# 2. Complete Python Code
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from skimage import exposure, filters

# -----------------------------
# 1. Load Medical Image (DICOM)
# -----------------------------
def load_dicom_image(path):
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array.astype(np.float32)

    # Normalize to 0–255 for visualization
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image.astype(np.uint8)

# -----------------------------
# 2. Noise Removal
# -----------------------------
def remove_noise(image):
    # Gaussian filter
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)

    # Median filter (good for salt-pepper noise)
    median = cv2.medianBlur(image, 5)

    return gaussian, median

# -----------------------------
# 3. Contrast Enhancement
# -----------------------------
def enhance_contrast(image):
    # Histogram Equalization
    hist_eq = cv2.equalizeHist(image)

    # CLAHE (better for medical images)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(image)

    return hist_eq, clahe_img

# -----------------------------
# 4. Edge / Feature Enhancement
# -----------------------------
def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

# -----------------------------
# 5. Resize Image
# -----------------------------
def resize_image(image, size=(256, 256)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# -----------------------------
# 6. Full Pipeline
# -----------------------------
def preprocess_pipeline(dicom_path):
    image = load_dicom_image(dicom_path)

    gaussian, median = remove_noise(image)
    hist_eq, clahe = enhance_contrast(gaussian)
    sharp = sharpen_image(clahe)
    resized = resize_image(sharp)

    return {
        "original": image,
        "gaussian": gaussian,
        "median": median,
        "hist_eq": hist_eq,
        "clahe": clahe,
        "sharpened": sharp,
        "resized": resized
    }

# -----------------------------
# 7. Visualization
# -----------------------------
def show_results(results):
    plt.figure(figsize=(12, 8))

    for i, (key, img) in enumerate(results.items()):
        plt.subplot(3, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(key)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    path = "sample.dcm"  # Replace with your DICOM file
    results = preprocess_pipeline(path)
    show_results(results)

