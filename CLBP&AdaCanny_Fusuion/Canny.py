import cv2
import numpy as np

def adapt_gaussian_blur(image, window_size=3):
    local_std = cv2.GaussianBlur(image, (window_size, window_size), 0) - \
                cv2.GaussianBlur(image, (window_size * 3, window_size * 3), 0)
    noise_std = np.mean(local_std)
    kernel_size = {True: {True: 3, False: 5}, False: {True: 7, False: 9}}[noise_std < 15][noise_std < 30]
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred

def compute_sobel(gray_image):
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    return gx, gy

def non_max_suppression(mag, angle):
    rows, cols = mag.shape
    non_max_image = np.zeros_like(mag, dtype=np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            g_angle = angle[i, j]
            K_mag = mag[i, j]
            if (337.5 < g_angle or g_angle <= 22.5):  # 水平方向
                neighbors = [mag[i, j - 1], mag[i, j + 1]]
            elif 22.5 < g_angle <= 67.5:  # +45度方向
                neighbors = [mag[i - 1, j + 1], mag[i + 1, j - 1]]
            elif 67.5 < g_angle <= 112.5:  # 垂直方向
                neighbors = [mag[i - 1, j], mag[i + 1, j]]
            elif 112.5 < g_angle <= 157.5:  # -45度方向
                neighbors = [mag[i - 1, j - 1], mag[i + 1, j + 1]]
            else:
                neighbors = []
            if K_mag >= max(neighbors, default=0):
                non_max_image[i, j] = K_mag
    return non_max_image

def manual_canny(image, th_ratio=0.5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = adapt_gaussian_blur(gray_image)
    gx, gy = compute_sobel(gray_image)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    non_max_image = non_max_suppression(mag, angle)
    if np.isnan(non_max_image).any() or np.isinf(non_max_image).any():
        print("Warning: non_max_image contains NaN or Inf values.")
        non_max_image = np.nan_to_num(non_max_image)
    non_max_flat = ((non_max_image / (non_max_image.max() + 1e-6)) * 255).astype(np.uint8).flatten()
    th_high_val, _ = cv2.threshold(non_max_flat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_high = int(th_high_val)
    th_low = int(th_ratio * th_high)
    edges = cv2.Canny(gray_image.astype(np.uint8), th_low, th_high)
    return edges