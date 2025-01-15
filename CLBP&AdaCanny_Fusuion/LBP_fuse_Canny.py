import cv2
import numpy as np

def calculate_local_brightness(image, kernel_size=15):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
    return blurred

def adaptive_fusion_fine_grained(original_image, canny_image, brightness_threshold=150, kernel_size=15):
    canny_image_3channel = cv2.cvtColor(canny_image, cv2.COLOR_GRAY2BGR)
    local_brightness = calculate_local_brightness(original_image, kernel_size)
    alpha_map = np.where(local_brightness > brightness_threshold, 0.6, 0.85)
    beta_map = np.where(local_brightness > brightness_threshold, 0.4, 0.15)
    original_image_float = original_image.astype(np.float32)
    canny_image_3channel_float = canny_image_3channel.astype(np.float32)
    alpha_map = alpha_map[:, :, np.newaxis].repeat(3, axis=2)
    beta_map = beta_map[:, :, np.newaxis].repeat(3, axis=2)
    output_image = (alpha_map * original_image_float + beta_map * canny_image_3channel_float).astype(np.uint8)
    return output_image