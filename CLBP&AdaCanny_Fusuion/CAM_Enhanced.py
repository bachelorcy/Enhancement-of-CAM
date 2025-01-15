from Canny import manual_canny
from LBP_feature import generate_lbp_features, fuse_rgb_lbp
from LBP_fuse_Canny import adaptive_fusion_fine_grained

import cv2
import numpy as np
from pathlib import Path
import os


def process_image(input_path, output_dir, lbp_params, fusion_weights, th_ratio, brightness_threshold=150,
                  kernel_size=15):
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    original_image = cv2.imread(input_path)
    if original_image is None:
        print(f"Failed to read image: {input_path}")
        return

    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # 生成LBP特征图，并确保其尺寸与原图相同
    lbp_image = generate_lbp_features(gray_image, lbp_params)
    lbp_image = cv2.resize(lbp_image, (gray_image.shape[1], gray_image.shape[0]))

    # 将LBP图像转换为三通道图像
    lbp_image_3ch = cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2BGR)

    fused_image = fuse_rgb_lbp(original_image, lbp_image_3ch, fusion_weights)

    canny_edges = manual_canny(fused_image, th_ratio)

    final_image = adaptive_fusion_fine_grained(fused_image, canny_edges, brightness_threshold, kernel_size)

    # 确保最终图像与原图尺寸相同
    if final_image.shape[:2] != original_image.shape[:2]:
        final_image = cv2.resize(final_image, (original_image.shape[1], original_image.shape[0]))

    output_path = os.path.join(output_dir, os.path.basename(input_path))
    cv2.imwrite(output_path, final_image)
    print(f"Processed and saved: {output_path}")


def process_datasets(datasets_path, output_root, lbp_params, fusion_weights, th_ratio):
    for dataset in datasets_path:
        for name, input_dir in dataset.items():
            output_dir = os.path.join(output_root, name)
            for filename in os.listdir(input_dir):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    print(f"Skipping non-image file: {filename}")
                    continue
                input_path = os.path.join(input_dir, filename)
                process_image(input_path, output_dir, lbp_params, fusion_weights, th_ratio)


if __name__ == '__main__':
    datasets_path = [
        {'CAMO_test': '/home/cc/dataset/Test/CAMO/Img'},
        {'COD10K_test': '/home/cc/dataset/Test/COD10K/Img'},
        {'NC4K_test': '/home/cc/dataset/Test/NC4K/Img'},
        {'CHAMELEON_test': '/home/cc/dataset/Test/CHAMELEON/Img'},
        {'COD10K_train_CAM': '/home/cc/dataset/COD10K/Train/Image'}
    ]

    output_root_directory = '/home/cc/CLBP&AdaCanny_Fusuion/output'  # 替换为实际输出图像根路径

    lbp_params = {'radius': 2, 'n_points': 16, 'method': 'uniform'}
    fusion_weights = (0.0, 0.0)  # 这里的权重只使用了一个值，因为fuse_rgb_lbp只接受一个权重
    th_ratio = 0.4  # Canny边缘检测的阈值比例

    process_datasets(datasets_path, output_root_directory, lbp_params, fusion_weights, th_ratio)