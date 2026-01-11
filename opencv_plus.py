import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 绘制三个图像对比图
def plot_images(original, smoothed, sharpened, save_path):
    plt.figure(figsize=(15, 5))

    # 是否为彩色
    is_color = len(original.shape) == 3 and original.shape[2] == 3

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB) if is_color else original,
               cmap=None if is_color else 'gray')
    plt.title('Original Image')
    plt.axis('off')

    # 平滑图像
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB) if is_color else smoothed,
               cmap=None if is_color else 'gray')
    plt.title('Smoothed Image (3x3 Mean)')
    plt.axis('off')

    # 锐化图像
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB) if is_color else sharpened,
               cmap=None if is_color else 'gray')
    plt.title('Sharpened Image (Laplacian)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    plt.show()
    plt.close()


def main():
    input_filename = r"D:\郑家浩\科研综述\PythonProject\标准测试图片\lenna.bmp"
    output_path = r"D:\郑家浩\科研综述\PythonProject\train"

    # PIL读取图像
    pil_img = Image.open(input_filename).convert('RGB')
    img_array = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    print(f"Image loaded: {img_array.shape[1]}x{img_array.shape[0]} pixels")

    # 输出文件名
    file_name = os.path.splitext(os.path.basename(input_filename))[0]
    output_comparison = os.path.join(output_path, f"{file_name}_compared.jpg")

    # 平滑滤波
    smoothed_img = cv2.blur(img_array, (3, 3))

    # 锐化滤波
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    sharpened_img = cv2.filter2D(img_array, -1, kernel)
    sharpened_img = np.clip(sharpened_img, 0, 255).astype(np.uint8)

    # 保存
    cv2.imwrite(os.path.join(output_path, f"{file_name}_smoothed.bmp"), smoothed_img)
    cv2.imwrite(os.path.join(output_path, f"{file_name}_sharpened.bmp"), sharpened_img)

    # 对比图
    plot_images(img_array, smoothed_img, sharpened_img, output_comparison)

if __name__ == "__main__":
    main()