import struct
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 读取BMP灰度图像，返回像素矩阵、宽度和高度
def read_bmp(filename):
    with open(filename, 'rb') as f:
        # 读取文件头 (14字节)
        file_header = f.read(14)
        # 读取信息头 (40字节)
        info_header = f.read(40)
        # 解析图像宽度和高度
        width = struct.unpack('<i', info_header[4:8])[0]
        height = struct.unpack('<i', info_header[8:12])[0]
        # 计算每行字节数（按4字节对齐）
        row_size = ((width * 8 + 31) // 32) * 4
        # 读取像素数据
        f.seek(struct.unpack('<i', file_header[10:14])[0])  # 跳到像素数据起始位置
        pixels = []
        for _ in range(height):
            row = f.read(row_size)
            # 提取有效像素数据，忽略填充字节
            pixels.extend(row[:width])
        # 转换为二维数组（注意BMP像素是从下到上、从左到右存储的）
        img_array = np.array(pixels, dtype=np.uint8).reshape(height, width)[::-1, :]
    return img_array, width, height

# 3x3均值模板进行平滑滤波
def smooth_filter(img):
    h, w = img.shape
    smoothed = np.zeros_like(img, dtype=np.float32)
    # 定义3x3均值模板
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    # 处理图像内部像素
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 提取3x3邻域
            neighborhood = img[i - 1:i + 2, j - 1:j + 2].astype(np.float32)
            # 应用模板卷积
            smoothed[i, j] = np.sum(neighborhood * kernel)

    # 处理边缘像素（原值复用）
    smoothed[0, :] = img[0, :]
    smoothed[-1, :] = img[-1, :]
    smoothed[:, 0] = img[:, 0]
    smoothed[:, -1] = img[:, -1]

    return smoothed.astype(np.uint8)

# 拉普拉斯模板进行锐化滤波
def sharpen_filter(img):
    h, w = img.shape
    sharpened = np.zeros_like(img, dtype=np.float32)
    # 定义拉普拉斯锐化模板
    kernel = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # 处理图像内部像素
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # 提取3x3邻域
            neighborhood = img[i - 1:i + 2, j - 1:j + 2].astype(np.float32)
            # 应用模板卷积
            sharpened[i, j] = np.sum(neighborhood * kernel)

    # 限制像素值在0-255范围内
    sharpened = np.clip(sharpened, 0, 255)

    # 处理边缘像素
    sharpened[0, :] = img[0, :]
    sharpened[-1, :] = img[-1, :]
    sharpened[:, 0] = img[:, 0]
    sharpened[:, -1] = img[:, -1]

    return sharpened.astype(np.uint8)


# PIL库保存图像
def save_bmp(img_array, filename):
    img = Image.fromarray(img_array, mode='L')  # 'L'表示灰度图
    img.save(filename)
# 绘制三个图像对比图
def plot_images(original, smoothed, sharpened,save_path):
    plt.figure(figsize=(15, 5))
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    # 平滑图像
    plt.subplot(1, 3, 2)
    plt.imshow(smoothed, cmap='gray')
    plt.title('Smoothed Image (3x3 Mean)')
    plt.axis('off')
    # 锐化图像
    plt.subplot(1, 3, 3)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened Image (Laplacian)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存至: {save_path}")
    plt.show()
    plt.close()

def main():
    input_filename = r"D:\郑家浩\科研综述\PythonProject\标准测试图片\testpat.bmp" #
    output_path = r'D:\郑家浩\科研综述\PythonProject\train'  # 定义输出
    try:
        # 提取文件名和扩展名
        base_name = os.path.basename(input_filename)
        file_name, file_ext = os.path.splitext(base_name)
        print(f"Processing file: {base_name}")

        # 读取BMP图像
        print(f"Reading {input_filename}")
        img_array, width, height = read_bmp(input_filename)
        print(f"Image loaded: {width}x{height} pixels")

        # 文件名生成
        output_original = os.path.join(output_path, f"{file_name}_original{file_ext}")
        output_smoothed = os.path.join(output_path, f"{file_name}_smoothed{file_ext}")
        output_sharpened = os.path.join(output_path, f"{file_name}_sharpened{file_ext}")
        output_comparison = os.path.join(output_path, f"{file_name}_compared.jpg")

        # 保存原始图像
        save_bmp(img_array, output_original)

        # 应用平滑滤波
        smoothed_img = smooth_filter(img_array)
        print(f"平滑处理图已保存至: {output_comparison}")
        save_bmp(smoothed_img, output_smoothed)

        # 应用锐化滤波
        sharpened_img = sharpen_filter(img_array)
        print(f"锐化图已保存至: {output_comparison}")
        save_bmp(sharpened_img, output_sharpened)

        # 绘制对比图
        plot_images(img_array, smoothed_img, sharpened_img, output_comparison)

        print("All processing completed successfully!")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
