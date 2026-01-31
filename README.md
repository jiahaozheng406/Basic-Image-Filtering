# Basic-Image-Filtering


**Python 图像滤波算法--南京邮电大学数据结构与算法实践周**

**项目介绍** 这是一个基于 Python 编写的数字图像处理（计算机视觉）项目，专注于空间域的**平滑（去噪）**与**锐化（边缘增强）**。两种不同的实现方式——手写算法与调用库——来展示图像处理的底层数学原理与实际工程应用的区别。

**功能概览** 在**底层实现端** (`manual_filtering.py`)，代码完全脱离了图像处理库的核心功能。它利用 `struct` 模块手动解析 BMP 文件头的二进制数据，并通过嵌套循环**手写实现**了 3x3 均值滤波卷积 和 拉普拉斯锐化算子，直观展示了像素级的数学运算过程。
而在**工程应用端** (`opencv_filtering.py`)，项目利用 OpenCV (`cv2`) 库的高效算法处理图像。它不仅支持彩色/灰度图的自动识别与转换，增强了代码的鲁棒性，还集成了 `Matplotlib` 绘图功能，能在运行结束后自动生成包含“原图、平滑图、锐化图”的**横向对比可视化报表**，并保存为高分辨率图片。

**运行说明** 项目包含两个独立的 Python 脚本。运行前请确保安装了 `numpy`, `opencv-python`, `matplotlib` 及 `pillow` 依赖库。代码默认读取项目目录下的测试图片（支持 BMP/JPG），处理结果将自动保存至指定输出文件夹。请确保文件路径配置正确即可直接运行。

更加详细的算法原理请参考代码注释，如有相关问题和漏洞恳请批评指正。

---

**Project Introduction** This is a Python-based demonstration project in the field of Computer Vision and Digital Image Processing, focusing on spatial domain **Smoothing (Noise Reduction)** and **Sharpening (Edge Enhancement)**. By contrasting two distinct implementation approaches—algorithms written from scratch versus utilizing established libraries—it highlights the distinction between underlying mathematical principles and practical engineering applications.

**Features Overview** In the **Low-Level Implementation** (`manual_filtering.py`), the code operates independently of core image processing libraries. It utilizes the `struct` module to manually parse the binary data of BMP file headers and implements **3x3 Mean Filtering convolution** and **Laplacian Sharpening operators** from scratch using nested loops, offering an intuitive demonstration of pixel-level mathematical operations.
In the **Engineering Application** (`opencv_filtering.py`), the project leverages the efficient algorithms of the OpenCV (`cv2`) library. It enhances robustness by supporting automatic detection and conversion of color/grayscale images and integrates `Matplotlib` to automatically generate a **side-by-side visual comparison report** featuring "Original, Smoothed, and Sharpened" images upon completion, saving the result as a high-resolution file.

**How to Run** The project consists of two independent Python scripts. Before running, please ensure that the `numpy`, `opencv-python`, `matplotlib`, and `pillow` dependencies are installed. The code defaults to reading test images (supporting BMP/JPG) from the project directory, and results are automatically saved to the specified output folder. Simply ensure the file paths are configured correctly to execute the scripts directly.

Please refer to the code comments for more detailed algorithmic principles. We welcome any feedback and corrections regarding potential issues or vulnerabilities.
