#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt;
import scipy.io
from pyDOE import lhs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from numpy import genfromtxt
import matplotlib.gridspec as gridspec
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# filepath_to_save_mode = '1D.pt'
# data = genfromtxt("data.csv", delimiter=',')#运行哪个解放哪个解数据
import pandas as pd

from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
from skimage import io, filters, measure, morphology
import matplotlib.pyplot as plt


# In[23]:


# 读取第一张图像
#image_path1 = "C:/Users/0808/Desktop/learning/毕业successful/two time/simple model/useful/useful_p3.5a.png"
image_path1 = ".../case1_2_p0a.png"
image1 = io.imread(image_path1, as_gray=True)  # 转换为灰度图像

# 读取第二张图像
#image_path2 = "C:/Users/0808/Desktop/learning/毕业successful/two time/simple model/useful/useful_r3.5a.png"
image_path2 = ".../case1_2_r0a.png"
image2 = io.imread(image_path2, as_gray=True)  # 转换为灰度图像

# 定义一个函数来处理图像并查找轮廓
def process_image(image):
    # 使用高斯模糊减少噪声
    blurred = filters.gaussian(image, sigma=1)
    # 二值化处理
    threshold = filters.threshold_otsu(blurred)
    binary = blurred > threshold
    # 去除小的噪声点
    binary = morphology.remove_small_objects(binary, min_size=100)
    # 查找轮廓
    contours = measure.find_contours(binary, 0.5)
    return contours

# 处理第一张图像
contours1 = process_image(image1)

# 处理第二张图像
contours2 = process_image(image2)

# 定义一个函数来检查轮廓是否靠近图像边框
def is_near_border(contour, image_shape, border_width=10):
    min_row, min_col = np.min(contour, axis=0)
    max_row, max_col = np.max(contour, axis=0)
    return (min_row < border_width or max_row >= image_shape[0] - border_width or
            min_col < border_width or max_col >= image_shape[1] - border_width)

# 筛选轮廓（假设气泡的轮廓面积较大且不靠近边框）
def filter_contours(contours, image, min_area=100):
    filtered_contours = []
    for cnt in contours:
        # 计算轮廓的面积
        area = measure.moments(cnt).sum()
        # 排除靠近边框的轮廓和小面积的轮廓
        if area > min_area and not is_near_border(cnt, image.shape):
            # 排除上方线条的轮廓
            if np.min(cnt[:, 0]) > image.shape[0] * 0.3:  # 根据实际图像调整0.3的值
                filtered_contours.append(cnt)
    return filtered_contours

filtered_contours1 = filter_contours(contours1, image1)
filtered_contours2 = filter_contours(contours2, image2)

# 创建一个指定尺寸的 figure
# ...（前面的图像处理代码保持不变）


# 创建figure并设置轴和样式
fig_1 = plt.figure(figsize=(1.8, 2.5))
ax = fig_1.add_subplot(111)
ax.set_facecolor('white')

# 绘制黑色边框
#ax.plot([0, 0.06, 0.06, 0, 0], [0, 0, 0.08, 0.08, 0], 'k', linewidth=1)

# ---- 设置坐标轴 ----
# 设置y轴刻度（只显示0, 0.04, 0.08）
#ax.set_yticks([0,0.04, 0.08])
#ax.tick_params(axis='both', labelsize=15)
#ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))  # 保留两位小数

# 设置x轴刻度（保持自动）
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# ---- 设置标签和标题 ----
#ax.set_title('t=0.5s', fontsize=15, pad=6)
#ax.set_xlabel('x(m)', fontsize=15, labelpad=2)
#ax.set_ylabel('y(m)', fontsize=15, labelpad=2)

# ---- 绘制轮廓时控制图例 ----
# 添加标签标记（只在第一次绘制时添加label）

for i, contour in enumerate(filtered_contours2):
    x_norm = contour[:, 1] / image2.shape[1] * 0.06
    y_norm = (1 - contour[:, 0] / image2.shape[0]) * 0.08
    label = 'CFD' if i == 0 else None  # 只在第一个轮廓添加标签
    ax.plot(x_norm, y_norm, 'r', linewidth=1, label=label)
    
for i, contour in enumerate(filtered_contours1):
    x_norm = contour[:, 1] / image1.shape[1] * 0.06
    y_norm = (1 - contour[:, 0] / image1.shape[0]) * 0.08
    label = 'MS-PINN' if i == 0 else None  # 只在第一个轮廓添加标签
    ax.plot(x_norm, y_norm, 'b--', linewidth=1, dashes=(3,5),label=label)


# 添加图例（调整位置和样式）
#ax.legend(loc='upper left', fontsize=9, frameon=True,
         #handlelength=2, borderaxespad=0.4)
ax.axis('off')
# ---- 保持范围设置 ----
ax.set_xlim(0, 0.06)
ax.set_ylim(0, 0.08)

# 在文件头添加格式控制
from matplotlib.ticker import FormatStrFormatter

plt.savefig('0lunkuo_without.png', dpi=600, bbox_inches='tight')
plt.show()


# In[ ]:





# In[ ]:





# In[6]:


import matplotlib.pyplot as plt
from PIL import Image
import os

# 设置图片文件夹路径和图片文件名
image_folder = ".../case1双气泡融合"  # 文件夹路径
image_files = ["0.5lunkuo.png", "1lunkuo.png", "1.75lunkuo.png", "2lunkuo.png", "2.75lunkuo.png", "3lunkuo.png"]  # 图片文件名
# 创建一个 1 行 5 列的 figure
fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(9.5, 2))  # 15 是总宽度，3 是高度

# 遍历每个子图和对应的图片文件
for i, ax in enumerate(axes):
    # 读取图片
    image_path = os.path.join(image_folder, image_files[i])
    img = Image.open(image_path)
    
    # 显示图片
    ax.imshow(img)
    # 隐藏坐标轴
    ax.axis('off')

# 调整子图之间的间距
plt.subplots_adjust(wspace=0.00001)  # 设置子图之间的水平间距
# 显示图像
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




