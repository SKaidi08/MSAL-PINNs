#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
traditional_df = pd.read_excel('case1_bad_stage3.xlsx')
improved_df = pd.read_excel('case1_stage3.xlsx')

# 提取损失数据和epoch列
traditional_loss = traditional_df['mse'].values
improved_loss = improved_df['mse'].values
all_epochs = traditional_df['epoch'].values  # 获取获取Excel中的epoch列
iterations = len(traditional_loss)

# 核心：确保X轴最后一个刻度为3000
target_end = 30000  # 目标结尾值
sample_step = 200  # 间隔步长

# 1. 找到小于等于3000的最大epoch值的索引
# （防止3000超出实际数据范围）
valid_epochs = all_epochs[all_epochs <= target_end]
if len(valid_epochs) == 0:
    # 极端情况：所有epoch都大于3000，取最小的那个
    last_idx = np.argmin(all_epochs)
else:
    last_epoch = valid_epochs[-1]  # 最接近3000的有效值
    last_idx = np.where(all_epochs == last_epoch)[0][0]  # 找到其索引

# 2. 生成从0到last_idx的采样点，步长200
sample_indices = np.arange(0, last_idx + 1, sample_step)

# 3. 确保最后一个点是3000（或最接近的值）
if sample_indices[-1] != last_idx:
    sample_indices = np.append(sample_indices, last_idx)

# 4. 获取对应的刻度位置和标签
ticks_pos = sample_indices
ticks_label = all_epochs[sample_indices]

# 绘制图像
plt.figure(figsize=(4.5, 3))
plt.plot(traditional_loss, label='Single-stage-stage PINN Loss', color='red')
plt.plot(improved_loss, label='MSAL-PINN Loss', color='blue')

# 应用刻度
plt.xticks(ticks_pos, ticks_label)

# 其他设置
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim(-0.05, 2.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
    


# In[52]:


# 读取数据
traditional_df = pd.read_excel('case1_bad_stage2.xlsx')
improved_df = pd.read_excel('case1_stage2.xlsx')

# 提取损失数据和epoch列
traditional_loss = traditional_df['mse'].values
improved_loss = improved_df['mse'].values
all_epochs = traditional_df['epoch'].values  # 获取Excel中的epoch列
iterations = len(traditional_loss)

# 基础设置：每隔200采样
sample_step = 200
base_indices = np.arange(0, iterations, sample_step)

# 核心：确保3000的位置被包含
target = 20000
# 找到3000在epoch列中的索引（如果存在）
if target in all_epochs:
    target_idx = np.where(all_epochs == target)[0][0]
    # 如果3000不在基础采样中，则添加进去
    if target_idx not in base_indices:
        sample_indices = np.append(base_indices, target_idx)
        # 保持排序
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = base_indices
else:
    # 如果数据中没有3000，可选择添加最接近3000的值
    closest_idx = np.argmin(np.abs(all_epochs - target))
    sample_indices = np.append(base_indices, closest_idx)
    sample_indices = np.sort(sample_indices)

# 获取对应的刻度位置和标签
ticks_pos = sample_indices
ticks_label = all_epochs[sample_indices]

# 绘制图像
plt.figure(figsize=(4.5, 3))
# 获取当前坐标轴对象，仅设置框内背景为淡蓝色
ax = plt.gca()
#ax.set_facecolor('#E6F7FF')  

plt.plot(traditional_loss, label='Single-stage PINN Loss', color='red')
plt.plot(improved_loss, label='MSAL-PINN Loss', color='blue')

# 应用刻度
plt.xticks(ticks_pos, ticks_label)

# 其他设置
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim(-0.05, 2.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
    


# In[51]:


# 读取数据
traditional_df = pd.read_excel('case1_bad0_stage1.xlsx')
improved_df = pd.read_excel('case1_stage1.xlsx')

# 提取损失数据和epoch列
traditional_loss = traditional_df['mse'].values
improved_loss = improved_df['mse'].values
all_epochs = traditional_df['epoch'].values  # 获取Excel中的epoch列
iterations = len(traditional_loss)

# 基础设置：每隔200采样
sample_step = 200
base_indices = np.arange(0, iterations, sample_step)

# 核心：确保3000的位置被包含
target = 10000
# 找到3000在epoch列中的索引（如果存在）
if target in all_epochs:
    target_idx = np.where(all_epochs == target)[0][0]
    # 如果3000不在基础采样中，则添加进去
    if target_idx not in base_indices:
        sample_indices = np.append(base_indices, target_idx)
        # 保持排序
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = base_indices
else:
    # 如果数据中没有3000，可选择添加最接近3000的值
    closest_idx = np.argmin(np.abs(all_epochs - target))
    sample_indices = np.append(base_indices, closest_idx)
    sample_indices = np.sort(sample_indices)

# 获取对应的刻度位置和标签
ticks_pos = sample_indices
ticks_label = all_epochs[sample_indices]

# 绘制图像
plt.figure(figsize=(4.5, 3))
# 获取当前坐标轴对象，仅设置框内背景为淡蓝色
ax = plt.gca()
#ax.set_facecolor('#FFFFCC')  

plt.plot(traditional_loss, label='Single-stage PINN Loss', color='red')
plt.plot(improved_loss, label='MSAL-PINN Loss', color='blue')

# 应用刻度
plt.xticks(ticks_pos, ticks_label)

# 其他设置
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim(-0.05, 2.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
    


# In[42]:


# 读取数据
traditional_df = pd.read_excel('case1_bad.xlsx')[::13]
improved_df = pd.read_excel('case1.xlsx')[::13]

# 提取损失数据和epoch列
traditional_loss = traditional_df['mse'].values
improved_loss = improved_df['mse'].values
all_epochs = traditional_df['epoch'].values  # 获取Excel中的epoch列
iterations = len(traditional_loss)

# 基础设置：每隔200采样
#sample_step = 2000
#base_indices = np.arange(0, iterations, sample_step)

# 核心：确保3000的位置被包含
target = 30000
# 找到3000在epoch列中的索引（如果存在）
if target in all_epochs:
    target_idx = np.where(all_epochs == target)[0][0]
    # 如果3000不在基础采样中，则添加进去
    if target_idx not in base_indices:
        sample_indices = np.append(base_indices, target_idx)
        # 保持排序
        sample_indices = np.sort(sample_indices)
    else:
        sample_indices = base_indices
else:
    # 如果数据中没有3000，可选择添加最接近3000的值
    closest_idx = np.argmin(np.abs(all_epochs - target))
    sample_indices = np.append(base_indices, closest_idx)
    sample_indices = np.sort(sample_indices)

# 获取对应的刻度位置和标签
ticks_pos = sample_indices
ticks_label = all_epochs[sample_indices]

# 绘制图像
plt.figure(figsize=(4.5, 3))
plt.plot(traditional_loss, label='Single-stage PINN Loss', color='red')
plt.plot(improved_loss, label='MSAL-PINN Loss', color='blue')

# 应用刻度
plt.xticks(ticks_pos, ticks_label)

# 其他设置
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim(-0.05, 2.5)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
    


# In[ ]:




