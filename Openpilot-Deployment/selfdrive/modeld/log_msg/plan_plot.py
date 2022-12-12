#!/usr/bin/env python3
import os
from time import sleep
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np

# file = open("./extra_plan_log.txt")

# # data = file.readline()
# x = []
# y = []

# for num in file:
#     num = num.strip('\n')
#     num = num.split(',')
#     y.append(float(num[1]))
#     x.append(float(num[2]))
# file.close
   
# for index in len(x)/33:
    
#     for num in range(0,33):
#         tmp_x = x.append(x[index * 33 + num])
#         tmp_y = y.append(y[index * 33 + num])
#         plt.figure()
#         plt.title('best_plan')
#         plt.plot(tmp_x,tmp_y)
#         plt.xlim(0,100)
#         plt.ylim(-50,50)
#         x_ticks = np.arange(0,100,1)
#         y_ticks = np.arange(-50,50,1)
#         plt.xticks(x_ticks)
#         plt.yticks(y_ticks)
#         plt.show()
#     sleep(1)
#创建数据
x = np.linspace(-5, 5, 100)
y1 = np.sin(x)
y2 = np.cos(x)

#创建figure窗口，figsize设置窗口的大小
plt.figure(num=3, figsize=(8, 5))
#画曲线1
plt.plot(x, y1)
#画曲线2
plt.plot(x, y2, color='blue', linewidth=5.0, linestyle='--')
#设置坐标轴范围
plt.xlim((-5, 5))
plt.ylim((-2, 2))
#设置坐标轴名称
plt.xlabel('xxxxxxxxxxx')
plt.ylabel('yyyyyyyyyyy')
#设置坐标轴刻度
my_x_ticks = np.arange(-5, 5, 0.5)
#对比范围和名称的区别
#my_x_ticks = np.arange(-5, 2, 0.5)
my_y_ticks = np.arange(-2, 2, 0.3)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)

#显示出所有设置
plt.show()







