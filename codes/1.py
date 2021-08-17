import os

# os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
# import keras



# from tensorflow import keras
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
path = "1974-lamborghini-countach-classic-car-supercar-orange.jpg"
img = cv.imread(path)
# cv.imshow(path,img)

def resize (img, scale=0.3):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)

img = resize(img)



# 1.	Преобразовать изображение к монохромному виду (самостоятельно найти формулу вычисления яркостной составляющей по цветовым компонентам);
#
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("image", img)
# cv.imshow("gray", gray)

# 2.	Выполнить цветокоррекцию изображения (корректировку уровней красного, синего и зеленого компонентов цвета);

#
#
#
# zeros = np.zeros(img.shape[:2], dtype='uint8')
# blu = cv.applyColorMap(img, cv.COLORMAP_WINTER,zeros )
# cv.imshow("blue", blu)
#
# zeros = np.zeros(img.shape[:2], dtype='uint8')
# green = cv.applyColorMap(img, cv.COLORMAP_SUMMER,zeros )
# cv.imshow("green", green)
#
# zeros = np.zeros(img.shape[:2], dtype='uint8')
# red = cv.applyColorMap(img, cv.COLORMAP_HOT,zeros )
# cv.imshow("red", red)

# 3.	Выполнить корректировку яркости и контрастности изображения;


# zeros = np.zeros(img.shape, dtype='uint8')
#
# contrast = 3.0
# brightness = 60
#
# contrased = cv.convertScaleAbs(img, alpha=contrast)
# brightend = cv.convertScaleAbs(img, beta=brightness)
#
# cv.imshow('New contrasted', contrased)
# cv.imshow('New brightend', brightend)



# вариант с циклами и указанием отдельных пискелей
# for y in range(img.shape[0]):
#     for x in range(img.shape[1]):
#         for c in range(img.shape[2]):
#             zeros[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
# cv.imshow('New Image', zeros)

# 4.	Открыть группу изображений (прочитать список из файла) и составить из них атлас (используя регионы интереса);

#
#
# folder1 = os.path.abspath("atlas pics")#имя папки с картинками
# project_folder = os.path.split(folder1)[0]
# listdir = os.listdir(folder1) # список с файлами
#
# img1 = cv.imread(folder1 + "/" + listdir[0])
#
# y = img1.shape[0]
# x = img1.shape[1]
#
# zeros = np.zeros((int(y/2),int(x/2),3), dtype='uint8')
#
# def resize (img, scale=0.25):
#     width = int(img.shape[1]*scale)
#     height = int(img.shape[0]*scale)
#     dimensions = (width, height)
#     return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
# #
# counter = 0
# x_now = 0
# y_now = 0
# print(listdir)
# for each in listdir:
#     pik = cv.imread(folder1 + "/" + each)
#     pik = resize(pik)
#     for y in range(pik.shape[0]):
#         y_now +=1
#         for x in range(pik.shape[1]):
#             for c in range(pik.shape[2]):
#                 try:
#                     zeros[y_now,x-x_now,c] = np.clip(pik[y,x,c] , 0, 255)
#                 except:
#                     continue
#     if y_now==pik.shape[0]*2:
#         x_now = pik.shape[1]
#         y_now = 0
# cv.imshow('New Image', zeros)


# 5.	Открыть изображение и перемешать полосы (строки или столбцы) пикселей (ширина полосы должна задаваться).

# import random
# wide = 40
# a = 4#random.randint(2,8)
#
#
# new_img = cv.convertScaleAbs(img)
# y_len = new_img.shape[0]
#
# y_target = new_img.shape[0]/100*a*10
#
# for y in range(img.shape[0]):
#     if y<y_target or y>y_target+wide:
#         pass
#     else:
#         # print("first", y)
#         for x in range(img.shape[1]):
#             for c in range(img.shape[2]):
#                 new_img[y,x,c] = np.clip(img[y+100,x,c], 0, 255)
#
# for y in range(img.shape[0]):
#     if y<y_target+100 or y>y_target+wide+100:
#         pass
#     else:
#         # print("second", y)
#         for x in range(img.shape[1]):
#             for c in range(img.shape[2]):
#                 new_img[y,x,c] = np.clip(img[y-100,x,c], 0, 255)
#
# cv.imshow('New Image', new_img)



# 6.	Написать алгоритм рисования прямой линии (окружности, эллипса, график функции).
# def resize (img, scale=0.25):
#     width = int(img.shape[1]*scale)
#     height = int(img.shape[0]*scale)
#     dimensions = (width, height)
#     return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
#
# plt.show()
#
# from matplotlib import image
# from matplotlib import pyplot as plt
#
# data = image.imread('sobaka.jpg')
# data = resize(data,0.2)
# y = data.shape[0]
# x = data.shape[1]
#
#
#
#
# plt.plot(x, y)
# plt.xlabel('Bins')
# plt.ylabel('# of pixels')
# colors = ("b","g","r")
#
# for each,col in enumerate(colors):
#     hist =  cv.calcHist([data],[each],None,[256],[0,256])#при желании вместо нан можно вставить маску
#     plt.plot(hist,color = col)
#     plt.xlim([0,256])
#
# plt.imshow(data)
# plt.show()
#
# x2 = [15, 240]#тут слева одна точка - справа вторая
# y2 = [50, 150]
# plt.plot(x2, y2, color="violet", linewidth=5)
# plt.axis('off')
# plt.imshow(data)
# plt.show()



cv.waitKey(00)