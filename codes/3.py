import os

# os.environ['KERAS_BACKEND'] = "plaidml.keras.backend"
# import keras



# from tensorflow import keras
import pandas as pd
import cv2 as cv
import numpy as np
import random
import matplotlib.pyplot as plt




def PSNR(I1, I2):
    s1 = cv.absdiff(I1, I2) #|I1 - I2|
    s1 = np.float32(s1)     # cannot make a square on 8 bits
    s1 = s1 * s1            # |I1 - I2|^2
    sse = s1.sum()          # sum elements per channel
    if sse <= 1e-10:        # sum channels
        return 0            # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

def MSSISM(i1, i2):
    C1 = 6.5025
    C2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv.mean(ssim_map)       # mssim = average of ssim map
    return mssim

def MSE(img1, img2):
    squared_diff = (img1 - img2) ** 2
    summed = np.sum(squared_diff)
    num_pix = img1.shape[0] * img1.shape[1]  # img1 and 2 should have same shape
    err = summed / num_pix
    return err


# 1.	Подготовить несколько тестов по 4 изображения, из которых второе два будут похожими, с точки зрения человека (например, фотография сделанная с немного другого ракурса),а третье будет значительно отличаться от первого (фотография другой сцены), четвертое изображение получить, применив какие либо шумы или размытие к первому. Подготовить не менее 2 тестов, отличающихся характером изображений (темные, светлые, большие, малые и т.п.)


img1 = cv.imread("cat1.jpg")
# img1 = cv.resize(img1, (960, 540), interpolation=cv.INTER_AREA)
img2 = cv.imread("cat2.jpg")
# img2 = cv.resize(img2, (960, 540), interpolation=cv.INTER_AREA)
img3 = cv.imread("untitled4.png")
# img3 = cv.resize(img3, (960, 540), interpolation=cv.INTER_AREA)
img4 = img1.copy()
img4 = cv.medianBlur(img4,21)

# cv.imwrite("D:/image-2.png", img4)

# Сравнить значения метрик MSE, PSNR и SSIM попарно вычисленных между первым изображением и остальными, в каждом тесте
#
# print("MSE img1, img1 :",MSE(img1, img1))
# print("MSE img1, img2 :",MSE(img1, img2))
# print("MSE img1, img3 :",MSE(img1, img3))
# print("MSE img1, img4 :",MSE(img1, img4))
#
# print("PSNR img1, img1 :",PSNR(img1, img1))
# print("PSNR img1, img2 :",PSNR(img1, img2))
# print("PSNR img1, img3 :",PSNR(img1, img3))
# print("PSNR img1, img4 :",PSNR(img1, img4))
#
# print("MSSISM img1, img1 :",MSSISM(img1, img1))
# print("MSSISM img1, img2 :",MSSISM(img1, img2))
# print("MSSISM img1, img3 :",MSSISM(img1, img3))
# print("MSSISM img1, img4 :",MSSISM(img1, img4))


# 2.	Оценить значения метрик MSE, PSNR и SSIM для искусственно зашумленного изображения. В качестве шума реализовать случайное заполнение заданного процента пикселей изображения случайными числами от 0 до 255 (для цветных изображений – в каждом канале). Исследовать влияние процента зашумленности изображения на значения метрик.

# def noiser(img, percent = 50):
#     noise  = img.copy()
#
#     for y in range(noise.shape[0]):
#         for x in range(noise.shape[1]):
#
#             p = random.randint(0, 100)
#             if p < percent:
#                 for c in range(noise.shape[2]):
#                         noise[y,x,c] = np.clip(noise[y,x,c], random.randint(1,255), random.randint(1,255))
#             else:
#                 pass
#     return noise
#
#
# noised1 = noiser(img1, 5)
# noised2 = noiser(img1, 30)
# noised3 = noiser(img1, 70)
#
# cv.imshow("noised 5% ", noised1)
# cv.imshow("noised 30% ", noised2)
# cv.imshow("noised 70% ", noised3)
#
# # cv.imwrite("D:/noised 5%.png", noised1)
# # cv.imwrite("D:/noised 30%.png", noised2)
# # cv.imwrite("D:/noised 70%.png", noised3)
#
# print("MSE img1, noised 5% :",MSE(img1, noised1))
# print("MSE img1, noised 30% :",MSE(img1, noised2))
# print("MSE img1, noised 70% :",MSE(img1, noised3))
#
# print("PSNR img1, noised 5% :",PSNR(img1, noised1))
# print("PSNR img1, noised 30% :",PSNR(img1, noised2))
# print("PSNR img1, noised 70% :",PSNR(img1, noised3))
#
# print("MSSISM img1, noised 5% :",MSSISM(img1, noised1))
# print("MSSISM img1, noised 30% :",MSSISM(img1, noised2))
# print("MSSISM img1, noised 70% :",MSSISM(img1, noised3))


# 3.	Оценить значения MSE, PSNR и SSIM при JPEG сжатии с различными параметрами качества (с шагом 5, от 0 до 100).

# from PIL import Image
# im1 = Image.open("cat1.jpg")
#
# IMAGE_5 = os.path.join('./cat5.jpg')
# im1.save(IMAGE_5,"JPEG", quality=5)
# IMAGE_10 = os.path.join('./cat10.jpg')
# im1.save(IMAGE_10,"JPEG", quality=10)
# IMAGE_20 = os.path.join('./cat20.jpg')
# im1.save(IMAGE_20,"JPEG", quality=20)
# IMAGE_40 = os.path.join('./cat40.jpg')
# im1.save(IMAGE_40,"JPEG", quality=40)
#
# cat5 = cv.imread("cat5.jpg")
# cat10 = cv.imread("cat10.jpg")
# cat20 = cv.imread("cat20.jpg")
# cat40 = cv.imread("cat40.jpg")
#
# print("MSE img1, JPEG compression, quality=5 :",MSE(img1, cat5))
# print("MSE img1, JPEG compression, quality=10 :",MSE(img1, cat10))
# print("MSE img1, JPEG compression, quality=20 :",MSE(img1, cat20))
# print("MSE img1, JPEG compression, quality=40 :",MSE(img1, cat40))
#
# print("PSNR img1, JPEG compression, quality=5 :",PSNR(img1, cat5))
# print("PSNR img1, JPEG compression, quality=10 :",PSNR(img1, cat10))
# print("PSNR img1, JPEG compression, quality=20 :",PSNR(img1, cat20))
# print("PSNR img1, JPEG compression, quality=40 :",PSNR(img1, cat40))
#
# print("MSSISM img1, JPEG compression, quality=5 :",MSSISM(img1, cat5))
# print("MSSISM img1, JPEG compression, quality=10 :",MSSISM(img1, cat10))
# print("MSSISM img1, JPEG compression, quality=20 :",MSSISM(img1, cat20))
# print("MSSISM img1, JPEG compression, quality=40 :",MSSISM(img1, cat40))



# 4.	Оценить деградацию изображения при его последовательном уменьшении и обратном увеличении, сравнить разные методы интерполяции при изменении размеров и разные коэффициенты масштабирования изображения.

# def resize (img, scale=0.3):
#     width = int(img.shape[1]*scale)
#     height = int(img.shape[0]*scale)
#     dimensions = (width, height)
#     return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
#
# img50 = resize(resize(img1,0.5),2.0)
# img25 = resize(resize(img1,0.25),4.0)
# img01 = resize(resize(img1,0.1),10.0)
# # cv.imshow("img01",img01)
# cv.imwrite("D:/img01INTER_AREA.png", img01)
#
# print("MSE img1, rescaling, quality=50% :",MSE(img1, img50))
# print("MSE img1, rescaling, quality=25% :",MSE(img1, img25))
# print("MSE img1, rescaling, quality=10% :",MSE(img1, img01))
#
# print("PSNR img1, rescaling, quality=50%  :",PSNR(img1, img50))
# print("PSNR img1, rescaling, quality=25%  :",PSNR(img1, img25))
# print("PSNR img1, rescaling, quality=10%  :",PSNR(img1, img01))
#
# print("MSSISM img1, rescaling, quality=50%  :",MSSISM(img1, img50))
# print("MSSISM img1, rescaling, quality=25%  :",MSSISM(img1, img25))
# print("MSSISM img1, rescaling, quality=10%  :",MSSISM(img1, img01))




#
# def resize (img, scale=0.3):
#     width = int(img.shape[1]*scale)
#     height = int(img.shape[0]*scale)
#     dimensions = (width, height)
#     return cv.resize(img, dimensions, interpolation=cv.INTER_NEAREST)
# img50 = resize(resize(img1,0.5),2.0)
# img25 = resize(resize(img1,0.25),4.0)
# img01 = resize(resize(img1,0.1),10.0)
# cv.imwrite("D:/img01INTER_NEAREST.png", img01)
#
# print("MSE img1, rescaling, quality=50% :",MSE(img1, img50))
# print("MSE img1, rescaling, quality=25% :",MSE(img1, img25))
# print("MSE img1, rescaling, quality=10% :",MSE(img1, img01))
#
# print("PSNR img1, rescaling, quality=50%  :",PSNR(img1, img50))
# print("PSNR img1, rescaling, quality=25%  :",PSNR(img1, img25))
# print("PSNR img1, rescaling, quality=10%  :",PSNR(img1, img01))
#
# print("MSSISM img1, rescaling, quality=50%  :",MSSISM(img1, img50))
# print("MSSISM img1, rescaling, quality=25%  :",MSSISM(img1, img25))
# print("MSSISM img1, rescaling, quality=10%  :",MSSISM(img1, img01))

# def resize (img, scale=0.3):
#     width = int(img.shape[1]*scale)
#     height = int(img.shape[0]*scale)
#     dimensions = (width, height)
#     return cv.resize(img, dimensions, interpolation=cv.INTER_CUBIC)
# img50 = resize(resize(img1,0.5),2.0)
# img25 = resize(resize(img1,0.25),4.0)
# img01 = resize(resize(img1,0.1),10.0)
# cv.imwrite("D:/img01INTER_CUBIC.png", img01)
#
# print("MSE img1, rescaling, quality=50% :",MSE(img1, img50))
# print("MSE img1, rescaling, quality=25% :",MSE(img1, img25))
# print("MSE img1, rescaling, quality=10% :",MSE(img1, img01))
#
# print("PSNR img1, rescaling, quality=50%  :",PSNR(img1, img50))
# print("PSNR img1, rescaling, quality=25%  :",PSNR(img1, img25))
# print("PSNR img1, rescaling, quality=10%  :",PSNR(img1, img01))
#
# print("MSSISM img1, rescaling, quality=50%  :",MSSISM(img1, img50))
# print("MSSISM img1, rescaling, quality=25%  :",MSSISM(img1, img25))
# print("MSSISM img1, rescaling, quality=10%  :",MSSISM(img1, img01))
# 5.	Оценить влияние размытия (по Гауссу, билатерального) на значение MSE, PSNR и SSIM метрик изображения. Для этого необходимо сравнить исходное изображение с размытым с различной степенью размытия.

gauss = cv.GaussianBlur(img1, (3,3), cv.BORDER_DEFAULT)
bilateral= cv.bilateralFilter(img1, 100 ,25,300)

print("MSE img1, GaussianBlur :",MSE(img1, gauss))
print("MSE img1, bilateralFilter :",MSE(img1, bilateral))

print("PSNR img1, GaussianBlur :",PSNR(img1, gauss))
print("PSNR img1, bilateralFilter :",PSNR(img1, bilateral))

print("MSSISM img1, GaussianBlur :",MSSISM(img1, gauss))
print("MSSISM img1, bilateralFilter :",MSSISM(img1, bilateral))


cv.waitKey(00)


