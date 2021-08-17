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




def resize (img, scale=0.3):
    width = int(img.shape[1]*scale)
    height = int(img.shape[0]*scale)
    dimensions = (width, height)
    return cv.resize(img, dimensions, interpolation=cv.INTER_AREA)
img = resize(img)

# # cv.imshow(path,img)
# # 1.	Сделать изображение размытым (или/и монохромным), кроме области внутри круга заданного радиуса, граница круга должна быть плавной.
#
# radius = 150
#
# bilateral= cv.bilateralFilter(img, 50 ,800,300)
# # cv.imshow("bilateral",bilateral)
#
# blank1 = np.zeros(img.shape[:2],dtype = 'uint8')
#
# mask = cv.circle(blank1, (200,200),radius,255,-1 )
# masked = cv.bitwise_not(img,img,mask=mask)
# nottttt = cv.bitwise_not(masked,bilateral,mask=mask)
#
# cv.imshow("Masked", nottttt)
#
#
#
#
# # 2.	Выделить границы на изображении (canny) и применить размытие по гауссу ко всему изображению, кроме границ.
#
#
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#
# canny = cv.Canny(gray, 125,100)
#
# gauss = cv.GaussianBlur(gray, (1111,1111), cv.BORDER_DEFAULT)
#
# mask = canny
# masked = cv.bitwise_not(gray,gray,mask=mask)
#
# nottttt = cv.bitwise_not(masked,gauss,mask=mask)
# cv.imshow("final", nottttt)
#
#
# # 3.	Разработать алгоритм, позволяющий выделять границы черным цветом, толщину линий и параметры их выделения (параметры для фильтра Кенни)
#
# canny = cv.Canny(img,125,100)
# new  = img.copy()
# for y in range(new.shape[0]):
#     for x in range(new.shape[1]):
#         for c in range(new.shape[2]):
#             new[y,x,c] = np.clip(new[y,x,c], 0, 0)
#
# masked = cv.bitwise_not(new,new,mask=canny)
#
# nottttt = cv.bitwise_not(masked,img,mask=canny)
#
# cv.imshow("masked", masked)
# cv.imshow("ready", nottttt)
#
#
# # 4.	Используя filter2D реализовать размытие с ядром в виде равномерно заполненного круга настраиваемого радиуса и кольца заданного радиуса и толщины, сравнить полученные изображения.
# radius = 70
# width = 20
#
#
# img_src = img#= cv.imread('sample.jpg')
#
#
#
# kernel = np.array([[1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1],
#                    [1,1,1,1,1,1,1,1,1]])
#
# kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
#
# img_rst = cv.filter2D(img_src,-1,kernel)
#
# blank1 = np.zeros(img.shape[:2],dtype = 'uint8')
#
# mask = cv.circle(blank1, (1100,750),radius,255,-1 )
# masked = cv.bitwise_not(img,img,mask=mask)
# nottttt = cv.bitwise_not(masked,img_rst,mask=mask)
# cv.imshow('result.jpg',masked)
#
# mask1 = cv.circle(blank1.copy(), (650,900),radius,255,-1 )
# mask2 = cv.circle(blank1.copy(), (650,900),int(radius-width),255,-1 )
# masked2 = cv.bitwise_xor(mask1,mask2)
#
# masked3 = cv.bitwise_not(img,img,mask=masked2)
# nottttt3 = cv.bitwise_not(masked3,img_rst,mask=masked2)
#
#
# cv.imshow('round.jpg',nottttt3)

#
# # 5.	Реализовать эффект смазывания от движения (motion blur) с настраиваемым направлением движения и степенью смазывания
#
#
# try:
#     vector = int(input("введите направление смазывания - вертикальное:0, горизонтальное:1 "))
#     if vector!= 1 or vector!=0:
#         vector = 1
#     force = int(input("введите силу смазывания"))
#     if force%2==0:
#         force=force+3
# except:
#     vector = 0
#     force = 4003
#
#
# if vector ==0:
#     kortej = (1,force)
# else:
#     kortej = (force, 1)
#
# gauss = cv.GaussianBlur(img, kortej, cv.BORDER_DEFAULT)
# cv.imshow("gauss",gauss)

#
# # 6.	Сравнить фильтрацию по Гауссу, билатеральную и медианную фильтрацию (реализовать программу, выводящую результат применения разных фильтров одновременно, сравнить их для разных значений параметров)
#
#
# gauss = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)
# cv.imshow("gauss",gauss)
#
# median = cv.medianBlur(img,7)
# cv.imshow("median",median)
#
#
# bilateral= cv.bilateralFilter(img, 50 ,140,150)
# # второй - диаметор . 3 - цвет сигмы- больше значение - больше разных цветов будет влиять на пиксель. сигма спейс- расстояние влияния
# cv.imshow("bilateral",bilateral)
#
#
#
#
#
#
#
#
# # 7.	Сравнить эффект от применения фильтрации по Гауссу и билатеральной фильтрации (продемонстрировать поведение фильтров на границах для различных значений параметров фильтра)
#
# blank1 = np.zeros(img.shape[:2],dtype = 'uint8')
# gauss = cv.GaussianBlur(img.copy(), (5,5), cv.BORDER_DEFAULT)
#
# mask = cv.rectangle(blank1.copy(),(img.shape[1]//2,0),(img.shape[1] ,img.shape[0]),254,-1)
# masked = cv.bitwise_not(img.copy(),img.copy(),mask=mask)
# result = cv.bitwise_not(masked,gauss,mask=mask)
# cv.imshow('gauss.jpg',result)
#
#
# bilateral= cv.bilateralFilter(img.copy(), 150 ,150,150)
# mask2 = cv.rectangle(blank1.copy(),(img.shape[1]//2,0),(img.shape[1] ,img.shape[0]),254,-1)
# masked2 = cv.bitwise_not(img.copy(),img.copy(),mask=mask2)
# result2 = cv.bitwise_not(masked2,bilateral,mask=mask2)
# cv.imshow('bilateral.jpg',result2)
#
# # 8.	Используя фильтрацию по гауссу (либо билатеральный фильтр) выполнить многоуровневую фильтрацию (усилить детали изображения с размерами порядка 5-10 пикселей, оставив неизменными более мелкие детали)
#
#
#
# blank1 = np.zeros(img.shape[:2],dtype = 'uint8')
#
# bilateral= cv.bilateralFilter(img, 200 ,150,100)
# bilateral2= cv.bilateralFilter(bilateral, 200 ,150,100)
# bilateral3= cv.bilateralFilter(bilateral2, 200 ,150,100)
#
# mask = cv.rectangle(blank1.copy(),(img.shape[1]//2,0),(img.shape[1] ,img.shape[0]),254,-1)
# masked = cv.bitwise_not(img.copy(),img.copy(),mask=mask)
# result = cv.bitwise_not(masked,bilateral3,mask=mask)
# cv.imshow('gauss.jpg',result)
#
#
#
#
#
# # 9.	Реализовать алгоритм локального повышения контрастности retinex
#
# #
# # from scipy.spatial import distance
# # from scipy.ndimage.filters import convolve
# # from scipy.sparse import diags, csr_matrix
# # from scipy.sparse.linalg import spsolve
# # # from utils import get_sparse_neighbor
# #
# #
# #
# #
# #
# # def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15):
# #     """Create a kernel (`size` * `size` matrix) that will be used to compute the he spatial affinity based Gaussian weights.
# #     Arguments:
# #         spatial_sigma {float} -- Spatial standard deviation.
# #     Keyword Arguments:
# #         size {int} -- size of the kernel. (default: {15})
# #     Returns:
# #         np.ndarray - `size` * `size` kernel
# #     """
# #     kernel = np.zeros((size, size))
# #     for i in range(size):
# #         for j in range(size):
# #             kernel[i, j] = np.exp(
# #                 -0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))
# #
# #     return kernel
# #
# # def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
# #     """Compute the smoothness weights used in refining the illumination map optimization problem.
# #     Arguments:
# #         L {np.ndarray} -- the initial illumination map to be refined.
# #         x {int} -- the direction of the weights. Can either be x=1 for horizontal or x=0 for vertical.
# #         kernel {np.ndarray} -- spatial affinity matrix
# #     Keyword Arguments:
# #         eps {float} -- small constant to avoid computation instability. (default: {1e-3})
# #     Returns:
# #         np.ndarray - smoothness weights according to direction x. same dimension as `L`.
# #     """
# #     Lp = cv.Sobel(L, cv.CV_64F, int(x == 1), int(x == 0), ksize=1)
# #     T = convolve(np.ones_like(L), kernel, mode='constant')
# #     T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
# #     return T / (np.abs(Lp) + eps)
# #
# # def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
# #                                bc: float = 1, bs: float = 1, be: float = 1):
# #     """perform the exposure fusion method used in the DUAL paper.
# #     Arguments:
# #         im {np.ndarray} -- input image to be enhanced.
# #         under_ex {np.ndarray} -- under-exposure corrected image. same dimension as `im`.
# #         over_ex {np.ndarray} -- over-exposure corrected image. same dimension as `im`.
# #     Keyword Arguments:
# #         bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
# #         bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
# #         be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})
# #     Returns:
# #         np.ndarray -- the fused image. same dimension as `im`.
# #     """
# #     merge_mertens = cv.createMergeMertens(bc, bs, be)
# #     images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
# #     fused_images = merge_mertens.process(images)
# #     return fused_images
# #
# # def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray,
# #                                    eps: float = 1e-3):
# #     """Refine the illumination map based on the optimization problem described in the two papers.
# #        This function use the sped-up solver presented in the LIME paper.
# #     Arguments:
# #         L {np.ndarray} -- the illumination map to be refined.
# #         gamma {float} -- gamma correction factor.
# #         lambda_ {float} -- coefficient to balance the terms in the optimization problem.
# #         kernel {np.ndarray} -- spatial affinity matrix.
# #     Keyword Arguments:
# #         eps {float} -- small constant to avoid computation instability (default: {1e-3}).
# #     Returns:
# #         np.ndarray -- refined illumination map. same shape as `L`.
# #     """
# #     # compute smoothness weights
# #     wx = compute_smoothness_weights(L, x=1, kernel=kernel, eps=eps)
# #     wy = compute_smoothness_weights(L, x=0, kernel=kernel, eps=eps)
# #
# #     n, m = L.shape
# #     L_1d = L.copy().flatten()
# #
# #     # compute the five-point spatially inhomogeneous Laplacian matrix
# #     row, column, data = [], [], []
# #     for p in range(n * m):
# #         diag = 0
# #         for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
# #             weight = wx[k, l] if x else wy[k, l]
# #             row.append(p)
# #             column.append(q)
# #             data.append(-weight)
# #             diag += weight
# #         row.append(p)
# #         column.append(p)
# #         data.append(diag)
# #     F = csr_matrix((data, (row, column)), shape=(n * m, n * m))
# #
# #     # solve the linear system
# #     Id = diags([np.ones(n * m)], [0])
# #     A = Id + lambda_ * F
# #     L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))
# #
# #     # gamma correction
# #     L_refined = np.clip(L_refined, eps, 1) ** gamma
# #
# #     return L_refined
# #
# # def correct_underexposure(im: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
# #     """correct underexposudness using the retinex based algorithm presented in DUAL and LIME paper.
# #     Arguments:
# #         im {np.ndarray} -- input image to be corrected.
# #         gamma {float} -- gamma correction factor.
# #         lambda_ {float} -- coefficient to balance the terms in the optimization problem.
# #         kernel {np.ndarray} -- spatial affinity matrix.
# #     Keyword Arguments:
# #         eps {float} -- small constant to avoid computation instability (default: {1e-3})
# #     Returns:
# #         np.ndarray -- image underexposudness corrected. same shape as `im`.
# #     """
# #
# #     # first estimation of the illumination map
# #     L = np.max(im, axis=-1)
# #     # illumination refinement
# #     L_refined = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)
# #
# #     # correct image underexposure
# #     L_refined_3d = np.repeat(L_refined[..., None], 3, axis=-1)
# #     im_corrected = im / L_refined_3d
# #     return im_corrected
# #
# # # TODO: resize image if too large, optimization take too much time
# #
# # def enhance_image_exposure(im: np.ndarray, gamma: float, lambda_: float, dual: bool = True, sigma: int = 3,
# #                            bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3):
# #     """Enhance input image, using either DUAL method, or LIME method. For more info, please see original papers.
# #     Arguments:
# #         im {np.ndarray} -- input image to be corrected.
# #         gamma {float} -- gamma correction factor.
# #         lambda_ {float} -- coefficient to balance the terms in the optimization problem (in DUAL and LIME).
# #     Keyword Arguments:
# #         dual {bool} -- boolean variable to indicate enhancement method to be used (either DUAL or LIME) (default: {True})
# #         sigma {int} -- Spatial standard deviation for spatial affinity based Gaussian weights. (default: {3})
# #         bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
# #         bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
# #         be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})
# #         eps {float} -- small constant to avoid computation instability (default: {1e-3})
# #     Returns:
# #         np.ndarray -- image exposure enhanced. same shape as `im`.
# #     """
# #     # create spacial affinity kernel
# #     kernel = create_spacial_affinity_kernel(sigma)
# #
# #     # correct underexposudness
# #     im_normalized = im.astype(float) / 255.
# #     under_corrected = correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)
# #
# #     if dual:
# #         # correct overexposure and merge if DUAL method is selected
# #         inv_im_normalized = 1 - im_normalized
# #         over_corrected = 1 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)
# #         # fuse images
# #         im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
# #     else:
# #         im_corrected = under_corrected
# #
# #     # convert to 8 bits and returns
# #     return np.clip(im_corrected * 255, 0, 255).astype("uint8")
# #
# #
# #
# #
# # enhanced_image = enhance_image_exposure(img, 0.6, 0.15, True, sigma=3, bc=1, bs=1, be=1, eps=1e-3)
# #
# #
# #
# #
# # cv.imshow("enhanced_image",enhanced_image)
#
cv.waitKey(00)
