
#Implementation of gaussian filter algorithm

from itertools import product
import cv2
import numpy as np



def gen_gaussian_kernel(n_size, sigma):
    cent = n_size // 2
    a, b = np.mgrid[0 - cent : n_size - cent, 0 - cent : n_size - cent]
    gauss = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(a) + np.square(b)) / (2 * np.square(sigma)))
    return gauss


def gaussian_filter(img, n_size, sigma):
    h, w = img.shape[0], img.shape[1]
    dst_h = h - n_size + 1
    dst_w = w - n_size + 1

    img_arr = np.zeros((dst_h * dst_w, n_size * n_size))
    att = 0
    for i, j in product(range(dst_h), range(dst_w)):
        view = np.ravel(img[i : i + n_size, j : j + n_size])
        img_arr[att, :] = view
        att += 1

    gaussian_kernel = gen_gaussian_kernel(n_size, sigma)
    filter_array = np.ravel(gaussian_kernel)

    dst = np.dot(img_arr, filter_array).reshape(dst_h, dst_w).astype(np.uint8)

    return dst


if __name__ == "__main__":

    img = cv2.imread(r"cat.1860.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian4x4 = gaussian_filter(gray, 4, sigma=1)
    gaussian6x6 = gaussian_filter(gray, 6, sigma=0.8)

    #show
    cv2.imshow("gaussian filter with 4x4 mask", gaussian4x4)
    cv2.imshow("gaussian filter with 6x6 mask", gaussian6x6)
    cv2.waitKey()
