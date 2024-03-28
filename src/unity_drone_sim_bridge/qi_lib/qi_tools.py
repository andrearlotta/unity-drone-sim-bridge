import cv2
import numpy as np

def saturation(I):
    # Saturation is computed as the standard deviation of the color channels
    R = I[:, :, 0]
    G = I[:, :, 1]
    B = I[:, :, 2]

    mu = (R + G + B) / 3

    return np.sqrt(((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)

def relativeLuminance(I):
    return 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]

def imhist(img):
    hist = cv2.calcHist([(img*255.0).astype(np.uint8)], [0], None, [256], [0, 256])
    return  hist / hist.sum()

def img_2_qi(img):
    img = img.astype(float) /255.0
    s_hist = np.sum(np.array(imhist(saturation(img.astype(float))))[:50, 0])
    l_hist  = np.sum(np.array(imhist(relativeLuminance(img.astype(float))))[50:200, 0])
    return (s_hist + l_hist)
