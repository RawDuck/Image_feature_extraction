import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.color import rgb2gray, rgba2rgb
from skimage import io
import numpy as np
import matplotlib.image as im
from scipy.interpolate import UnivariateSpline
from collections import defaultdict
import math


def grayscale_fourier_desc(img, size):
    #cartesian coordinates of image to polar coordinates
    value = np.sqrt(((img.shape[0] / 2.0) ** 2.0) + ((img.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    polar_image = np.rot90(polar_image)

    # fourier
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))
    # plt.imshow(spectrum, "gray")
    # plt.title("Spectrum")
    # plt.show()
    out = []
    for i in range(0, size):
        tmp = []
        for j in range(0, size):
            tmp.append(spectrum[i][j])
        out.append(tmp)
    return list(np.concatenate(out).flat)


def fit_fft(train, size):
    train_learned = []
    for item in train:
        train_learned.append(grayscale_fourier_desc(item, size))
    return train_learned

def classify_signature(img, train, name, enc_dict, s):
    tmp = grayscale_fourier_desc(img, s)
    res = enc_dict[find_nearest(train, tmp)]
    out = []
    #print("Deskryptor - UNL-Fourier: ", res)
    out.append(1) if res == name else out.append(0)
    return out


def create_enc():
    n = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
         "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
    dict = defaultdict(lambda: None, {0: n[0], 1: n[0], 20: n[0], 21: n[0], 22: n[0],
                                      2: n[1], 3: n[1], 23: n[1], 24: n[1], 25: n[1],
                                      4: n[2], 5: n[2], 26: n[2], 27: n[2], 28: n[2],
                                      6: n[3], 7: n[3], 29: n[3], 30: n[3], 31: n[3],
                                      8: n[4], 9: n[4], 32: n[4], 33: n[4], 34: n[4],
                                      10: n[5], 11: n[5], 35: n[5], 36: n[5], 37: n[5],
                                      12: n[6], 13: n[6], 38: n[6], 39: n[6], 40: n[6],
                                      14: n[7], 15: n[7], 41: n[7], 42: n[7], 43: n[7],
                                      16: n[8], 17: n[8], 44: n[8], 45: n[8], 46: n[8],
                                      18: n[9], 19: n[9], 47: n[9], 48: n[9], 49: n[9]})
    return dict


def find_nearest(array, value):
    dist = []
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    return idx


def score(scores):
    # print(scores)
    res = 0
    for i in range(len(scores)):
        res += scores[i][0]
    res = res / len(scores) * 100
    return res


if __name__ == "__main__":
    # p = "./BAZA/learning/black-and-tan_coonhound/" + str(6) + ".png"
    # # # p = "./BAZA/learning/_TRAIN2/" + str(20) + ".png"
    # # p = "./BAZA/test.png"
    # s = 5
    # img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # res = unl_fourier_desc(img_test, s)
    # print(res)


    # train = []
    # for i in range(0, 20):
    # # for i in range(0, 50):
    #     p = "./BAZA/learning/_TRAIN/" + str(i) + ".png"
    #     # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
    #     img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    #     train.append(img)
    #
    # clf = fit_signature(train)
    # # print(clf)
    # # print(classify_signature(img_test, clf, "black-and-tan_coonhound"))
    #



    sizes = [5, 10, 25, 50, 100]
    for s in sizes:
        train = []
        for i in range(0, 20):
            # for i in range(0, 50):
            p = "./BAZA/no_bg/_TRAIN/" + str(i) + ".png"
            # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            train.append(img)

        clf = fit_fft(train, s)
        # print(clf)
        # classify_fft(img_test, clf, "borzoi", s)

        folders = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
                   "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
        results = []
        enc_dict = create_enc()
        for name in folders:
            for i in range(2, 10):
                # for i in range(5, 10):
                filename = "./BAZA/no_bg/" + name + "/" + str(i) + ".png"
                #print("-------------------------------------------------------------")
                #print(name + " - " + str(i) + ".png")
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                results.append(classify_signature(img, clf, name, enc_dict, s))
                #print("-------------------------------------------------------------")

        scores = score(results)
        print("-----------------------------------------------------------------------------")
        print("Poprawność - Grayscale-Fourier przy rozmiarze ", s, "x", s, ": ", scores, "%")
        print("-----------------------------------------------------------------------------")


