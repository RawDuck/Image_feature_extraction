import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.color import rgb2gray, rgba2rgb
from skimage import io
import numpy as np
import matplotlib.image as im
import scipy.ndimage as ndimage
from collections import defaultdict
from PIL import Image


def dominant_color_desc(filename):
    img = Image.open(filename)
    img = img.copy()
    n_colors = 50
    paletted = img.convert('P', palette=Image.ADAPTIVE, colors=n_colors)
    palette = paletted.getpalette()
    color_counts = sorted(paletted.getcolors(), reverse=True)
    colors = list()
    for i in range(n_colors):
        palette_index = color_counts[i][1]
        colors.append(palette[palette_index * 3:palette_index * 3 + 3][0])
    return colors


def color_mean_desc(filename):
    img = cv2.imread(filename)
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def color_dhv_desc(filename):
    img = cv2.imread(filename)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([image], [0], None, [359], [1, 360])
    v = cv2.calcHist([image], [2], None, [100], [0, 100])
    return list(np.concatenate([h / sum(h), v / sum(v)]).flat)


def color_hist_desc(filename):
    img = cv2.imread(filename)
    r = cv2.calcHist([img], [2], mask=None, histSize=[64], ranges=[0, 256])
    g = cv2.calcHist([img], [1], mask=None, histSize=[64], ranges=[0, 256])
    b = cv2.calcHist([img], [0], mask=None, histSize=[64], ranges=[0, 256])
    return np.ravel(b).tolist() + np.ravel(g).tolist() + np.ravel(r).tolist()


def gray_color_desc(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    size = 75
    out = []
    for i in range(0, size):
        tmp = []
        for j in range(0, size):
            tmp.append(img[i][j])
        out.append(tmp)
    return list(np.concatenate(out).flat)


def fit_colors(train):
    train_learned = []
    dominant = [dominant_color_desc(item) for item in train]
    mean = [color_mean_desc(item) for item in train]
    variance = [color_dhv_desc(item) for item in train]
    hist = [color_hist_desc(item) for item in train]
    gray = [gray_color_desc(item) for item in train]
    # for item in train:
    #     train_learned.append(dominant_color_desc(item))
    #     train_learned.append(color_mean_desc(item))
    #     train_learned.append(color_dhv_desc(item))
    #     train_learned.append(color_hist_desc(item))
    #     train_learned.append(gray_color_desc(item))
    train_learned = [dominant, mean, variance, hist, gray]

    return train_learned


def classifications(img, train, name, enc_dict):
    tmp1 = dominant_color_desc(img)
    tmp2 = color_mean_desc(img)
    tmp3 = color_dhv_desc(img)
    tmp4 = color_hist_desc(img)
    tmp5 = gray_color_desc(img)
    res = [enc_dict[find_nearest(train[0], tmp1)], enc_dict[find_nearest(train[1], tmp2)],
           enc_dict[find_nearest(train[2], tmp3)], enc_dict[find_nearest(train[3], tmp4)],
           enc_dict[find_nearest(train[4], tmp5)]]
    out = []
    #print("Deskryptor: ", res)
    #print(res)
    for item in res:
        out.append(1) if item == name else out.append(0)
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
    res = [0, 0, 0, 0, 0]
    for i in range(len(scores)):
        res[0] += scores[i][0]
        res[1] += scores[i][1]
        res[2] += scores[i][2]
        res[3] += scores[i][3]
        res[4] += scores[i][4]
    for i in range(len(res)):
        res[i] = res[i] / len(scores) * 100
    return res


if __name__ == "__main__":
    # p = "./BAZA/learning/black-and-tan_coonhound/" + str(6) + ".png"
    # # # p = "./BAZA/learning/_TRAIN2/" + str(20) + ".png"
    # # p = "./BAZA/test.png"
    # #img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # res = color_mean_desc(p)
    # print(res)

    train = []
    for i in range(0, 20):
        # for i in range(0, 50):
        p = "./BAZA/no_bg/_TRAIN/" + str(i) + ".png"
        # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
        train.append(p)

    clf = fit_colors(train)

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

            results.append(classifications(filename, clf, name, enc_dict))
            #print("-------------------------------------------------------------")

    scores = score(results)
    descs=["Dominant Colors", "Color Mean", "Color DHV", "Color Histogram", "Gray Color"]
    for i, d in enumerate(descs):
        print("-------------------------------------------------------------")
        print("Poprawność - " + d + ": ", scores[i], "%")
        print("-------------------------------------------------------------")

