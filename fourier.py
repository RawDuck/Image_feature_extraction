import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.color import rgb2gray,rgba2rgb
from skimage import io
import numpy as np
import matplotlib.image as im


def fourier_desc(img, size):
    img_fft = np.fft.fft2(img)
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
    for i, item in enumerate(train):
        train_learned.append(fourier_desc(item, size))
    return train_learned


def classify_fft(img, train, name, size):
    tmp = fourier_desc(img, size)
    #print(tmp)
    res = enc_idx(find_nearest(train, tmp))
    out = []
    # print("Deskryptor - Fourier: ", res)
    out.append(1) if res == name else out.append(0)
    return out


def enc_idx(i):
    n = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
               "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
    if i == 0 or i == 1 or i == 20 or i == 21 or i == 22:
        return n[0]
    elif i == 2 or i == 3 or i == 23 or i == 24 or i == 25:
        return n[1]
    elif i == 4 or i == 5 or i == 26 or i == 27 or i == 28:
        return n[2]
    elif i == 6 or i == 7 or i == 29 or i == 30 or i == 31:
        return n[3]
    elif i == 8 or i == 9 or i == 32 or i == 33 or i == 34:
        return n[4]
    elif i == 10 or i == 11 or i == 35 or i == 36 or i == 37:
        return n[5]
    elif i == 12 or i == 13 or i == 38 or i == 39 or i == 40:
        return n[6]
    elif i == 14 or i == 15 or i == 41 or i == 42 or i == 43:
        return n[7]
    elif i == 16 or i == 17 or i == 44 or i == 45 or i == 46:
        return n[8]
    elif i == 18 or i == 19 or i == 47 or i == 48 or i == 49:
        return n[9]
    else:
        return None


def find_nearest(array, value):
    dist=[]
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
    # p = "./BAZA/learning/borzoi/" + str(4) + ".png"
    # img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # res = fourier_desc(img_test, 2)
    # print(res)
    sizes=[5, 10, 25, 50, 100]
    for s in sizes:
        train = []
        for i in range(0, 20):
        # for i in range(0, 50):
            p = "./BAZA/learning/_TRAIN/" + str(i) + ".png"
            # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            train.append(img)

        clf = fit_fft(train, s)
        #print(clf)
        # classify_fft(img_test, clf, "borzoi", s)

        folders = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
                    "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
        results = []
        for name in folders:
            for i in range(2, 10):
            # for i in range(5, 10):
                filename = "./BAZA/learning/" + name + "/" + str(i) + ".png"
                # print("-------------------------------------------------------------")
                # print(name + " - " + str(i) + ".png")
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                results.append(classify_fft(img, clf, name, s))
                # print("-------------------------------------------------------------")

        scores = score(results)
        print("-------------------------------------------------------------")
        print("Poprawność - Fourier przy rozmiarze ",s,"x",s,": ", scores, "%")
        print("-------------------------------------------------------------")

    # plt.imshow(img_test, cmap=plt.cm.gray)
    # plt.show()
    # print(img)
    #example_of_descs(img)


