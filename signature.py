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


def signature_desc(img):
    # thresh = cv2.Canny(img, 30, 200)
    ret, thresh = cv2.threshold(img, 0, 200, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=len)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    centroid = [cX, cY]
    # thresh = cv2.Canny(img, 30, 200)
    # plt.imshow(thresh, cmap='gray')
    # plt.show()
    # cv2.circle(thresh, (cX, cY), 5, (255, 255, 255), -1)
    # cv2.putText(thresh, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.imshow("Image", thresh)
    # cv2.waitKey(0)
    dist = []
    for c in contour:
        dist.append(np.linalg.norm(c - centroid))
    return dist


def fit_signature(train):
    train_learned = []
    for item in train:
        train_learned.append(signature_desc(item))
    default_size = find_max_list(train_learned)
    train_learned2 = interpolate_to_fixed_size(train_learned, default_size)
    return train_learned2


def interpolate_to_fixed_size(given_list, default_size):
    for i in range(len(given_list)):
        old_indices = np.arange(0, len(given_list[i]))
        new_indices = np.linspace(0, len(given_list[i]) - 1, default_size)
        spl = UnivariateSpline(old_indices, given_list[i], k=1, s=0)
        given_list[i] = spl(new_indices)
    return given_list


def find_max_list(given_list):
    list_len = [len(i) for i in given_list]
    return max(list_len)


def classify_signature(img, train, name, enc_dict):
    tmp = signature_desc(img)
    default_size = find_max_list(train)
    img2 = interpolate_to_fixed_size([tmp], default_size)
    res = enc_dict[find_nearest(train, img2)]
    out = []
    print("Deskryptor - Sygnatura: ", res)
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
    # p = "./BAZA/learning/borzoi/" + str(4) + ".png"
    # # p = "./BAZA/learning/_TRAIN2/" + str(20) + ".png"
    # p = "./BAZA/test.png"
    # img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # res = signature_desc(img_test)
    # print(res)


    train = []
    for i in range(0, 20):
    # for i in range(0, 50):
        p = "./BAZA/learning/_TRAIN/" + str(i) + ".png"
        # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        train.append(img)

    clf = fit_signature(train)
    # print(clf)
    # print(classify_signature(img_test, clf, "black-and-tan_coonhound"))

    folders = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
               "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
    results = []
    enc_dict = create_enc()
    for name in folders:
        for i in range(2, 10):
        # for i in range(5, 10):
            filename = "./BAZA/learning/" + name + "/" + str(i) + ".png"
            print("-------------------------------------------------------------")
            print(name + " - " + str(i) + ".png")
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            results.append(classify_signature(img, clf, name, enc_dict))
            print("-------------------------------------------------------------")

    scores = score(results)
    print("-------------------------------------------------------------")
    print("Poprawność - Sygnatura: ", scores, "%")
    print("-------------------------------------------------------------")

    # plt.imshow(img_test, cmap=plt.cm.gray)
    # plt.show()
    # print(img)
    #example_of_descs(img)


