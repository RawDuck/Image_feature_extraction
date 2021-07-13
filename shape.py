import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.filters import try_all_threshold
import os
from skimage.color import rgb2gray,rgba2rgb
from skimage import io
from skimage.filters import threshold_otsu, threshold_minimum, threshold_triangle, threshold_local, rank, threshold_yen
from skimage.morphology import disk
import numpy as np
import matplotlib.image as im

def preprocessing():
    p = "./BAZA/no_bg/komondor/" + str(0) + ".png"
    img = io.imread(p)
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    img2 = rgb2gray(rgba2rgb(img))
    kernel = np.ones((5,5), np.uint8)
    img2 = cv2.dilate(img2, kernel, iterations=1)
    fig, ax = try_all_threshold(img2, figsize=(10, 8), verbose=False)
    plt.show()

    folders = ["borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
               "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier",
               "whippet", "black-and-tan_coonhound"]
    for name in folders:
        for i in range(0, 10):
            filename = "./BAZA/no_bg/" + name + "/" + str(i) + ".png"
            img = io.imread(filename)
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7)
            img2 = rgb2gray(rgba2rgb(img))
            kernel = np.ones((5,5), np.uint8)
            img2 = cv2.dilate(img2, kernel, iterations=2)
            thresh = threshold_triangle(img2)
            binary = img2 < thresh


            # selem = disk(10)
            # local_otsu = rank.otsu(img2, selem)
            # threshold_global_otsu = threshold_otsu(img2)
            # global_otsu = img2 >= local_otsu

            # plt.imshow(binary, cmap=plt.cm.gray)
            # plt.axis('off')
            # plt.show()

            path_img_target = os.path.join("./BAZA/binarized/" + name + "/" + str(i) + ".png")
            io.imsave(path_img_target, skimage.img_as_ubyte(binary))
            print("./BAZA/binarized/" + name + "/" + str(i) + ".png ---- DONE" )


def area_desc(img):
    return np.count_nonzero(img == 0)


def perimeter_desc(img):
    edged = cv2.Canny(img, 30, 200)
    #contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #plt.imshow(edged, cmap='gray')
    #plt.show()
    return np.count_nonzero(edged == 255)


def roundness_desc(img):
    area = np.count_nonzero(img == 0)
    edged = cv2.Canny(img, 30, 200)
    perimeter = np.count_nonzero(edged == 255)
    return np.power(perimeter, 2) / (4 * np.pi * area)


def compactness_desc(img):
    area = np.count_nonzero(img == 0)
    edged = cv2.Canny(img, 30, 200)
    perimeter = np.count_nonzero(edged == 255)
    return np.power(perimeter, 2) / area


# https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
def convex_hull_desc(img):
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    for i in range(len(contours)):
        color_contours = (0, 255, 0)
        color = (255, 0, 0)
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    # plt.imshow(drawing, cmap=plt.cm.gray)
    # plt.show()
    edged = cv2.Canny(drawing, 30, 200)
    return np.count_nonzero(edged == 255)


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
    return out


def fit_fft(train, size):
    train_learned = []
    for i, item in enumerate(train):
        train_learned.append(fourier_desc(item, size))
    return train_learned


def classify_fft(img, train, name, size):
    tmp = fourier_desc(img, size)
    print(tmp)
    res = enc_idx(find_nearest(train, tmp))
    out = []
    print("Deskryptor - Fourier: ", res)
    out.append(1) if res == name else out.append(0)
    return out


def example_of_descs(img):
    print("------Wartości deskryptorów------")
    print("Pole powierzchni: ", area_desc(img))
    print("Obwód: ", perimeter_desc(img))
    print("Kołowość: ", roundness_desc(img))
    print("Zwartość: ", compactness_desc(img))
    print("Obwód powłoki wypukłej: ", convex_hull_desc(img))
    print("------Wartości deskryptorów------")


def fit(train):
    train_learned = []
    area = []
    per = []
    round = []
    compact = []
    convex = []
    for i, item in enumerate(train):
        area.append(area_desc(item))
        per.append(perimeter_desc(item))
        round.append(roundness_desc(item))
        compact.append(compactness_desc(item))
        convex.append(convex_hull_desc(item))
    train_learned.append(area)
    train_learned.append(per)
    train_learned.append(round)
    train_learned.append(compact)
    train_learned.append(convex)
    return train_learned

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
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def classify(img, train, name):
    tmp = [area_desc(img), perimeter_desc(img), roundness_desc(img), compactness_desc(img), convex_hull_desc(img)]
    res = [enc_idx(find_nearest(train[0], tmp[0])), enc_idx(find_nearest(train[1], tmp[1])),
          enc_idx(find_nearest(train[2], tmp[2])), enc_idx(find_nearest(train[3], tmp[3])),
          enc_idx(find_nearest(train[4], tmp[4]))]
    out = []
    print("Deskryptor - pole powierzchni: ", res[0])
    print("Deskryptor - obwód: ", res[1])
    print("Deskryptor - kołowość: ", res[2])
    print("Deskryptor - zwartość: ", res[3])
    print("Deskryptor - obwód powłoki wypukłej: ", res[4])

    out.append(1) if res[0] == name else out.append(0)
    out.append(1) if res[1] == name else out.append(0)
    out.append(1) if res[2] == name else out.append(0)
    out.append(1) if res[3] == name else out.append(0)
    out.append(1) if res[4] == name else out.append(0)
    return out


def score(scores):
    # print(scores)
    area = 0
    peri = 0
    roun = 0
    comp = 0
    conv = 0
    for i in range(len(scores)):
        area += scores[i][0]
        peri += scores[i][1]
        roun += scores[i][2]
        comp += scores[i][3]
        conv += scores[i][4]
    area = area / len(scores) * 100
    peri = peri / len(scores) * 100
    roun = roun / len(scores) * 100
    comp = comp / len(scores) * 100
    conv = conv / len(scores) * 100
    return [area, peri, roun, comp, conv]


if __name__ == "__main__":
    p = "./BAZA/learning/borzoi/" + str(4) + ".png"
    img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

    # clf = fit(train)
    # # print(clf)
    # classify(img_test, clf)

    train = []
    for i in range(0, 20):
        # for i in range(0, 50):
        p = "./BAZA/learning/_TRAIN/" + str(i) + ".png"
        # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        train.append(img)

    clf = fit(train)

    folders = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
                "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
    results = []
    for name in folders:
        for i in range(2, 10):
        # for i in range(5, 10):
            filename = "./BAZA/learning/" + name + "/" + str(i) + ".png"
            print("-------------------------------------------------------------")
            print(name + " - " + str(i) + ".png")
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            results.append(classify(img, clf, name))
            print("-------------------------------------------------------------")

    scores = score(results)
    #print(results)
    print("Poprawność - pole powierzchni: ", scores[0], "%")
    print("Poprawność - obwód: ", scores[1], "%")
    print("Poprawność - kołowość: ", scores[2], "%")
    print("Poprawność - zwartość: ", scores[3], "%")
    print("Poprawność - obwód powłoki wypukłej: ", scores[4], "%")

    # plt.imshow(img, cmap=plt.cm.gray)
    # plt.show()
    # print(img)
    # example_of_descs(img)


