import cv2
import numpy as np
import os
import sklearn
import pickle
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
import time


'''
Loading, resizing and saving images from INRIAPerson dataset as train and test sets
'''
def GetDataForTraining(folder):
    X_test = []
    X_train = []
    y_train = []
    hog = cv2.HOGDescriptor()
    pos_folder = os.path.join(folder, "Train/crowd")
    for filename in os.listdir(pos_folder):
        image = cv2.imread(os.path.join(pos_folder, filename))
        if image is not None:
            x_size = 64
            y_size = 128
            height, width, channels = image.shape
            while x_size < width and y_size < height:
                y_first = 0
                y_second = y_size
                while y_second <= height:
                    x_first = 0
                    x_second = x_size
                    while x_second <= width:
                        roi = image[y_first:y_second, x_first:x_second]
                        roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                        random_number = np.random.random_sample()
                        if random_number > 0.75:
                            # Transpose to have rows of images and columns of features
                            roi_hog = hog.compute(roi_resized).transpose()
                            X_train.extend(roi_hog)
                            y_train.append(1)
                        elif random_number < 0.25:
                            roi_resized = cv2.flip(roi_resized, 1)
                            X_train.extend(hog.compute(roi_resized).transpose())
                            y_train.append(1)
                        if x_second == width:
                            break
                        x_first += x_size // 2
                        x_second += x_size // 2
                        if x_second > width:
                            x_second = width
                            x_first = width - x_size
                    if y_second == height:
                        break
                    y_first += y_size // 2
                    y_second += y_size // 2
                    if y_second > height:
                        y_second = height
                        y_first = height - y_size
                x_size *= 2
                y_size *= 2

            image = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
            X_train.extend(hog.compute(image).transpose())
            y_train.append(1)
            image = cv2.flip(image, 1)
            X_train.extend(hog.compute(image).transpose())
            y_train.append(1)

    neg_folder = os.path.join(folder, "Train/neg")
    for filename in os.listdir(neg_folder):
        image = cv2.imread(os.path.join(neg_folder, filename))
        if image is not None:
            x_size = 64
            y_size = 128
            height, width, channels = image.shape
            while x_size < width and y_size < height:
                y_first = 0
                y_second = y_size
                while y_second <= height:
                    x_first = 0
                    x_second = x_size
                    while x_second <= width:
                        random_number = np.random.random_sample()
                        if random_number > 0.95:
                            roi = image[y_first:y_second, x_first:x_second]
                            roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                            roi_hog = hog.compute(roi_resized).transpose()
                            X_train.extend(roi_hog)
                            y_train.append(0)
                        if x_second == width:
                            break
                        x_first += x_size // 2
                        x_second += x_size // 2
                        if x_second > width:
                            x_second = width
                            x_first = width - x_size
                    if y_second == height:
                        break
                    y_first += y_size // 2
                    y_second += y_size // 2
                    if y_second > height:
                        y_second = height
                        y_first = height - y_size
                x_size *= 2
                y_size *= 2

            image = cv2.resize(image, (64, 128), interpolation = cv2.INTER_AREA)
            X_train.extend(hog.compute(image).transpose())
            y_train.append(0)

    neg_folder = os.path.join(folder, "Train/notcrowd")
    for filename in os.listdir(neg_folder):
        image = cv2.imread(os.path.join(neg_folder, filename))
        if image is not None:
            x_size = 64
            y_size = 128
            height, width, channels = image.shape
            while x_size < width and y_size < height:
                y_first = 0
                y_second = y_size
                while y_second <= height:
                    x_first = 0
                    x_second = x_size
                    while x_second <= width:
                        roi = image[y_first:y_second, x_first:x_second]
                        roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                        # Transpose to have rows of images and columns of features
                        random_number = np.random.random_sample()
                        if random_number > 0.75:
                            roi_hog = hog.compute(roi_resized).transpose()
                            X_train.extend(roi_hog)
                            y_train.append(0)
                        elif random_number < 0.25:
                            roi_resized = cv2.flip(roi_resized, 1)
                            X_train.extend(hog.compute(roi_resized).transpose())
                            y_train.append(0)
                        if x_second == width:
                            break
                        x_first += x_size // 2
                        x_second += x_size // 2
                        if x_second > width:
                            x_second = width
                            x_first = width - x_size
                    if y_second == height:
                        break
                    y_first += y_size // 2
                    y_second += y_size // 2
                    if y_second > height:
                        y_second = height
                        y_first = height - y_size
                x_size *= 2
                y_size *= 2

            image = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
            X_train.extend(hog.compute(image).transpose())
            y_train.append(0)
            image = cv2.flip(image, 1)
            X_train.extend(hog.compute(image).transpose())
            y_train.append(0)

    y_test = []
    images_folder = os.path.join(folder, "Test/crowd")
    for filename in os.listdir(images_folder):
        img = cv2.imread(os.path.join(images_folder, filename))
        if img is not None:
            X_test.append(img)
            #X_test.append(cv2.flip(img, 1))

    truth_folder = os.path.join(folder, "Test/crowdtrue")
    for filename in os.listdir(truth_folder):
        img = cv2.imread(os.path.join(truth_folder, filename))
        if img is not None:
            y_test.append(img)
            #y_test.append(cv2.flip(img, 1))

    return X_train, X_test, y_train, y_test


'''
Determining best hyperparameters
Train SVM and save it to file
'''
def GetBestHyperparameters(X_train, X_test, y_train, y_test):
    print("Number of images: ", len(X_train))
    best = 0
    all_cs = [0, 0.5, 1, 1.5, 2]
    all_gammas = [-6, -5.5, -5, -4.5, -4]
    kernel = 'rbf'
    progress = 0
    start_time = time.time()
    best_gamma = 2 ** -5
    for C in all_cs:
        '''progress += 100 / (len(all_cs) + len(all_gammas))
        model = SVC(C=pow(2, C), kernel=kernel, gamma=best_gamma).fit(X_train, y_train)
        print("Progress: {0:3.2f}%".format(progress))
        print("--- Time elapsed: %s seconds ---" % (time.time() - start_time))
        tmp = CheckImage(model, X_test, y_test)
        if best < tmp:
            best = tmp
            best_c = 2 ** C'''
        for gamma in all_gammas:
            progress += 100 / (len(all_cs) * len(all_gammas))
            model = SVC(C = 2 ** C, kernel = kernel, gamma = pow(2, gamma)).fit(X_train, y_train)
            print("Progress: {0:3.2f}%".format(progress))
            print("--- Time elapsed: %s seconds ---" % (time.time() - start_time))
            tmp = CheckImage(model, X_test, y_test)
            print("C: ", C, " gamma: ", gamma, " acc: ", tmp * 100)
            if best < tmp:
                print("Best: ", tmp * 100)
                best = tmp
                best_c = C
                best_gamma = 2 ** gamma
    print("Best accuracy: ", tmp * 100, "%")
    filename = "svm_crowd_background_separate_model3.sav"
    model = SVC(C=best_c, kernel=kernel, gamma=best_gamma).fit(X_train, y_train)
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print("C: ", best_c, " gamma: ", best_gamma)
    return best_c, best_gamma


'''
Load SVM from file
'''
def loadSVM(filename="svm_crowd_background_separate_model2.sav"):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model


def checkSVM(model, X_test, y_test):
    values = model.predict(X_test)
    false_positive = 0
    false_negative = 0
    for value, true in zip(values, y_test):
        if true == 1 and value == 0:
            false_negative += 1
        if true == 0 and value == 1:
            false_positive += 1
    correct_classification = np.sum(values == y_test)
    print("Correctly classified {}/{} ({}%)".format(correct_classification, len(values), correct_classification / len(values)))
    print("False positives: ", false_positive)
    print("False negatives: ", false_negative)


'''
Check if there is a person on an image
'''
def CheckImage(model, X_test, y_test):
    hog = cv2.HOGDescriptor()
    average = 0
    for image, img in zip(X_test, y_test):
        x_size = 128
        y_size = 256
        black_white = image.copy()
        black_white[:, :, :] = 255
        show_image = image.copy()
        height, width, channels = image.shape
        while x_size < width and y_size < height:
            y_first = 0
            y_second = y_size
            while y_second <= height:
                x_first = 0
                x_second = x_size
                while x_second <= width:
                    roi = image[y_first:y_second, x_first:x_second]
                    roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                    roi_hog = hog.compute(roi_resized).transpose()
                    value = model.predict(roi_hog)
                    if value == 1:
                        black_white[y_first:y_second, x_first:x_second, :] = 0           # Color black
                        show_image[y_first:y_second, x_first:x_second, 1] = 240
                    if x_second == width:
                        break
                    x_first += int(x_size / 2)
                    x_second += int(x_size / 2)
                    if x_second > width:
                        x_second = width
                        x_first = width - x_size
                if y_second == height:
                    break
                y_first += int(y_size / 2)
                y_second += int(y_size / 2)
                if y_second > height:
                    y_second = height
                    y_first = height - y_size
            x_size *= 2
            y_size *= 2

        hit = np.sum(black_white[:, :, 0] == img[:, :, 0])
        hit_acc = hit / (width * height)
        average += hit_acc

    average /= len(X_test)
    return average


def separateCrowdBackground(model):
    hog = cv2.HOGDescriptor()
    images = []
    images.append(cv2.imread("./crowdone.jpg"))
    images.append(cv2.imread("./crowdtwo.jpg"))
    images.append(cv2.imread("./crowdthree.jpg"))
    images.append(cv2.imread("./crowdfour.jpg"))
    images.append(cv2.imread("./crowdfive.jpg"))
    images.append(cv2.imread("./crowdsix.jpg"))
    images.append(cv2.imread("./crowdseven.jpg"))
    images.append(cv2.imread("./crowdeight.jpg"))
    images.append(cv2.imread("./crowdnine.jpg"))
    images.append(cv2.imread("./crowdten.jpg"))

    for image in images:
        x_size= 128
        y_size= 256
        img = image.copy()
        height, width, channels = image.shape
        while x_size < width and y_size < height:
            y_first = 0
            y_second = y_size
            while y_second <= height:
                x_first = 0
                x_second = x_size
                while x_second <= width:
                    roi = image[y_first:y_second, x_first:x_second]
                    roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                    roi_hog = hog.compute(roi_resized).transpose()
                    value = model.predict(roi_hog)
                    if value == 1:
                        img[y_first:y_second, x_first:x_second, 1] = 200
                    if x_second == width:
                        break
                    x_first += int(x_size / 2)
                    x_second += int(x_size / 2)
                    if x_second > width:
                        x_second = width
                        x_first = width - x_size
                if y_second == height:
                    break
                y_first += int(y_size / 2)
                y_second += int(y_size / 2)
                if y_second > height:
                    y_second = height
                    y_first = height - y_size
            x_size *= 2
            y_size *= 2
        roi_resized = cv2.resize(image, (64, 128), interpolation=cv2.INTER_AREA)
        roi_hog = hog.compute(roi_resized).transpose()
        value = model.predict(roi_hog)
        if value == 1:
            img[:, :, 1] = 200
        cv2.imshow('Crowd-background separated', cv2.resize(img, (512, 512)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def trainSVM(X_train, y_train, C, gamma, save, filename="tmp.sav"):
    start_time = time.time()
    model = SVC(C=C, kernel='rbf', gamma=gamma).fit(X_train, y_train)
    print("-- trained for %s seconds --" % (time.time() - start_time))
    if save:
        with open(filename, 'wb') as file:
            pickle.dump(model, file) 
    return model


X_train, X_test, y_train, y_test = GetDataForTraining("./INRIAPerson")
#C, gamma = GetBestHyperparameters(X_train, X_test, y_train, y_test)
print("Number of images: ", len(X_train))
best_C = 2 ** 1.5
best_gamma = 2 ** -5.0
print("C: ", best_C, " gamma: ", best_gamma)
filename = "svm_crowd_background_separate_model.sav"
#model = trainSVM(X_train, y_train, C=best_C, gamma=best_gamma, save=True, filename = filename)
model = loadSVM(filename)
#checkSVM(model, X_test, y_test)
#avg = CheckImage(model, X_test, y_test)
#print("Avg = ", avg * 100)
separateCrowdBackground(model)