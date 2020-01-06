import cv2
import numpy as np
import os
import sklearn
import itertools as it
import pickle
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss


'''
Loading, resizing and saving images from INRIAPerson dataset as train and test sets
'''
def GetDataForTraining(folder):
    X_test = []
    X_train = []
    y_train = []
    hog = cv2.HOGDescriptor()
    pos_folder = os.path.join(folder, "Train/pos")
    for filename in os.listdir(pos_folder):
        img = cv2.imread(os.path.join(pos_folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 128), interpolation = cv2.INTER_AREA)
            X_train.extend(hog.compute(img).transpose())
            y_train.append(1)
            img = cv2.flip(img, 1)
            X_train.extend(hog.compute(img).transpose())
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
                while y_second < height:
                    x_first = 0
                    x_second = x_size
                    while x_second < width:
                        random_number = np.random.random_sample()
                        if random_number > 0.95:
                            roi = image[y_first:y_second, x_first:x_second]
                            roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                            roi_hog = hog.compute(roi_resized).transpose()
                            X_train.extend(roi_hog)
                            y_train.append(0)
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

            image = cv2.resize(image, (64, 128), interpolation = cv2.INTER_AREA)
            X_train.extend(hog.compute(image).transpose())
            y_train.append(0)

    y_test = []
    pos_folder = os.path.join(folder, "Test/pos")
    for filename in os.listdir(pos_folder):
        img = cv2.imread(os.path.join(pos_folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
            X_test.extend(hog.compute(img).transpose())
            y_test.append(1)

    neg_folder = os.path.join(folder, "Test/neg")
    for filename in os.listdir(neg_folder):
        img = cv2.imread(os.path.join(neg_folder, filename))
        if img is not None:
            img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)
            X_test.extend(hog.compute(img).transpose())
            y_test.append(0)
    return X_train, X_test, y_train, y_test


'''
Determining best hyperparameters
Train SVM and save it to file
'''
def GetBestHyperparameters(X_train, X_test, y_train, y_test):
    print("Number of images: ", len(X_train))
    ''' best = 1
    all_cs = range(-5, 21)
    all_gammas = range(-15, 6)
    kernel = 'rbf'
    error_test = []
    progress = 0
    for C in all_cs:
        for gamma in all_gammas:
            progress += 100 / (len(all_cs) * len(all_gammas))
            model = SVC(C = pow(2, C), kernel = kernel, gamma = pow(2, gamma)).fit(X_train, y_train)
            values = model.predict(X_test)
            error_test.append(zero_one_loss(y_test, values))
            print("Progress: {0:3.2f}%".format(progress))
            if best > zero_one_loss(y_test, values):
                best = zero_one_loss(y_test, values)
                best_c = C
                best_gamma = gamma'''
    best_c = 2
    best_gamma = 3e-2
    kernel = 'rbf'
    filename = "svm_model.sav"
    model = SVC(C = pow(2, best_c), kernel = kernel, gamma = best_gamma).fit(X_train, y_train)
    pickle.dump(model, open(filename, 'wb'))
    print("C: ", best_c, " gamma: ", best_gamma)


'''
Load SVM from file
'''
def LoadSVM():
    filename = "svm_model.sav"
    model = pickle.load(open(filename, 'rb'))
    correct_classification = 0
    values = model.predict(X_test)
    for i in range(len(values)):
        if values[i] == y_test[i]:
            correct_classification += 1

    print("Correctly classified " + str(correct_classification) + " examples out of " + str(len(values)))
    return model


'''
Check if there is a person on an image
'''
def CheckImage(model, imagefe):
    hog = cv2.HOGDescriptor()
    images = []
    images.append(cv2.imread("./one.png"))
    images.append(cv2.imread("./two.jpg"))
    images.append(cv2.imread("./three.jpg"))
    images.append(cv2.imread("./four.jpg"))
    images.append(cv2.imread("./five.jpg"))

    for image in images:
        person_detected = False
        x_size= 64
        y_size= 128
        height, width, channels = image.shape
        while x_size < width and y_size < height:
            y_first = 0
            y_second = y_size
            while y_second < height:
                x_first = 0
                x_second = x_size
                while x_second < width:
                    roi = image[y_first:y_second, x_first:x_second]
                    roi_resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
                    roi_hog = hog.compute(roi_resized).transpose()
                    value = model.predict(roi_hog)
                    '''if value == 0:
                        cv2.imshow('Not person', cv2.resize(roi, (256, 512)))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()'''
                    if value == 1:
                        person_detected = True
                        cv2.imshow('Person', cv2.resize(roi, (256, 512)))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
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
        '''if value == 0:
            cv2.imshow('Not person', cv2.resize(image, (256, 512)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''
        if value == 1:
            person_detected = True
            cv2.imshow('Person', cv2.resize(image, (256, 512)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if person_detected:
            cv2.imshow('Person in image', cv2.resize(image, (256, 512)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imshow('No person in image', cv2.resize(image, (256, 512)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


X_train, X_test, y_train, y_test = GetDataForTraining("./INRIAPerson")
GetBestHyperparameters(X_train, X_test, y_train, y_test)
model = LoadSVM()
CheckImage(model, "mememe")