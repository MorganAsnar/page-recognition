import numpy as np
import numpy.linalg as LA
import cv2
import os
import matplotlib.pyplot as plt
import time
from neirestUtil import nearestFeatures

pathToDir = 'SET_YOUR_PATH_HERE'
pathToTrainDir = pathToDir + '/training'
pathToValDir = pathToDir + '/validation'
resolution = (375,500)
rotated_resolution = (500,375)


if __name__ == '__main__':

    sift = cv2.xfeatures2d.SIFT_create()
    timeResults = []
    print("Loading training images...")
    trainClasses=[]
    picPaths = []
    dirList = os.listdir(pathToTrainDir)
    start = time.time()
    for directory in dirList:
        picList = os.listdir(pathToTrainDir+'/'+directory)
        for pic in picList:
            picPaths.append(pathToTrainDir+'/'+directory+'/'+pic)
            trainClasses.append(directory)

    featuresIndex = []
    featuresDB = []
    for i in range(len(picPaths)):
        trainPic = cv2.imread(picPaths[i], 0)
        if trainPic.shape[1] < trainPic.shape[0]:
            trainPic = cv2.resize(trainPic, resolution)
        else:
            trainPic = cv2.resize(trainPic, rotated_resolution)
        keypoints, trainPicDes = sift.detectAndCompute(trainPic, None)
        try:
            featuresDB = np.concatenate((featuresDB, trainPicDes), axis=0)
        except:
            featuresDB = trainPicDes
        featuresIndex.append([trainClasses[i]] * len(keypoints))
        print("Pic {} loaded".format(i + 1), end='\r')

    featuresIndex = np.asarray(featuresIndex)
    featuresIndex = np.concatenate((featuresIndex), axis=None)
    featuresIndex = np.ndarray.tolist(featuresIndex)

    end = time.time()
    print("Done in {} seconds.".format(end-start))


    print("Loading validation images...")
    valClasses = []
    picPaths = []
    dirList = os.listdir(pathToValDir)
    start = time.time()
    for directory in dirList:
        picList = os.listdir(pathToValDir + '/' + directory)
        for pic in picList:
            picPaths.append(pathToValDir + '/' + directory + '/' + pic)
            valClasses.append(directory)

    valMatrix = []
    for i in range(len(picPaths)):
        valPic = cv2.imread(picPaths[i], 0)
        if valPic.shape[1] < valPic.shape[0]:
            valPic = cv2.resize(valPic, resolution)
        else:
            valPic = cv2.resize(valPic, rotated_resolution)
        _, valPicDes = sift.detectAndCompute(valPic, None)
        valMatrix.append(valPicDes)
        print("Pic {} loaded".format(i + 1), end='\r')

    end = time.time()
    print("Done in {} seconds.".format(end - start))


    results = []
    start = time.time()
    confidences = []
    for i, valDes in enumerate(valMatrix):
        print("Progress : {:.2f}%".format(100*i/len(valMatrix)), end='\r')
        nearest, confidenceRate = nearestFeatures(valDes, featuresDB, featuresIndex)
        results.append(nearest)
        confidences.append(confidenceRate)

    end = time.time()
    precision = 0
    noneCounter=0
    trapsAvoided=0
    for i in range(len(results)):
        if results[i] is not None:
            if results[i]==valClasses[i]:
                precision+=1
        else:
            if valClasses[i]=="Traps":
                trapsAvoided+=1
            else:
                noneCounter+=1

    precision = precision/(len(results)-noneCounter-trapsAvoided)
    noneCounter = noneCounter/len(results)
    averageTime = (end-start)/len(valMatrix)
    trapsAvoided = trapsAvoided/len(os.listdir(pathToValDir + '/Traps'))

    plt.figure(figsize=(16, 10))
    plt.semilogy(range(len(confidences)), confidences, label='confidence rate')
    plt.legend()
    plt.xlabel('index')
    plt.ylabel('confidence rate')
    plt.show()

    print("Results: precision rate of {:.4f}\n"
            "Non-acceptable picture rate: {:.4f}\n"
            "Traps detection rate: {:.4f}\n"
            "It took {} seconds per picture."
          .format(precision, noneCounter, trapsAvoided, averageTime))
