import numpy as np
import cv2
import os
import time

pathToTrainDir = 'SET_YOUR_PATH_HERE'
pathToSaveDir = 'SET_YOUR_PATH_HERE'
if pathToSaveDir != '':
    pathToSaveDir = pathToSaveDir + '/'
resolution = (375,500)
rotated_resolution = (500,375)

if __name__ == '__main__':

    sift = cv2.xfeatures2d.SIFT_create()
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

    end = time.time()
    print("Done in {:.4f} seconds.".format(end-start))


    print("Saving...")
    np.save(pathToSaveDir + 'featuresDB.npy', featuresDB)
    np.save(pathToSaveDir + 'featuresIndex.npy', featuresIndex)
    print("done.")