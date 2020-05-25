import numpy as np
import cv2
from neirestUtil import nearestFeatures
import sys

pathToSaveDir = 'SET_YOUR_PATH_HERE'
if pathToSaveDir != 'SET_YOUR_PATH_HERE':
    pathToSaveDir = pathToSaveDir + '/'
resolution = (375,500)
rotated_resolution = (500,375)

if __name__ == '__main__':

    sift = cv2.xfeatures2d.SIFT_create()
    try:
        featuresDB = np.load(pathToSaveDir + 'featuresDB.npy')
        featuresIndex = np.load(pathToSaveDir + 'featuresIndex.npy')
    except:
        sys.exit('Database not found. Considere running CreateDB.py first, '
                 'or check the path to the save directory.')
    featuresIndex = np.ndarray.tolist(featuresIndex)
    nearest = None

    path = sys.argv[1].replace("\\", "\\\\")
    queryPic = cv2.imread(path)
    if queryPic is not None:
        if queryPic.shape[1] < queryPic.shape[0]:
            queryPic = cv2.resize(queryPic, resolution)
        else:
            queryPic = cv2.resize(queryPic, rotated_resolution)
        _, descriptors = sift.detectAndCompute(queryPic, None)
        nearest = nearestFeatures(descriptors, featuresDB, featuresIndex)
    else:
        sys.exit('Invalid path')

    print(nearest)
