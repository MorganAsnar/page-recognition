import numpy as np
import cv2
from collections import Counter

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

def nearestFeatures(query, ref, featuresIndex):
    indexes = []
    matches = matcher.knnMatch(query, ref, k=2)
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            refIndex = m.trainIdx
            indexes.append(featuresIndex[refIndex])
    identifiedClasses = Counter(indexes)
    try:
        confidenceRate = identifiedClasses.most_common(1)[0][1]/len(indexes)
        return identifiedClasses.most_common(1)[0][0], confidenceRate
    except:
        return None, -1