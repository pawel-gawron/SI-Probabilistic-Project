from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import JointProbabilityDistribution
from pgmpy.factors.discrete import JointProbabilityDistribution as JPD

from pathlib import Path
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt

def calcHistBoundBox(boundBox):
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    v_ranges = [0, 256]

    h_bins = 50
    s_bins = 60
    v_bins = 60

    boundBox = cv2.resize(boundBox, (500, 500))

    histH = cv2.calcHist([boundBox],[0], None, [h_bins], h_ranges, accumulate=False)
    cv2.normalize(histH, histH, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    histS = cv2.calcHist([boundBox],[1], None, [s_bins], s_ranges, accumulate=False)
    cv2.normalize(histS, histS, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    histV = cv2.calcHist([boundBox],[2], None, [v_bins], v_ranges, accumulate=False)
    cv2.normalize(histV, histV, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    plt.plot(histH,color = 'red')
    plt.plot(histS,color = 'green')
    plt.plot(histV,color = 'blue')
    # boundBox_bgr = cv2.cvtColor(boundBox, cv2.COLOR_HSV2BGR)
    # plt.imshow(boundBox_bgr)
    # plt.axis('off')
    # plt.show()
    # cv2.imshow("image", boundBox_bgr)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
    # plt.show()

    # boundBox_rgb = cv2.cvtColor(boundBox, cv2.COLOR_HSV2RGB)
    # plt.imshow(boundBox_rgb)
    # plt.axis('off')
    # plt.show()

    return [histH, histS, histV]

def compareHistBoundBox(boundBoxesPreviousHist, boundBoxesCurrentHist):
    similarityVect = []

    for counterPrev, histPrev in enumerate(boundBoxesPreviousHist):
        for counterCurr, histCurr in enumerate(boundBoxesCurrentHist):
            histCompH = cv2.compareHist(histPrev[counterPrev][0], histCurr[counterCurr][0], cv2.HISTCMP_BHATTACHARYYA)
            histCompS = cv2.compareHist(histPrev[counterPrev][1], histCurr[counterCurr][1], cv2.HISTCMP_BHATTACHARYYA)
            histCompV = cv2.compareHist(histPrev[counterPrev][2], histCurr[counterCurr][2], cv2.HISTCMP_BHATTACHARYYA)

            similarity = 1-((histCompH + histCompS + histCompV)/3)
            similarityVect.append(similarity)

    return similarityVect


def computeProbability(imagePath, boundingBoxPath):
    imagesPath = imagePath
    boundingBoxFile = boundingBoxPath
    boundingBoxNumberPrev = None
    histogramsPrev = []
    histogramsCurrent = []

    for imageNumber in range(len(imagesPath)):
        coordinatesBoundingBoxes = []
        imageName = boundingBoxFile.readline().rstrip("\n")
        image = cv2.imread(str(imagePath[imageNumber]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # print(imageName)

        if not imageName:
            break

        # print("imageName: ", imageName)

        boundingBoxNumber = boundingBoxFile.readline().rstrip("\n")
        # print(boundingBoxNumber)

        if boundingBoxNumber == 0:
            boundingBoxNumberPrev = boundingBoxNumber
            continue

        if boundingBoxNumberPrev == 0 or imageNumber == 0:
            boundingBoxNumberPrev = None
            histogramsPrev = histogramsCurrent

            for _ in range(int(float(boundingBoxNumber))):
                print("-1 ")

        histogramsPrev = histogramsCurrent
        histogramsCurrent = []
        for _ in range(int(boundingBoxNumber)):
            coordinates = boundingBoxFile.readline().rstrip("\n").split(" ")
            coordinatesBoundingBoxes.append(coordinates)
            x = int(float(coordinates[0]))
            y = int(float(coordinates[1]))
            w = int(float(coordinates[2]))
            h = int(float(coordinates[3]))
            histogramsCurrent.append(calcHistBoundBox(image[y:y+h, x:x+w]))

        if boundingBoxNumberPrev != None:
            comp = compareHistBoundBox(histogramsPrev, histogramsCurrent)
            print(comp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    args = parser.parse_args()

    dataDir = Path(args.data_dir)
    imagesDir = Path(os.path.join(str(dataDir) + '/', 'frames'))
    boundingBoxDir = Path(os.path.join(dataDir, 'bboxes.txt'))

    imagesPath = sorted([image_path for image_path in imagesDir.iterdir() if image_path.name.endswith('.jpg')])

    file = open(boundingBoxDir, 'r')

    computeProbability(imagesPath, file)
    # for image_path in imagesPath:
    #     image = cv2.imread(str(image_path))
    #     if image is None:
    #         print(f'Error loading image {image_path}')
    #         continue