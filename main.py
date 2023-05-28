from pgmpy.models import BayesianModel, FactorGraph
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.factors.discrete import JointProbabilityDistribution, DiscreteFactor
from collections import OrderedDict

from pathlib import Path
import numpy as np
import cv2
import os
import argparse
import matplotlib.pyplot as plt
from itertools import combinations

def calcHistBoundBox(boundBox):
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    v_ranges = [0, 256]

    h_bins = 50
    s_bins = 60
    v_bins = 60

    boundBox = cv2.resize(boundBox, (500, 500))

    histH = cv2.calcHist([boundBox],[0], None, [h_bins], h_ranges, accumulate=False)
    cv2.normalize(histH, histH, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    histS = cv2.calcHist([boundBox],[1], None, [s_bins], s_ranges, accumulate=False)
    cv2.normalize(histS, histS, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    histV = cv2.calcHist([boundBox],[2], None, [v_bins], v_ranges, accumulate=False)
    cv2.normalize(histV, histV, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

    # plt.plot(histH,color = 'red')
    # plt.plot(histS,color = 'green')
    # plt.plot(histV,color = 'blue')
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

def compareHistBoundBox(boundBoxesCurrentHist, boundBoxesPreviousHist, factorGraph):
    # print("boundBoxesPreviousHist: ", len(boundBoxesPreviousHist))
    # print("boundBoxesCurrentHist: ", len(boundBoxesCurrentHist))
    # print("boundBoxesPreviousHist: ", boundBoxesPreviousHist)

    for counterCurr, histCurr in enumerate(boundBoxesCurrentHist):
        similarityVect = []
        for counterPrev, histPrev in enumerate(boundBoxesPreviousHist):
    # for i in range(len(boundBoxesCurrentHist)):
    #     for j in range(len(boundBoxesPreviousHist)):
            # print("boundBoxesCurrentHist[0]: ", histCurr)
            # histCompH = cv2.compareHist(boundBoxesPreviousHist[i][0], boundBoxesCurrentHist[j][0], cv2.HISTCMP_BHATTACHARYYA)
            # histCompS = cv2.compareHist(boundBoxesPreviousHist[i][1], boundBoxesCurrentHist[j][1], cv2.HISTCMP_BHATTACHARYYA)
            # histCompV = cv2.compareHist(boundBoxesPreviousHist[i][2], boundBoxesCurrentHist[j][2], cv2.HISTCMP_BHATTACHARYYA)
            histCompH = cv2.compareHist(histPrev[0], histCurr[0], cv2.HISTCMP_CORREL)
            histCompS = cv2.compareHist(histPrev[1], histCurr[1], cv2.HISTCMP_CORREL)
            histCompV = cv2.compareHist(histPrev[2], histCurr[2], cv2.HISTCMP_CORREL)
            # plt.plot(histPrev,color = 'blue')
            # plt.show()

            similarity = (histCompH + histCompS + histCompV)/3
            if similarity <= 0.0:
                similarity = 0.01
            similarityVect.append(similarity)
            # print(histCompH)
            # print("[histPrev[0]]: ", histPrev[0])
        print(len(boundBoxesPreviousHist) + 1)
        print([[0.3] + similarityVect])
        factor = DiscreteFactor([str(counterCurr)], [len(boundBoxesPreviousHist) + 1], [[0.3] + similarityVect])
        print(factor)
        factorGraph.add_factors(factor)
        factorGraph.add_edge(str(counterCurr),factor)


def computeProbability(imagePath, boundingBoxPath):
    imagesPath = imagePath
    boundingBoxFile = boundingBoxPath
    boundingBoxNumberPrev = None
    histogramsPrev = []
    histogramsCurrent = []
    probNewObject = 0.3

    for imageNumber in range(len(imagesPath)):
        coordinatesBoundingBoxes = []
        nodes = []
        factorGraph = FactorGraph()
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
            boundingBoxNumberPrev = boundingBoxNumber
            histogramsPrev = histogramsCurrent

            for _ in range(int(float(boundingBoxNumber))):
                print("-1 ")
                coordinates = boundingBoxFile.readline().rstrip("\n").split(" ")
                x = int(float(coordinates[0]))
                y = int(float(coordinates[1]))
                w = int(float(coordinates[2]))
                h = int(float(coordinates[3]))
                histogramsCurrent.append(calcHistBoundBox(image[y:y+h, x:x+w]))
            continue

        histogramsPrev = histogramsCurrent
        boundingBoxNumberPrev = boundingBoxNumber
        histogramsCurrent = []
        for bbObject in range(int(boundingBoxNumber)):
            nodes.append("Camera_object_" + str(bbObject))
            coordinates = boundingBoxFile.readline().rstrip("\n").split(" ")
            coordinatesBoundingBoxes.append(coordinates)
            x = int(float(coordinates[0]))
            y = int(float(coordinates[1]))
            w = int(float(coordinates[2]))
            h = int(float(coordinates[3]))
            # print("bbObject: ", bbObject)
            histogramsCurrent.append(calcHistBoundBox(image[y:y+h, x:x+w]))

        factorGraph.add_nodes_from(nodes)

        matrixSize = int(boundingBoxNumberPrev)+1
        nodesPossibilityMatrix = np.ones((matrixSize, matrixSize))

        for k in range(matrixSize):
            for l in range(matrixSize):
                if k == l:
                    nodesPossibilityMatrix[k][l] = 0
        nodesPossibilityMatrix[0][0] = 1

        if boundingBoxNumberPrev != None or boundingBoxNumberPrev != 0:
            comp = compareHistBoundBox(histogramsCurrent, histogramsPrev, factorGraph)
            # print(comp)

            # for current_histrogram1, current_histrogram2 in combinations(range(int(boundingBoxNumber)), 2):
            #     tmp = DiscreteFactor([str(current_histrogram1), str(current_histrogram2)], [matrixSize,
            #                                                                             matrixSize],
            #                                                                             nodesPossibilityMatrix)
            #     factorGraph.add_factors(tmp)
            #     factorGraph.add_edge(str(current_histrogram1), tmp)
            #     factorGraph.add_edge(str(current_histrogram2), tmp)

            # # for i in range(len(matrixSize)):
            # #     for j in range(len(matrixSize)):

            # BP = BeliefPropagation(factorGraph)
            # BP.calibrate()

            # pre_result = (BP.map_query(factorGraph.get_variable_nodes(),show_progress=False))
            # pre_result2 = OrderedDict(sorted(pre_result.items()))
            # result = list(pre_result2.values())
            # final_result = []
            # for i in range(len(result)):
            #     value = result[i] - 1
            #     final_result.append(value)
            # print(*final_result,sep = ' ')

        


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